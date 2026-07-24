from __future__ import annotations

import os
import random
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_nme(
    pred: np.ndarray,
    target: np.ndarray,
    inter_ocular: np.ndarray | None = None,
) -> np.ndarray:
    """Normalised Mean Error per sample.

    pred, target: (B, 31, 2)
    inter_ocular: (B,) — if None, uses inter-ocular distance from target.
    Returns (B,) array of NME values.
    """
    diffs = np.linalg.norm(pred - target, axis=2)
    if inter_ocular is None:
        left_eye = target[:, 4, :]
        right_eye = target[:, 5, :]
        ioc = np.linalg.norm(left_eye - right_eye, axis=1) + 1e-10
    else:
        ioc = inter_ocular + 1e-10
    return diffs.mean(axis=1) / ioc


def detection_rate_at_threshold(
    nme: np.ndarray, thresholds: list[float] | None = None
) -> dict[float, float]:
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    return {t: float((nme < t).mean()) for t in thresholds}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: dict[str, Any] | None = None,
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        cfg = config or {}

        self.num_epochs = cfg.get("epochs", cfg.get("num_epochs", 100))
        self.warmup_epochs = cfg.get("warmup_epochs", 5)
        self.patience = cfg.get("early_stop_patience", cfg.get("patience", 15))
        self.log_interval = cfg.get("log_interval", 50)
        self.save_dir = save_dir if save_dir else cfg.get("save_dir", "checkpoints")
        self.exp_name = cfg.get("exp_name", "landmark")

        os.makedirs(self.save_dir, exist_ok=True)
        log_path = os.path.join(log_dir, self.exp_name) if log_dir else os.path.join("runs", self.exp_name)
        self.writer = SummaryWriter(log_dir=log_path)

        # LR scheduler: linear warmup → cosine annealing
        if self.warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1.0 / self.warmup_epochs,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs - self.warmup_epochs,
            )
            self.scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                optimizer, T_max=self.num_epochs
            )

        self.best_nme = float("inf")
        self.best_epoch = -1
        self.epochs_no_improve = 0
        self.start_epoch = 0

    def _capture_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if (
            self.device.type == "mps"
            and hasattr(torch, "mps")
            and torch.backends.mps.is_available()
        ):
            state["mps"] = torch.mps.get_rng_state()
        return state

    def _restore_rng_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "python" in state:
            random.setstate(state["python"])
        if self.device.type == "cuda" and "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        if (
            self.device.type == "mps"
            and "mps" in state
            and hasattr(torch, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.set_rng_state(state["mps"])

    def resume_from_checkpoint(self, path: str) -> dict[str, Any]:
        ckpt = self.load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.best_nme = ckpt.get("best_nme", self.best_nme)
        self.best_epoch = ckpt.get("best_epoch", self.best_epoch)
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        self.start_epoch = ckpt.get("epoch", -1) + 1
        # Restoring RNG keeps shuffle, augmentation, and dropout aligned as
        # closely as possible with an uninterrupted run.
        self._restore_rng_state(ckpt.get("rng_state"))
        return ckpt

    def _to_device(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        images = batch["image"].to(self.device)
        landmarks = batch["landmarks"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        return images, landmarks, mask

    def _run_epoch(
        self, loader: DataLoader, epoch: int, training: bool
    ) -> dict[str, float]:
        if training:
            self.model.train()
            mode = "train"
        else:
            self.model.eval()
            mode = "val"

        loss_meter = AverageMeter()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        loader_iter = tqdm(loader, desc=f"{mode} E{epoch}", leave=False)

        for step, batch in enumerate(loader_iter):
            images, landmarks, mask = self._to_device(batch)
            B = images.size(0)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                output = self.model(images)
                loss = self.criterion(output, landmarks, mask, self.model if training else None)

            if training:
                loss.backward()
                self.optimizer.step()

            loss_meter.update(loss.item(), B)

            pred_np = output.detach().cpu().numpy().reshape(B, -1, 2)
            target_np = landmarks.cpu().numpy().reshape(B, -1, 2)
            all_preds.append(pred_np)
            all_targets.append(target_np)

            if training and step % self.log_interval == 0:
                self.writer.add_scalar(
                    f"loss/{mode}_step", loss.item(), epoch * len(loader) + step
                )

            loader_iter.set_postfix(loss=loss_meter.avg)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        nme = compute_nme(all_preds, all_targets)
        mean_nme = float(nme.mean())
        det_rates = detection_rate_at_threshold(nme)

        metrics: dict[str, float] = {
            f"{mode}_loss": loss_meter.avg,
            f"{mode}_nme": mean_nme,
        }
        for thr, rate in det_rates.items():
            metrics[f"{mode}_dr@{thr:.2f}"] = rate

        self.writer.add_scalar(f"loss/{mode}", loss_meter.avg, epoch)
        self.writer.add_scalar(f"nme/{mode}", mean_nme, epoch)

        return metrics

    def train(self) -> nn.Module:
        for epoch in range(self.start_epoch, self.num_epochs):
            train_metrics = self._run_epoch(self.train_loader, epoch, training=True)
            val_metrics = self._run_epoch(self.val_loader, epoch, training=False)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", current_lr, epoch)

            val_nme = val_metrics["val_nme"]
            val_loss = val_metrics["val_loss"]
            train_nme = train_metrics["train_nme"]

            tqdm.write(
                f"E{epoch:03d}  "
                f"loss_t={train_metrics['train_loss']:.4f}  nme_t={train_nme:.4f}  "
                f"loss_v={val_loss:.4f}  nme_v={val_nme:.4f}  "
                f"lr={current_lr:.2e}"
            )

            # checkpoint by best NME
            if val_nme < self.best_nme:
                self.best_nme = val_nme
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_nme, kind="best")
            else:
                self.epochs_no_improve += 1

            # periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_nme, kind="epoch")

            self._save_checkpoint(epoch, val_nme, kind="last")

            # early stopping
            if self.epochs_no_improve >= self.patience:
                tqdm.write(
                    f"Early stopping at epoch {epoch}. "
                    f"Best NME {self.best_nme:.4f} at epoch {self.best_epoch}."
                )
                break

        self.writer.close()
        return self.model

    def _save_checkpoint(self, epoch: int, val_nme: float, kind: str = "epoch") -> str:
        ckpt = OrderedDict(
            epoch=epoch,
            val_nme=val_nme,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            best_nme=self.best_nme,
            best_epoch=self.best_epoch,
            epochs_no_improve=self.epochs_no_improve,
            rng_state=self._capture_rng_state(),
        )
        if kind == "best":
            path = os.path.join(self.save_dir, f"{self.exp_name}_best.pth")
        elif kind == "last":
            path = os.path.join(self.save_dir, f"{self.exp_name}_last.pth")
        else:
            path = os.path.join(self.save_dir, f"{self.exp_name}_epoch{epoch:04d}.pth")
        tmp_path = f"{path}.tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        return path

    @staticmethod
    def load_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt
