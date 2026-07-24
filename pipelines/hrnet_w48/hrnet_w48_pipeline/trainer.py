from __future__ import annotations

import os
import random
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import compute_masked_nme, decode_heatmaps_local_softargmax


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class HRNetHeatmapTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device: torch.device,
        config: dict[str, Any],
        save_dir: str,
        log_dir: str,
        ref_point_indices: tuple[int, int],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.ref_point_indices = ref_point_indices

        self.num_epochs = int(config.get("epochs", 100))
        self.warmup_epochs = int(config.get("warmup_epochs", 5))
        self.patience = int(config.get("early_stop_patience", 20))
        self.log_interval = int(config.get("log_interval", 50))
        self.freeze_low_level_epochs = int(config.get("freeze_low_level_epochs", 0))
        self.amp = bool(config.get("amp", False)) and device.type == "cuda"
        self.grad_clip_norm = float(config.get("grad_clip_norm", 0.0))
        self.exp_name = str(config.get("exp_name", "hrnet_w48"))
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        log_path = os.path.join(log_dir, self.exp_name) if log_dir else os.path.join("runs", self.exp_name)
        self.writer = SummaryWriter(log_dir=log_path)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)

        min_lr = float(config.get("min_lr", 1.0e-6))
        if self.warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=1.0 / self.warmup_epochs, end_factor=1.0, total_iters=self.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, self.num_epochs - self.warmup_epochs), eta_min=min_lr)
            self.scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])
        else:
            self.scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=min_lr)

        self.best_nme = float("inf")
        self.best_epoch = -1
        self.epochs_no_improve = 0
        self.start_epoch = 0

    def _capture_rng_state(self) -> dict[str, Any]:
        state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
        if self.device.type == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
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

    def resume_from_checkpoint(self, path: str) -> dict[str, Any]:
        ckpt = self.load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        self.best_nme = ckpt.get("best_nme", self.best_nme)
        self.best_epoch = ckpt.get("best_epoch", self.best_epoch)
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        self.start_epoch = ckpt.get("epoch", -1) + 1
        self._restore_rng_state(ckpt.get("rng_state"))
        return ckpt

    def _set_epoch_freeze_state(self, epoch: int) -> None:
        freeze = epoch < self.freeze_low_level_epochs
        self.model.set_low_level_frozen(freeze)

    def _run_epoch(self, loader, epoch: int, training: bool) -> dict[str, float]:
        self._set_epoch_freeze_state(epoch)
        if training:
            self.model.train()
            self.model.apply_freeze_state()
            mode = "train"
        else:
            self.model.eval()
            mode = "val"

        loss_meter = AverageMeter()
        heatmap_meter = AverageMeter()
        coord_meter = AverageMeter()
        pred_batches = []
        target_batches = []
        mask_batches = []

        loader_iter = tqdm(loader, desc=f"{mode} E{epoch}", leave=False)
        for step, batch in enumerate(loader_iter):
            images = batch["image"].to(self.device)
            target_heatmaps = batch["heatmaps"].to(self.device)
            target_coords = batch["landmarks"].to(self.device)
            mask = batch["mask"].to(self.device)
            batch_size = images.size(0)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                    pred_heatmaps = self.model(images)
                    loss_dict = self.criterion(pred_heatmaps, target_heatmaps, target_coords, mask)
                    loss = loss_dict["loss"]

            if training:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            loss_meter.update(float(loss.detach().item()), batch_size)
            heatmap_meter.update(float(loss_dict["heatmap_loss"].item()), batch_size)
            coord_meter.update(float(loss_dict["coord_loss"].item()), batch_size)

            pred_coords, _ = decode_heatmaps_local_softargmax(pred_heatmaps.detach())
            pred_batches.append(pred_coords.cpu().numpy())
            target_batches.append(target_coords.detach().cpu().numpy())
            mask_batches.append(mask.detach().cpu().numpy())

            if training and step % self.log_interval == 0:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar(f"loss/{mode}_step", float(loss.detach().item()), global_step)

            loader_iter.set_postfix(loss=loss_meter.avg)

        pred = np.concatenate(pred_batches, axis=0)
        target = np.concatenate(target_batches, axis=0)
        mask = np.concatenate(mask_batches, axis=0)
        nme = compute_masked_nme(pred, target, mask, ref_point_indices=self.ref_point_indices)
        valid_nme = nme[~np.isnan(nme)]
        mean_nme = float(valid_nme.mean()) if len(valid_nme) else float("inf")

        metrics = {
            f"{mode}_loss": loss_meter.avg,
            f"{mode}_heatmap_loss": heatmap_meter.avg,
            f"{mode}_coord_loss": coord_meter.avg,
            f"{mode}_nme": mean_nme,
        }
        self.writer.add_scalar(f"loss/{mode}", loss_meter.avg, epoch)
        self.writer.add_scalar(f"loss/{mode}_heatmap", heatmap_meter.avg, epoch)
        self.writer.add_scalar(f"loss/{mode}_coord", coord_meter.avg, epoch)
        self.writer.add_scalar(f"nme/{mode}", mean_nme, epoch)
        return metrics

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            train_metrics = self._run_epoch(self.train_loader, epoch, training=True)
            val_metrics = self._run_epoch(self.val_loader, epoch, training=False)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", current_lr, epoch)

            val_nme = val_metrics["val_nme"]
            tqdm.write(
                f"E{epoch:03d}  loss_t={train_metrics['train_loss']:.4f}  nme_t={train_metrics['train_nme']:.4f}  "
                f"loss_v={val_metrics['val_loss']:.4f}  nme_v={val_nme:.4f}  lr={current_lr:.2e}"
            )

            if val_nme < self.best_nme:
                self.best_nme = val_nme
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_nme, kind="best")
            else:
                self.epochs_no_improve += 1

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_nme, kind="epoch")
            self._save_checkpoint(epoch, val_nme, kind="last")

            if self.epochs_no_improve >= self.patience:
                tqdm.write(f"Early stopping at epoch {epoch}. Best NME {self.best_nme:.4f} at epoch {self.best_epoch}.")
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
    def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt
