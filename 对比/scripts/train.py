#!/usr/bin/env python3
"""Train a comparison landmark model (HRNet-W18 / ResNet50-SB / MobileNetV3-S)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "对比/scripts"), str(REPO_ROOT / "pipelines/hrnet_w48")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import build_model
from pipelines.hrnet_w48.hrnet_w48_pipeline.dataset import HRNetHeatmapDataset
from pipelines.hrnet_w48.hrnet_w48_pipeline.loss import CombinedHeatmapLoss
from pipelines.hrnet_w48.hrnet_w48_pipeline.trainer import HRNetHeatmapTrainer


def load_split_indices(split_path: str) -> list[int]:
    with open(split_path, "r", encoding="utf-8") as f:
        split_items = json.load(f)
    return [item["idx"] for item in split_items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train comparison landmark model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["train"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    data_cfg = config["data"]
    augment_cfg = config["augment"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device_t = torch.device(device)

    input_size = tuple(model_cfg["input_size"])
    heatmap_size = tuple(model_cfg["heatmap_size"])

    train_dataset = HRNetHeatmapDataset(
        ann_file=data_cfg["ann_file"],
        img_root=data_cfg["img_root"],
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma=float(loss_cfg["sigma"]),
        augment=HRNetHeatmapDataset.default_train_augment(augment_cfg),
    )
    val_dataset = HRNetHeatmapDataset(
        ann_file=data_cfg["ann_file"],
        img_root=data_cfg["img_root"],
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma=float(loss_cfg["sigma"]),
        augment=None,
    )

    split_dir = Path(data_cfg["split_dir"])
    train_idx = load_split_indices(str(split_dir / "train.json"))
    val_idx = load_split_indices(str(split_dir / "val.json"))
    dataset_train = torch.utils.data.Subset(train_dataset, train_idx)
    dataset_val = torch.utils.data.Subset(val_dataset, val_idx)

    num_workers = int(train_cfg.get("num_workers", 8))
    pin_memory = device_t.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(train_cfg["batch_size"]), shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(train_cfg["batch_size"]), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    model = build_model(config).to(device_t)
    criterion = CombinedHeatmapLoss(
        heatmap_weight=float(loss_cfg.get("heatmap_weight", 1.0)),
        coord_weight=float(loss_cfg.get("coord_weight", 0.05)),
        wing_w=float(loss_cfg.get("wing_w", 10.0)),
        wing_epsilon=float(loss_cfg.get("wing_epsilon", 2.0)),
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    trainer = HRNetHeatmapTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device_t,
        config=train_cfg,
        save_dir=str(Path(REPO_ROOT) / train_cfg["save_dir"]),
        log_dir=str(Path(REPO_ROOT) / train_cfg["log_dir"]),
        ref_point_indices=tuple(model_cfg.get("nme_reference_indices", [4, 5])),
    )

    print("=" * 60)
    print(f"Training comparison model: {model_cfg['name']}")
    print("=" * 60)
    print(f"  Backbone:     {model_cfg['backbone']}")
    print(f"  Type:         {model_cfg.get('model_type')}")
    print(f"  Device:       {device_t}")
    print(f"  Params:       {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Input:        {input_size}")
    print(f"  Heatmap:      {heatmap_size}")
    print(f"  Batch:        {train_cfg['batch_size']}")
    print(f"  Epochs:       {train_cfg['epochs']}")
    print(f"  LR:           {train_cfg['lr']}")

    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
