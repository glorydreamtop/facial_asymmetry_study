#!/usr/bin/env python3
"""
Phase 2 — Custom Model Training

Train a landmark detection model on the facial landmark dataset.

Usage:
    # Local test (CPU/MPS)
    python scripts/04_train.py \
        --ann_file data/annotations/annotations_usable.json \
        --img_root data/raw \
        --config pipelines/mobilenetv2/configs/mac_mps_320_all869_e80.yaml

    # Server training (CUDA)
    python scripts/04_train.py \
        --ann_file data/annotations/annotations_usable.json \
        --img_root data/raw \
        --config pipelines/mobilenetv2/configs/mac_mps_320_all869_e80.yaml \
        --device cuda \
        --epochs 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import FaceLandmarkDataset
from src.data.augment import LandmarkAugmentation
from src.landmark.model import build_model
from src.landmark.loss import CombinedLandmarkLoss
from src.landmark.trainer import Trainer


def load_split_indices(split_path: str) -> list[int]:
    with open(split_path, "r", encoding="utf-8") as f:
        split_items = json.load(f)
    return [item["idx"] for item in split_items]


def main():
    parser = argparse.ArgumentParser(description="Train facial landmark model")
    parser.add_argument("--ann_file", default="data/annotations/annotations_usable.json",
                        help="COCO annotation file")
    parser.add_argument("--img_root", default="data/raw",
                        help="Image root directory")
    parser.add_argument("--config", default="configs/mac_mps_320.yaml",
                        help="Configuration file")
    parser.add_argument("--device", default="auto",
                        help="Device: auto/cuda/cpu/mps")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--save_dir", default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--log_dir", default="logs",
                        help="TensorBoard log directory")
    parser.add_argument("--split_dir", default="data/splits",
                        help="Directory containing train/val/test split JSON files")
    args = parser.parse_args()

    if args.resume is not None and not Path(args.resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override from CLI
    if args.epochs is not None:
        config["landmark"]["custom"]["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["landmark"]["custom"]["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["landmark"]["custom"]["train"]["lr"] = args.lr

    train_cfg = config["landmark"]["custom"]["train"]
    model_cfg = config["landmark"]["custom"]
    augment_cfg = config["landmark"]["custom"]["augment"]
    num_points = config["landmark"]["custom"]["num_points"]

    # Device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else \
                 "cpu"

    print("=" * 60)
    print("Phase 2 — Custom Model Training")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Backbone:    {model_cfg['backbone']}")
    print(f"  Epochs:      {train_cfg['epochs']}")
    print(f"  Batch size:  {train_cfg['batch_size']}")
    print(f"  LR:          {train_cfg['lr']}")
    print(f"  Input size:  {model_cfg['input_size']}")

    pin_memory = device == "cuda"

    # Data augmentation
    train_transform = LandmarkAugmentation(
        rotation=augment_cfg["rotation"],
        scale=tuple(augment_cfg["scale"]),
        brightness=augment_cfg["brightness"],
        contrast=augment_cfg["contrast"],
        occlusion_prob=augment_cfg["occlusion"],
    )

    # Datasets
    input_size = tuple(model_cfg["input_size"])  # (H, W)

    train_dataset = FaceLandmarkDataset(
        ann_file=args.ann_file,
        img_root=args.img_root,
        transform=train_transform,
        target_size=input_size,
        normalize=True,
    )

    eval_dataset = FaceLandmarkDataset(
        ann_file=args.ann_file,
        img_root=args.img_root,
        transform=None,
        target_size=input_size,
        normalize=True,
    )

    train_idx = load_split_indices(str(Path(args.split_dir) / "train.json"))
    val_idx = load_split_indices(str(Path(args.split_dir) / "val.json"))
    test_idx = load_split_indices(str(Path(args.split_dir) / "test.json"))

    dataset_train = torch.utils.data.Subset(train_dataset, train_idx)
    dataset_val = torch.utils.data.Subset(eval_dataset, val_idx)
    dataset_test = torch.utils.data.Subset(eval_dataset, test_idx)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=pin_memory,
    )

    print(f"\n  Train samples: {len(dataset_train)}")
    print(f"  Val samples:   {len(dataset_val)}")
    print(f"  Test samples:  {len(dataset_test)}")

    # Model
    model = build_model(
        backbone=model_cfg["backbone"],
        num_points=num_points,
        pretrained=True,
    )
    model = model.to(torch.device(device))

    # Loss
    criterion = CombinedLandmarkLoss(
        w=config["landmark"]["custom"]["wing_w"],
        epsilon=config["landmark"]["custom"]["wing_epsilon"],
        l2_reg=0.0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(device),
        config=train_cfg,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    if args.resume is not None:
        trainer.resume_from_checkpoint(args.resume)
        print(f"  Resume from: {args.resume}")
        print(f"  Start epoch: {trainer.start_epoch}")
        print(f"  Best NME:    {trainer.best_nme:.6f} @ epoch {trainer.best_epoch}")
        if trainer.start_epoch >= train_cfg["epochs"]:
            print("  Resume checkpoint already reached the requested epoch limit.")

    # Train
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
