#!/usr/bin/env python3
"""
Phase 2 — Model Evaluation

Evaluate a trained model on the test set and produce detailed accuracy report.

Usage:
    # Evaluate custom model
    python scripts/05_evaluate.py --mode custom \
        --checkpoint checkpoints/best_model.pth \
        --ann_file data/annotations/annotations.json \
        --img_root data/raw \
        --output results/evaluation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import normalize_image_tensor
from src.landmark.model import build_model
from src.landmark.schema import NUM_POINTS, POINT_NAMES_31, REGIONS
from src.landmark.trainer import compute_nme


PIXEL_SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]


def compute_pixel_metrics(per_point_errors, thresholds=PIXEL_SDR_THRESHOLDS):
    all_errors = []
    for errors in per_point_errors:
        all_errors.extend(errors)

    if not all_errors:
        return None

    all_errors = np.asarray(all_errors, dtype=np.float32)
    return {
        "mre_px": float(np.mean(all_errors)),
        "std_px": float(np.std(all_errors)),
        "sdr_px": {str(th): float(np.mean(all_errors <= th)) for th in thresholds},
        "n_points": int(len(all_errors)),
    }


def save_ced_curve(all_nme, save_path: Path):
    if not all_nme:
        return

    nme = np.sort(np.asarray(all_nme, dtype=np.float32))
    y = np.arange(1, len(nme) + 1) / len(nme)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nme, y, color="#1f77b4", linewidth=2)
    ax.set_xlabel("NME")
    ax.set_ylabel("Cumulative sample ratio")
    ax.set_title("CED Curve")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(0.1, float(nme.max()) * 1.05))
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_nme_histogram(all_nme, save_path: Path):
    if not all_nme:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.asarray(all_nme, dtype=np.float32), bins=15, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("NME")
    ax.set_ylabel("Sample count")
    ax.set_title("Test-set NME Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_point_error_plot(per_point_errors, save_path: Path):
    means = []
    stds = []
    labels = []
    for name, errors in zip(POINT_NAMES_31, per_point_errors):
        if errors:
            means.append(float(np.mean(errors)))
            stds.append(float(np.std(errors)))
        else:
            means.append(0.0)
            stds.append(0.0)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color="#72B7B2", ecolor="#1B4332", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Mean error (pixels)")
    ax.set_title("Per-Landmark Error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_region_error_plot(per_point_errors, save_path: Path):
    region_names = []
    region_means = []
    region_stds = []

    for region, indices in REGIONS.items():
        region_values = []
        for idx in indices:
            region_values.extend(per_point_errors[idx])
        if not region_values:
            continue
        region_names.append(region)
        region_means.append(float(np.mean(region_values)))
        region_stds.append(float(np.std(region_values)))

    if not region_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(region_names))
    ax.bar(x, region_means, yerr=region_stds, color="#F58518", ecolor="#7F2704", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(region_names)
    ax.set_ylabel("Mean error (pixels)")
    ax.set_title("Regional Landmark Error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metric_tables(all_nme, per_point_errors, sample_records, output_dir: Path):
    pixel_metrics = compute_pixel_metrics(per_point_errors)
    region_stats = {}
    for region, indices in REGIONS.items():
        region_values = []
        for idx in indices:
            region_values.extend(per_point_errors[idx])
        if region_values:
            region_stats[region] = {
                "mean_error_px": float(np.mean(region_values)),
                "std_error_px": float(np.std(region_values)),
                "count": len(region_values),
            }

    per_point_stats = {}
    for name, errors in zip(POINT_NAMES_31, per_point_errors):
        per_point_stats[name] = {
            "mean_error_px": float(np.mean(errors)) if errors else None,
            "std_error_px": float(np.std(errors)) if errors else None,
            "count": len(errors),
        }

    sorted_samples = sorted(sample_records, key=lambda item: item["nme"])
    summary = {
        "mean_nme": float(np.mean(all_nme)) if all_nme else None,
        "median_nme": float(np.median(all_nme)) if all_nme else None,
        "std_nme": float(np.std(all_nme)) if all_nme else None,
        "dr_0.05": float(np.mean(np.array(all_nme) < 0.05)) if all_nme else None,
        "dr_0.10": float(np.mean(np.array(all_nme) < 0.10)) if all_nme else None,
        "mre_px": pixel_metrics["mre_px"] if pixel_metrics else None,
        "std_mre_px": pixel_metrics["std_px"] if pixel_metrics else None,
        "sdr_px": pixel_metrics["sdr_px"] if pixel_metrics else None,
        "n_points": pixel_metrics["n_points"] if pixel_metrics else None,
        "n_samples": len(all_nme),
        "per_point_error_px": per_point_stats,
        "per_region_error_px": region_stats,
        "best_samples": sorted_samples[:10],
        "worst_samples": sorted_samples[-10:],
    }

    with open(output_dir / "paper_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def evaluate_custom_model(
    model,
    annotations,
    images,
    img_root,
    device,
    input_size=(256, 256),
    vis_dir: Path | None = None,
    max_vis: int = 12,
):
    all_nme = []
    per_point_errors = [[] for _ in range(NUM_POINTS)]
    sample_records = []
    visualized = 0

    model.eval()

    for ann in tqdm(annotations, desc="Evaluating custom model"):
        img_info = images[ann["image_id"]]
        img_path = Path(img_root) / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        gt = kp[:, :2]
        valid = kp[:, 2] > 0

        resized = cv2.resize(image_rgb, (input_size[1], input_size[0]))
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        image_tensor = normalize_image_tensor(image_tensor).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Predict
        output = model(image_tensor).cpu().numpy()[0]  # (62,)
        pred_norm = output.reshape(-1, 2)  # (31, 2) in normalized [0,1] coords
        gt_norm = gt.copy()
        gt_norm[:, 0] /= orig_w
        gt_norm[:, 1] /= orig_h

        # Map back to original image coords
        pred_orig = pred_norm.copy()
        pred_orig[:, 0] *= orig_w
        pred_orig[:, 1] *= orig_h

        # NME
        errors = np.sqrt(np.sum((pred_orig - gt) ** 2, axis=1))
        valid_errors = errors[valid]
        if len(valid_errors) == 0:
            continue

        nme = float(compute_nme(pred_norm[None, ...], gt_norm[None, ...])[0])
        all_nme.append(nme)
        sample_records.append(
            {
                "image_id": int(ann["image_id"]),
                "file_name": img_info["file_name"],
                "nme": nme,
            }
        )

        for i in range(NUM_POINTS):
            if valid[i]:
                per_point_errors[i].append(errors[i])

        if vis_dir is not None and visualized < max_vis:
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            ax.imshow(image_rgb)
            ax.set_title(f"GT vs Pred | NME={nme:.4f}")
            ax.axis("off")

            ax.scatter(
                gt[valid, 0],
                gt[valid, 1],
                s=36,
                facecolors="none",
                edgecolors="#00FF66",
                linewidths=1.4,
                label="GT",
            )
            ax.scatter(
                pred_orig[valid, 0],
                pred_orig[valid, 1],
                c="#FF3B30",
                s=22,
                marker="x",
                linewidths=1.2,
                zorder=6,
                label="Prediction",
            )

            ax.legend(loc="lower right")
            save_path = vis_dir / f"{visualized:02d}_{Path(img_info['file_name']).stem}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            visualized += 1

    return all_nme, per_point_errors, sample_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate landmark detection model")
    parser.add_argument("--mode", choices=["custom"], default="custom")
    parser.add_argument("--checkpoint", default=None, help="Custom model checkpoint")
    parser.add_argument("--ann_file", default="data/annotations/annotations_usable.json")
    parser.add_argument("--img_root", default="data/raw")
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone", default="mobilenet_v2", choices=["mobilenet_v2"])
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 256], help="Model input size as H W")
    parser.add_argument("--vis_samples", type=int, default=12, help="Number of overlay visualizations to export")
    parser.add_argument("--split_file", default=None, help="Optional split JSON file to restrict evaluation samples")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.ann_file, "r") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    if args.split_file is not None:
        with open(args.split_file, "r", encoding="utf-8") as f:
            split_items = json.load(f)
        allowed_image_ids = {item["image_id"] for item in split_items}
        annotations = [ann for ann in annotations if ann["image_id"] in allowed_image_ids]

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for custom mode")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

    model = build_model(backbone=args.backbone, num_points=NUM_POINTS, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    all_nme, per_point_errors, sample_records = evaluate_custom_model(
        model,
        annotations,
        images,
        args.img_root,
        device,
        input_size=tuple(args.input_size),
        vis_dir=vis_dir,
        max_vis=args.vis_samples,
    )

    # Results
    print(f"\n{'='*60}")
    print(f"Evaluation Results — {args.mode.upper()}")
    print(f"{'='*60}")

    all_nme = [n for n in all_nme if n != float("inf")]
    pixel_metrics = compute_pixel_metrics(per_point_errors)
    if all_nme:
        print(f"  NME:           {np.mean(all_nme):.4f} ± {np.std(all_nme):.4f}")
        print(f"  Median NME:    {np.median(all_nme):.4f}")
        print(f"  DR@0.05:       {np.mean(np.array(all_nme) < 0.05):.3f}")
        print(f"  DR@0.10:       {np.mean(np.array(all_nme) < 0.10):.3f}")

    if pixel_metrics is not None:
        print(f"  MRE (px):      {pixel_metrics['mre_px']:.2f} ± {pixel_metrics['std_px']:.2f}")
        for th in PIXEL_SDR_THRESHOLDS:
            print(f"  SDR@{th}px:     {pixel_metrics['sdr_px'][str(th)]:.3f}")

    # Per-point
    print(f"\n{'='*60}")
    print(f"Per-Point Error (pixels)")
    print(f"{'='*60}")
    print(f"{'Point':<20} {'Mean':>8} {'Std':>8} {'Count':>8}")
    print(f"{'-'*48}")
    for i in range(NUM_POINTS):
        if per_point_errors[i]:
            mean_err = np.mean(per_point_errors[i])
            std_err = np.std(per_point_errors[i])
            print(f"{POINT_NAMES_31[i]:<20} {mean_err:>8.2f} {std_err:>8.2f} {len(per_point_errors[i]):>8}")

    # Save
    results = {
        "mode": args.mode,
        "mean_nme": float(np.mean(all_nme)) if all_nme else None,
        "median_nme": float(np.median(all_nme)) if all_nme else None,
        "std_nme": float(np.std(all_nme)) if all_nme else None,
        "mre_px": pixel_metrics["mre_px"] if pixel_metrics else None,
        "std_mre_px": pixel_metrics["std_px"] if pixel_metrics else None,
        "sdr_px": pixel_metrics["sdr_px"] if pixel_metrics else None,
        "n_points": pixel_metrics["n_points"] if pixel_metrics else None,
        "n_samples": len(all_nme),
    }
    result_path = output_dir / f"{args.mode}_eval_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_ced_curve(all_nme, figures_dir / f"{args.mode}_ced_curve.png")
    save_nme_histogram(all_nme, figures_dir / f"{args.mode}_nme_histogram.png")
    save_per_point_error_plot(per_point_errors, figures_dir / f"{args.mode}_per_point_error.png")
    save_region_error_plot(per_point_errors, figures_dir / f"{args.mode}_per_region_error.png")
    save_metric_tables(all_nme, per_point_errors, sample_records, output_dir)
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
