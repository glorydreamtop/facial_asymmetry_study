#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from _bootstrap import setup_project_paths

setup_project_paths()

from pipelines.hrnet_w48.hrnet_w48_pipeline.metrics import compute_masked_nme, decode_heatmaps_local_softargmax
from pipelines.hrnet_w48.hrnet_w48_pipeline.model import build_model
from pipelines.hrnet_w48.hrnet_w48_pipeline.reporting import (
    build_summary,
    save_error_details,
    save_ced_curve,
    save_nme_histogram,
    save_per_point_error_plot,
    save_region_error_plot,
)
from src.data.dataset import normalize_image_tensor
from src.landmark.schema import NUM_POINTS, REGIONS


def draw_overlay(image_rgb, gt, pred, valid, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.imshow(image_rgb)
    ax.set_title(title)
    ax.axis("off")
    ax.scatter(gt[valid, 0], gt[valid, 1], s=30, facecolors="none", edgecolors="#00FF66", linewidths=1.2, label="GT")
    ax.scatter(pred[valid, 0], pred[valid, 1], c="#FF3B30", s=18, marker="x", linewidths=1.0, label="Prediction")
    ax.legend(loc="lower right")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HRNet-W48 heatmap model")
    parser.add_argument("--config", default="pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/evaluation_hrnet_w48")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    eval_cfg = config.get("eval", {})

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        model_cfg["pretrained"] = False
        if model_cfg.get("pretrained_path"):
            pretrained_path = Path(model_cfg["pretrained_path"])
            if not pretrained_path.exists():
                model_cfg["pretrained_path"] = None

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(data_cfg["ann_file"], "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {int(img["id"]): img for img in coco["images"]}
    annotations = coco["annotations"]

    if args.split_file:
        with open(args.split_file, "r", encoding="utf-8") as f:
            split_items = json.load(f)
        allowed_image_ids = {int(item["image_id"]) for item in split_items}
        annotations = [ann for ann in annotations if int(ann["image_id"]) in allowed_image_ids]

    model = build_model(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device_t, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device_t)
    model.eval()

    input_h, input_w = tuple(model_cfg["input_size"])
    ref_indices = tuple(model_cfg.get("nme_reference_indices", [4, 5]))

    all_nme = []
    per_point_errors = [[] for _ in range(NUM_POINTS)]
    sample_records = []
    visualized = 0
    max_vis = int(eval_cfg.get("vis_samples", 12))

    for ann in tqdm(annotations, desc="Evaluating HRNet-W48"):
        img_info = images[int(ann["image_id"])]
        img_path = Path(data_cfg["img_root"]) / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        gt = kp[:, :2]
        valid = (kp[:, 2] > 0).astype(np.float32)

        resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        image_tensor = normalize_image_tensor(image_tensor).unsqueeze(0).to(device_t)

        with torch.no_grad():
            pred_heatmaps = model(image_tensor)
            pred_norm, _ = decode_heatmaps_local_softargmax(pred_heatmaps)

        pred_norm = pred_norm[0].cpu().numpy()
        gt_norm = gt.copy()
        gt_norm[:, 0] /= orig_w
        gt_norm[:, 1] /= orig_h

        nme_arr = compute_masked_nme(
            pred_norm[None, ...],
            gt_norm[None, ...],
            valid[None, ...],
            ref_point_indices=ref_indices,
        )
        if np.isnan(nme_arr[0]):
            continue
        nme = float(nme_arr[0])
        all_nme.append(nme)

        pred_orig = pred_norm.copy()
        pred_orig[:, 0] *= orig_w
        pred_orig[:, 1] *= orig_h

        per_point_error_map = {}
        per_region_error_map = {}
        valid_bool = valid > 0.5
        for i in range(NUM_POINTS):
            if valid_bool[i]:
                error_px = float(np.linalg.norm(pred_orig[i] - gt[i]))
                per_point_errors[i].append(error_px)
                per_point_error_map[str(i)] = error_px

        for region_name, indices in REGIONS.items():
            region_values = [per_point_error_map[str(idx)] for idx in indices if str(idx) in per_point_error_map]
            if region_values:
                per_region_error_map[region_name] = region_values

        sample_records.append(
            {
                "image_id": int(ann["image_id"]),
                "file_name": img_info["file_name"],
                "nme": nme,
                "per_point_error_px": per_point_error_map,
                "per_region_error_px": per_region_error_map,
            }
        )

        if visualized < max_vis:
            fig = draw_overlay(image_rgb, gt, pred_orig, valid_bool, title=f"NME={nme:.4f}")
            fig.savefig(vis_dir / f"{visualized:02d}_{Path(img_info['file_name']).stem}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            visualized += 1

    summary = build_summary(all_nme, per_point_errors, sample_records)
    with open(output_dir / "paper_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_error_details(sample_records, output_dir / "paper_error_details.json")

    result = {
        "mode": "custom_hrnet_w48_heatmap",
        "mean_nme": summary["mean_nme"],
        "median_nme": summary["median_nme"],
        "std_nme": summary["std_nme"],
        "mre_px": summary["mre_px"],
        "std_mre_px": summary["std_mre_px"],
        "sdr_px": summary["sdr_px"],
        "n_points": summary["n_points"],
        "n_samples": summary["n_samples"],
    }
    with open(output_dir / "custom_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    save_ced_curve(all_nme, figures_dir / "custom_ced_curve.png")
    save_nme_histogram(all_nme, figures_dir / "custom_nme_histogram.png")
    save_per_point_error_plot(per_point_errors, figures_dir / "custom_per_point_error.png")
    save_region_error_plot(per_point_errors, figures_dir / "custom_per_region_error.png")

    print("=" * 60)
    print("HRNet-W48 Evaluation Results")
    print("=" * 60)
    print(f"  NME:        {summary['mean_nme']:.4f} +/- {summary['std_nme']:.4f}")
    print(f"  Median NME: {summary['median_nme']:.4f}")
    print(f"  DR@0.05:    {summary['dr_0.05']:.3f}")
    print(f"  DR@0.10:    {summary['dr_0.10']:.3f}")
    print(f"  MRE (px):   {summary['mre_px']:.2f} +/- {summary['std_mre_px']:.2f}")


if __name__ == "__main__":
    main()
