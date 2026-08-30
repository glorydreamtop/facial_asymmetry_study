#!/usr/bin/env python3
"""Evaluate a comparison landmark model on the test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "对比/scripts"), str(REPO_ROOT / "pipelines/hrnet_w48")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import build_model
from pipelines.hrnet_w48.hrnet_w48_pipeline.metrics import compute_masked_nme, decode_heatmaps_local_softargmax
from pipelines.hrnet_w48.hrnet_w48_pipeline.reporting import build_summary, save_error_details
from src.data.dataset import normalize_image_tensor
from src.landmark.schema import NUM_POINTS, REGIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate comparison landmark model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device_t = torch.device(device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_cfg["ann_file"], "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {int(img["id"]): img for img in coco["images"]}
    annotations = coco["annotations"]

    with open(Path(data_cfg["split_dir"]) / "test.json", "r", encoding="utf-8") as f:
        split_items = json.load(f)
    allowed_image_ids = {int(item["image_id"]) for item in split_items}
    annotations = [ann for ann in annotations if int(ann["image_id"]) in allowed_image_ids]

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device_t, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device_t)
    model.eval()

    input_h, input_w = tuple(model_cfg["input_size"])
    ref_indices = tuple(model_cfg.get("nme_reference_indices", [4, 5]))

    all_nme = []
    per_point_errors = [[] for _ in range(NUM_POINTS)]
    sample_records = []

    for ann in tqdm(annotations, desc=f"Evaluating {model_cfg.get('name', 'model')}"):
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

        nme_arr = compute_masked_nme(pred_norm[None, ...], gt_norm[None, ...], valid[None, ...], ref_point_indices=ref_indices)
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

        sample_records.append({
            "image_id": int(ann["image_id"]),
            "file_name": img_info["file_name"],
            "nme": nme,
            "per_point_error_px": per_point_error_map,
            "per_region_error_px": per_region_error_map,
        })

    summary = build_summary(all_nme, per_point_errors, sample_records)
    with open(output_dir / "paper_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_error_details(sample_records, output_dir / "paper_error_details.json")

    result = {
        "mode": model_cfg.get("name", "comparison"),
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

    print("=" * 60)
    print(f"Evaluation: {model_cfg.get('name', 'model')}")
    print("=" * 60)
    print(f"  NME:        {summary['mean_nme']:.4f} +/- {summary['std_nme']:.4f}")
    print(f"  Median NME: {summary['median_nme']:.4f}")
    print(f"  MRE (px):   {summary['mre_px']:.2f} +/- {summary['std_mre_px']:.2f}")
    print(f"  SDR@15:     {summary['sdr_px'].get('15'):.3f}")
    print(f"  N samples:  {summary['n_samples']}")


if __name__ == "__main__":
    main()
