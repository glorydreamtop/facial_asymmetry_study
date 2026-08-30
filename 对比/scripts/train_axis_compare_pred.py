#!/usr/bin/env python3
"""Compare 4 axis models using PREDICTED landmarks (HRNet inference) as features.

This matches the formal 'axis_only' (pred) pipeline: all axis images get HRNet
predictions, then 4 regressors (Ridge/Linear/SVR/RF) are trained on the formal
198/42/43 split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "对比/scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "对比/scripts"))

from models import build_model as build_lm_model
from pipelines.hrnet_w48.hrnet_w48_pipeline.reporting import predict_landmarks
from src.symmetry.axis_regression import (
    LinearAxisRegressor,
    OrdinaryLinearAxisRegressor,
    RandomForestAxisRegressor,
    SVRAxisRegressor,
    axis_points_to_target,
    build_axis_feature_vector,
    compute_sample_normalization,
    parse_axis_labelme_json,
    summarize_axis_predictions,
)


def load_hrnet(config_path: Path, checkpoint_path: Path, device: torch.device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["pretrained"] = False
    cfg["model"]["pretrained_path"] = None
    model = build_lm_model(cfg)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, tuple(cfg["model"]["input_size"])


def load_axis_pairs(axis_dirs: list[Path]) -> list[dict]:
    pairs = []
    for axis_dir in axis_dirs:
        for axis_jp in sorted(axis_dir.rglob("*.json")):
            axis_ann = parse_axis_labelme_json(str(axis_jp))
            if axis_ann is None:
                continue
            # The axis image itself is used for HRNet inference
            image_path = Path(axis_ann.image_path)
            if not image_path.exists():
                continue
            pairs.append({
                "name": f"{axis_dir.name}/{axis_jp.stem.strip()}.JPG",
                "image_path": image_path,
                "axis_points": axis_ann.axis_points.astype(np.float32),
            })
    return pairs


def build_features(pairs: list[dict], model, device, input_size, feature_set: str = "coords") -> dict:
    features, targets, centers, scales, axis_points, names, ipds, ycs = [], [], [], [], [], [], [], []
    for pair in pairs:
        image = cv2.imread(str(pair["image_path"]))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = predict_landmarks(model, image_rgb, device, input_size=input_size).astype(np.float32)
        if np.isnan(landmarks).any():
            continue
        center, scale = compute_sample_normalization(landmarks)
        feature = build_axis_feature_vector(landmarks, center, scale, feature_set=feature_set)
        target = axis_points_to_target(pair["axis_points"], center, scale)
        features.append(feature)
        targets.append(target)
        centers.append(center)
        scales.append(scale)
        axis_points.append(pair["axis_points"])
        names.append(pair["name"])
        ipds.append(float(np.linalg.norm(landmarks[10] - landmarks[11])))
        ycs.append(float((landmarks[10, 1] + landmarks[11, 1]) / 2.0))

    if not features:
        return {"features": np.empty((0, 0), dtype=np.float32), "targets": np.empty((0, 2), dtype=np.float32),
                "centers": np.empty((0, 2), dtype=np.float32), "scales": np.empty((0,), dtype=np.float32),
                "axis_points": np.empty((0, 2, 2), dtype=np.float32), "names": [], "ipd": np.empty((0,), dtype=np.float32),
                "yc": np.empty((0,), dtype=np.float32)}
    return {
        "features": np.stack(features).astype(np.float32),
        "targets": np.stack(targets).astype(np.float32),
        "centers": np.stack(centers).astype(np.float32),
        "scales": np.asarray(scales, dtype=np.float32),
        "axis_points": np.stack(axis_points).astype(np.float32),
        "names": names,
        "ipd": np.asarray(ipds, dtype=np.float32),
        "yc": np.asarray(ycs, dtype=np.float32),
    }


def build_regressor(model_type: str, seed: int):
    if model_type == "ridge":
        return LinearAxisRegressor(alpha=300.0)  # CV-selected alpha
    if model_type == "linear":
        return OrdinaryLinearAxisRegressor()
    if model_type == "svr":
        return SVRAxisRegressor(C=1.0, epsilon=0.01, gamma="scale")
    if model_type == "rf":
        return RandomForestAxisRegressor(n_estimators=200, random_state=seed)
    raise ValueError(model_type)


def evaluate_dataset(regressor, dataset: dict) -> tuple[dict, list[dict]]:
    pred_targets = regressor.predict_targets(dataset["features"])
    summary, sample_metrics = summarize_axis_predictions(
        pred_targets=pred_targets, gt_targets=dataset["targets"], centers=dataset["centers"],
        scales=dataset["scales"], axis_points=dataset["axis_points"], file_names=dataset["names"])
    ipd = dataset.get("ipd", np.ones(len(pred_targets), dtype=np.float32))
    yc = dataset.get("yc", np.zeros(len(pred_targets), dtype=np.float32))
    for idx, rec in enumerate(sample_metrics):
        rec["signed_angle_diff_deg"] = float(np.degrees(np.atan(rec["pred_slope_xy"])) - np.degrees(np.atan(rec["gt_slope_xy"])))
        rec["signed_position_diff_px"] = float(rec["pred_intercept_x"] - rec["gt_intercept_x"])
        rec["ipd_px"] = float(ipd[idx])
        rec["yc"] = float(yc[idx])
        rec["gt_axis_points"] = dataset["axis_points"][idx].tolist()
    return summary, sample_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis_dirs", nargs="+", required=True)
    parser.add_argument("--model_type", choices=["ridge", "linear", "svr", "rf"], required=True)
    parser.add_argument("--config", required=True, help="HRNet config yaml")
    parser.add_argument("--checkpoint", required=True, help="HRNet checkpoint pth")
    parser.add_argument("--split_file", required=True, help="formal pred split manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)
    print(f"Device: {device}")

    model, input_size = load_hrnet(Path(args.config), Path(args.checkpoint), device)
    print(f"HRNet loaded, input_size={input_size}")

    pairs = load_axis_pairs([Path(d) for d in args.axis_dirs])
    print(f"Axis pairs: {len(pairs)}")

    with open(args.split_file) as f:
        manifest = json.load(f)
    name_to_pair = {p["name"]: p for p in pairs}
    train_pairs = [name_to_pair[n] for n in manifest["train"] if n in name_to_pair]
    val_pairs = [name_to_pair[n] for n in manifest["val"] if n in name_to_pair]
    test_pairs = [name_to_pair[n] for n in manifest["test"] if n in name_to_pair]
    print(f"Split: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    train_set = build_features(train_pairs, model, device, input_size)
    val_set = build_features(val_pairs, model, device, input_size)
    test_set = build_features(test_pairs, model, device, input_size)
    trainval_pairs = train_pairs + val_pairs
    trainval_set = build_features(trainval_pairs, model, device, input_size)
    print(f"Features: train={train_set['features'].shape[0]} trainval={trainval_set['features'].shape[0]} test={test_set['features'].shape[0]}")

    regressor = build_regressor(args.model_type, args.seed)
    regressor.fit(trainval_set["features"], trainval_set["targets"])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"model_type": args.model_type, "feature_source": "pred", "splits": {}}
    for split_name, dataset in [("train", train_set), ("val", val_set), ("trainval", trainval_set), ("test", test_set)]:
        summary, sample_metrics = evaluate_dataset(regressor, dataset)
        metrics["splits"][split_name] = summary
        with open(out_dir / f"{split_name}_sample_metrics.json", "w", encoding="utf-8") as f:
            json.dump(sample_metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    regressor.save(out_dir / "axis_model.joblib")

    print("=" * 60)
    print(f"Axis regression (pred features): {args.model_type}")
    print("=" * 60)
    for split_name in ["train", "val", "trainval", "test"]:
        sm = metrics["splits"][split_name]
        print(f"[{split_name}] n={sm.get('n_samples')} endpoint_mae_px={sm.get('endpoint_mae_px')} angle_mae_deg={sm.get('angle_mae_deg')}")


if __name__ == "__main__":
    main()
