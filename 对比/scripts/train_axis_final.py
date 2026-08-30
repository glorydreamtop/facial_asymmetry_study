#!/usr/bin/env python3
"""Final axis-model comparison (pred features, tuned hyperparameters).

Final tuned configs on coords_geom features:
  - Ridge:   alpha=1000
  - SVR:     C=10, gamma=scale, epsilon=0.01
  - RF:      n_estimators=500, max_depth=None, min_samples_leaf=2
  - Linear:  no params
Saves test sample metrics + metrics_summary per model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "对比/scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "对比/scripts"))

from train_axis_compare_pred import build_features, evaluate_dataset, load_axis_pairs, load_hrnet
from src.symmetry.axis_regression import (
    LinearAxisRegressor, OrdinaryLinearAxisRegressor, RandomForestAxisRegressor, SVRAxisRegressor,
)


def build_regressor(model_type: str, seed: int):
    if model_type == "ridge":
        return LinearAxisRegressor(alpha=1000.0)
    if model_type == "linear":
        return OrdinaryLinearAxisRegressor()
    if model_type == "svr":
        return SVRAxisRegressor(C=10.0, epsilon=0.01, gamma="scale")
    if model_type == "rf":
        return RandomForestAxisRegressor(n_estimators=500, random_state=seed)
    raise ValueError(model_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis_dirs", nargs="+", required=True)
    parser.add_argument("--model_type", choices=["ridge", "linear", "svr", "rf"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)

    model, input_size = load_hrnet(Path(args.config), Path(args.checkpoint), device)
    pairs = load_axis_pairs([Path(d) for d in args.axis_dirs])
    with open(args.split_file) as f:
        manifest = json.load(f)
    n2p = {p["name"]: p for p in pairs}
    tr = [n2p[n] for n in manifest["train"] if n in n2p]
    va = [n2p[n] for n in manifest["val"] if n in n2p]
    te = [n2p[n] for n in manifest["test"] if n in n2p]

    trainval = build_features(tr + va, model, device, input_size, feature_set="coords_geom")
    test = build_features(te, model, device, input_size, feature_set="coords_geom")
    print(f"trainval={trainval['features'].shape[0]} test={test['features'].shape[0]} feature_set=coords_geom")

    regressor = build_regressor(args.model_type, args.seed)
    regressor.fit(trainval["features"], trainval["targets"])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"model_type": args.model_type, "feature_source": "pred", "feature_set": "coords_geom", "splits": {}}
    for split_name, dataset in [("train", trainval), ("test", test)]:
        summary, sample_metrics = evaluate_dataset(regressor, dataset)
        metrics["splits"][split_name] = summary
        with open(out_dir / f"{split_name}_sample_metrics.json", "w", encoding="utf-8") as f:
            json.dump(sample_metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    regressor.save(out_dir / "axis_model.joblib")

    print("=" * 60)
    print(f"Axis regression (pred, tuned): {args.model_type}")
    print("=" * 60)
    for split_name in ["train", "test"]:
        sm = metrics["splits"][split_name]
        print(f"[{split_name}] n={sm.get('n_samples')} endpoint_mae_px={sm.get('endpoint_mae_px')} angle_mae_deg={sm.get('angle_mae_deg')}")


if __name__ == "__main__":
    main()
