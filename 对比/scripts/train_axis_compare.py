#!/usr/bin/env python3
"""Compare 4 symmetry-axis regression models: Ridge vs Linear vs SVR vs RandomForest.

Uses expert-drawn landmarks (gt features) + expert axis annotations.
Same train/val/test split for all models for a fair comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.parse_labelme import parse_labelme_json
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
from src.landmark.schema import POINT_NAMES_31


def load_pairs(axis_dirs: list[Path], landmark_root: Path) -> list[dict]:
    """Match axis JSONs to landmark JSONs by subject name."""
    lm_by_stem = {}
    for jp in landmark_root.rglob("*.json"):
        ann = parse_labelme_json(str(jp), raw_dir=str(landmark_root))
        if ann is not None and ann["num_valid"] >= 29:
            lm = ann["landmarks"]
            if lm.shape == (29, 2) and not np.isnan(lm).any():
                lm_by_stem[jp.stem.strip()] = jp

    pairs = []
    for axis_dir in axis_dirs:
        for axis_jp in sorted(axis_dir.rglob("*.json")):
            stem = axis_jp.stem.strip()
            if stem not in lm_by_stem:
                continue
            axis_ann = parse_axis_labelme_json(str(axis_jp))
            if axis_ann is None:
                continue
            pairs.append({
                "name": stem,
                "axis_json": axis_jp,
                "axis_points": axis_ann.axis_points.astype(np.float32),
                "landmark_json": lm_by_stem[stem],
            })
    return pairs


def build_features(pairs: list[dict], landmark_root: Path, feature_set: str = "coords") -> dict:
    features = []
    targets = []
    centers = []
    scales = []
    axis_points = []
    names = []
    ipds = []
    ycs = []

    for pair in pairs:
        ann = parse_labelme_json(str(pair["landmark_json"]), raw_dir=str(landmark_root))
        if ann is None or ann["num_valid"] < 29:
            continue
        landmarks = ann["landmarks"].astype(np.float32)
        if landmarks.shape != (29, 2) or np.isnan(landmarks).any():
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
        return LinearAxisRegressor(alpha=1.0)
    if model_type == "linear":
        return OrdinaryLinearAxisRegressor()
    if model_type == "svr":
        return SVRAxisRegressor(C=1.0, epsilon=0.01, gamma="scale")
    if model_type == "rf":
        return RandomForestAxisRegressor(n_estimators=200, random_state=seed)
    raise ValueError(model_type)


def select_ridge_alpha(dataset: dict, names: list[str], alpha_grid: list[float], cv_folds: int, seed: int = 42) -> tuple[float, list[dict]]:
    from sklearn.model_selection import KFold
    n = len(dataset["features"])
    if n < 2:
        return alpha_grid[0], []
    n_splits = min(cv_folds, n)
    if n_splits < 2:
        return alpha_grid[0], []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_results = []
    best_alpha, best_score = alpha_grid[0], None
    for alpha in alpha_grid:
        fold_scores = []
        fold_records = []
        for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(dataset["features"]), start=1):
            reg = LinearAxisRegressor(alpha=alpha)
            reg.fit(dataset["features"][tr_idx], dataset["targets"][tr_idx])
            summary, _ = summarize_axis_predictions(
                pred_targets=reg.predict_targets(dataset["features"][va_idx]),
                gt_targets=dataset["targets"][va_idx],
                centers=dataset["centers"][va_idx],
                scales=dataset["scales"][va_idx],
                axis_points=dataset["axis_points"][va_idx],
                file_names=[names[i] for i in va_idx],
            )
            score = summary["endpoint_mae_norm"]
            fold_scores.append(score)
            fold_records.append({"fold": fold_idx, "n_train": len(tr_idx), "n_val": len(va_idx), "endpoint_mae_norm": score})
        mean_score = float(np.mean(fold_scores))
        cv_results.append({"alpha": alpha, "mean_endpoint_mae_norm": mean_score, "folds": fold_records})
        if best_score is None or mean_score < best_score:
            best_score, best_alpha = mean_score, alpha
    return best_alpha, cv_results


def evaluate_dataset(regressor, dataset: dict) -> tuple[dict, list[dict]]:
    pred_targets = regressor.predict_targets(dataset["features"])
    summary, sample_metrics = summarize_axis_predictions(
        pred_targets=pred_targets,
        gt_targets=dataset["targets"],
        centers=dataset["centers"],
        scales=dataset["scales"],
        axis_points=dataset["axis_points"],
        file_names=dataset["names"],
    )

    # Enrich sample metrics with signed angle/position diffs, IPD, yc
    ipd = dataset.get("ipd", np.ones(len(pred_targets), dtype=np.float32))
    yc = dataset.get("yc", np.zeros(len(pred_targets), dtype=np.float32))
    for idx, rec in enumerate(sample_metrics):
        pred_slope = rec["pred_slope_xy"]
        gt_slope = rec["gt_slope_xy"]
        pred_angle = np.degrees(np.atan(pred_slope))
        gt_angle = np.degrees(np.atan(gt_slope))
        rec["signed_angle_diff_deg"] = float(pred_angle - gt_angle)
        rec["signed_position_diff_px"] = float(rec["pred_intercept_x"] - rec["gt_intercept_x"])
        rec["ipd_px"] = float(ipd[idx])
        rec["yc"] = float(yc[idx])
        rec["gt_axis_points"] = dataset["axis_points"][idx].tolist()
    return summary, sample_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis_dirs", nargs="+", required=True)
    parser.add_argument("--landmark_root", required=True)
    parser.add_argument("--feature_set", choices=["coords", "coords_geom"], default="coords")
    parser.add_argument("--model_type", choices=["ridge", "linear", "svr", "rf"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--split_file", default=None, help="JSON manifest with train/val/test name lists")
    parser.add_argument("--cv_alpha", action="store_true", help="select Ridge alpha via trainval CV")
    args = parser.parse_args()

    axis_dirs = [Path(d) for d in args.axis_dirs]
    landmark_root = Path(args.landmark_root)

    pairs = load_pairs(axis_dirs, landmark_root)
    print(f"Matched axis+landmark pairs: {len(pairs)}")
    if not pairs:
        raise RuntimeError("No matched pairs found")

    # Deterministic split (stratify-free, but stable): 70/10/20
    rng = np.random.RandomState(args.seed)
    names = [p["name"] for p in pairs]
    if args.split_file:
        # Use explicit split manifest (name lists) to match the formal version exactly
        with open(args.split_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        name_to_idx = {n: i for i, n in enumerate(names)}
        train_idx = np.array([name_to_idx[n] for n in manifest["train"]])
        val_idx = np.array([name_to_idx[n] for n in manifest["val"]])
        test_idx = np.array([name_to_idx[n] for n in manifest["test"]])
        print(f"  Using formal split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    else:
        order = rng.permutation(len(pairs))
        n = len(pairs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        print(f"  Random split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]

    train_set = build_features(train_pairs, landmark_root, args.feature_set)
    val_set = build_features(val_pairs, landmark_root, args.feature_set)
    test_set = build_features(test_pairs, landmark_root, args.feature_set)
    trainval_pairs = train_pairs + val_pairs
    trainval_set = build_features(trainval_pairs, landmark_root, args.feature_set)

    print(f"  train: {len(train_set['names'])}  val: {len(val_set['names'])}  test: {len(test_set['names'])}")

    # Select hyperparams via CV on trainval (Ridge alpha, SVR C) — keep simple defaults for baseline models
    alpha_grid = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0]
    cv_results = []
    if args.model_type == "ridge" and args.cv_alpha:
        best_alpha, cv_results = select_ridge_alpha(trainval_set, trainval_set["names"], alpha_grid, args.cv_folds, args.seed)
        print(f"  Ridge CV selected alpha = {best_alpha}")
        regressor = LinearAxisRegressor(alpha=best_alpha)
    else:
        regressor = build_regressor(args.model_type, args.seed)
    regressor.fit(trainval_set["features"], trainval_set["targets"])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "seed": args.seed,
        "cv_results": cv_results,
        "splits": {},
    }

    for split_name, dataset in [("train", train_set), ("val", val_set), ("trainval", trainval_set), ("test", test_set)]:
        if len(dataset["features"]) == 0:
            metrics["splits"][split_name] = {"n_samples": 0}
            continue
        summary, sample_metrics = evaluate_dataset(regressor, dataset)
        metrics["splits"][split_name] = summary
        with open(output_dir / f"{split_name}_sample_metrics.json", "w", encoding="utf-8") as f:
            json.dump(sample_metrics, f, indent=2, ensure_ascii=False)
        with open(output_dir / f"{split_name}_names.json", "w", encoding="utf-8") as f:
            json.dump(dataset["names"], f, indent=2, ensure_ascii=False)

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    regressor.save(output_dir / "axis_model.joblib")

    # Save split indices
    split_manifest = {
        "seed": args.seed,
        "train": [pairs[i]["name"] for i in train_idx],
        "val": [pairs[i]["name"] for i in val_idx],
        "test": [pairs[i]["name"] for i in test_idx],
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Axis Regression: {args.model_type}")
    print("=" * 60)
    for split_name in ["train", "val", "trainval", "test"]:
        sm = metrics["splits"][split_name]
        if sm.get("n_samples", 0) == 0:
            print(f"[{split_name}] n=0")
            continue
        print(f"[{split_name}] n={sm['n_samples']}  endpoint_mae_px={sm.get('endpoint_mae_px')}  angle_mae_deg={sm.get('angle_mae_deg')}")


if __name__ == "__main__":
    main()
