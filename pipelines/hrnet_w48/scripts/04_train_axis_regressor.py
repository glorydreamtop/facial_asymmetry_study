#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold

from _bootstrap import setup_project_paths

setup_project_paths()

from pipelines.hrnet_w48.hrnet_w48_pipeline.model import build_model
from pipelines.hrnet_w48.hrnet_w48_pipeline.reporting import predict_landmarks
from src.data.parse_labelme import parse_labelme_json
from src.symmetry.axis_regression import (
    LinearAxisRegressor,
    MLPAxisRegressor,
    PLSAxisRegressor,
    PolynomialAxisRegressor,
    axis_points_to_target,
    build_axis_feature_vector,
    compute_sample_normalization,
    make_axis_feature_names,
    normalize_landmarks,
    parse_axis_labelme_json,
    summarize_axis_predictions,
)


def load_split_paths(split_file: str | None) -> set[str] | None:
    if not split_file:
        return None
    with open(split_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {str(item["image_path"]).replace("\\", "/") for item in items}


def resolve_axis_relative_image_path(image_path: Path, axis_dir_path: Path) -> str:
    try:
        return str(image_path.relative_to(axis_dir_path)).replace("\\", "/")
    except ValueError:
        return image_path.name


def find_landmark_match(image_path: Path, landmark_root_path: Path) -> tuple[Path | None, str | None]:
    stem = image_path.stem
    suffix = image_path.suffix or ".jpg"
    for candidate in landmark_root_path.rglob(f"{stem}.json"):
        candidate_image = candidate.with_suffix(suffix)
        if candidate_image.exists():
            try:
                relative_image_path = str(candidate_image.relative_to(landmark_root_path)).replace("\\", "/")
            except ValueError:
                relative_image_path = candidate_image.name
            return candidate, relative_image_path
    return None, None


def load_hrnet_model(config_path: str, checkpoint_path: str, device: torch.device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # When loading a fully trained checkpoint, backbone pretraining weights are no longer needed.
    config["model"] = dict(config["model"])
    config["model"]["pretrained"] = False
    config["model"]["pretrained_path"] = None
    model = build_model(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    input_size = tuple(config["model"]["input_size"])
    return model, input_size


def build_dataset(
    axis_dir: str,
    landmark_root: str,
    split_paths: set[str] | None,
    feature_source: str,
    feature_set: str,
    split_key: str,
    model,
    device: torch.device,
    input_size: tuple[int, int] | None,
) -> tuple[dict[str, np.ndarray], list[dict], list[str]]:
    features = []
    targets = []
    centers = []
    scales = []
    axis_points_list = []
    file_names = []
    records = []

    axis_json_paths = sorted(Path(axis_dir).rglob("*.json"))
    landmark_root_path = Path(landmark_root)
    axis_dir_path = Path(axis_dir)

    for axis_json_path in axis_json_paths:
        axis_ann = parse_axis_labelme_json(str(axis_json_path))
        if axis_ann is None:
            continue

        image_path = Path(axis_ann.image_path)
        axis_relative_image_path = resolve_axis_relative_image_path(image_path, axis_dir_path)
        landmark_json_path, landmark_relative_image_path = find_landmark_match(image_path, landmark_root_path)

        split_value = axis_relative_image_path if split_key == "axis" else landmark_relative_image_path
        if split_paths is not None and split_value not in split_paths:
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if feature_source == "pred":
            if model is None or input_size is None:
                raise ValueError("HRNet model is required for predicted landmarks")
            landmarks = predict_landmarks(model, image_rgb, device, input_size=input_size)
        else:
            if landmark_json_path is None:
                continue
            landmark_ann = parse_labelme_json(str(landmark_json_path), raw_dir=str(landmark_root_path))
            if landmark_ann is None or landmark_ann["num_valid"] < 29:
                continue
            landmarks = landmark_ann["landmarks"].astype(np.float32)

        center, scale = compute_sample_normalization(landmarks)
        feature = build_axis_feature_vector(landmarks, center, scale, feature_set=feature_set)
        target = axis_points_to_target(axis_ann.axis_points, center, scale)

        features.append(feature)
        targets.append(target)
        centers.append(center)
        scales.append(scale)
        axis_points_list.append(axis_ann.axis_points.astype(np.float32))
        sample_key = split_value or axis_relative_image_path
        file_names.append(sample_key)
        records.append(
            {
                "file_name": sample_key,
                "axis_image_path": axis_relative_image_path,
                "image_path": str(image_path),
                "landmark_image_path": landmark_relative_image_path,
                "landmark_json": None if landmark_json_path is None else str(landmark_json_path),
                "axis_json": str(axis_json_path),
            }
        )

    if not features:
        empty = np.empty((0, 0), dtype=np.float32)
        return {
            "features": empty,
            "targets": np.empty((0, 2), dtype=np.float32),
            "centers": np.empty((0, 2), dtype=np.float32),
            "scales": np.empty((0,), dtype=np.float32),
            "axis_points": np.empty((0, 2, 2), dtype=np.float32),
        }, records, file_names

    dataset = {
        "features": np.stack(features).astype(np.float32),
        "targets": np.stack(targets).astype(np.float32),
        "centers": np.stack(centers).astype(np.float32),
        "scales": np.asarray(scales, dtype=np.float32),
        "axis_points": np.stack(axis_points_list).astype(np.float32),
    }
    return dataset, records, file_names


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def concat_datasets(*datasets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    non_empty = [dataset for dataset in datasets if len(dataset["features"]) > 0]
    if not non_empty:
        return {
            "features": np.empty((0, 0), dtype=np.float32),
            "targets": np.empty((0, 2), dtype=np.float32),
            "centers": np.empty((0, 2), dtype=np.float32),
            "scales": np.empty((0,), dtype=np.float32),
            "axis_points": np.empty((0, 2, 2), dtype=np.float32),
        }
    keys = ["features", "targets", "centers", "scales", "axis_points"]
    return {key: np.concatenate([dataset[key] for dataset in non_empty], axis=0).astype(non_empty[0][key].dtype) for key in keys}


def subset_dataset(dataset: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in dataset.items()}


def evaluate_dataset(regressor: LinearAxisRegressor, dataset: dict[str, np.ndarray], names: list[str]) -> tuple[dict, list[dict]]:
    pred_targets = regressor.predict_targets(dataset["features"])
    return summarize_axis_predictions(
        pred_targets=pred_targets,
        gt_targets=dataset["targets"],
        centers=dataset["centers"],
        scales=dataset["scales"],
        axis_points=dataset["axis_points"],
        file_names=names,
    )


def build_regressor(args: argparse.Namespace, selected_value: float | int):
    if args.model_type == "ridge":
        return LinearAxisRegressor(alpha=float(selected_value), solver=args.ridge_solver)
    if args.model_type == "poly_ridge":
        return PolynomialAxisRegressor(alpha=float(selected_value), degree=args.poly_degree, solver=args.ridge_solver)
    if args.model_type == "pls":
        return PLSAxisRegressor(n_components=int(selected_value))
    if args.model_type == "mlp":
        return MLPAxisRegressor(
            hidden_layer_sizes=tuple(args.mlp_hidden_sizes),
            alpha=float(selected_value),
            max_iter=args.mlp_max_iter,
            random_state=args.random_state,
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def get_search_values(args: argparse.Namespace) -> list[float | int]:
    if args.model_type == "pls":
        return [int(value) for value in args.pls_components_grid]
    if args.model_type == "mlp":
        return [float(value) for value in args.mlp_alpha_grid]
    return [float(value) for value in args.alpha_grid]


def select_alpha_via_cv(
    dataset: dict[str, np.ndarray],
    names: list[str],
    search_values: list[float | int],
    cv_folds: int,
    args: argparse.Namespace,
) -> tuple[float, list[dict]]:
    n_samples = len(dataset["features"])
    if n_samples < 2:
        return search_values[0], []

    n_splits = min(cv_folds, n_samples)
    if n_splits < 2:
        return search_values[0], []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []
    best_value = search_values[0]
    best_score = None

    for search_value in search_values:
        fold_scores = []
        fold_records = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset["features"]), start=1):
            train_subset = subset_dataset(dataset, train_idx)
            val_subset = subset_dataset(dataset, val_idx)
            val_names = [names[i] for i in val_idx]

            regressor = build_regressor(args, search_value)
            regressor.fit(train_subset["features"], train_subset["targets"])
            summary, _ = evaluate_dataset(regressor, val_subset, val_names)
            score = summary["endpoint_mae_norm"]
            fold_scores.append(score)
            fold_records.append(
                {
                    "search_value": search_value,
                    "fold": fold_idx,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "endpoint_mae_norm": score,
                    "midpoint_x_mae_norm": summary["midpoint_x_mae_norm"],
                    "angle_mae_deg": summary["angle_mae_deg"],
                }
            )

        mean_score = float(np.mean(fold_scores)) if fold_scores else None
        if args.model_type == "pls":
            result_key = "n_components"
        else:
            result_key = "alpha"
        cv_results.append(
            {
                result_key: search_value,
                "mean_endpoint_mae_norm": mean_score,
                "folds": fold_records,
            }
        )

        if mean_score is not None and (best_score is None or mean_score < best_score):
            best_score = mean_score
            best_value = search_value

    return best_value, cv_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ridge regressor for facial symmetry axis prediction")
    parser.add_argument("--config", default="pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--axis_dir", default="对称轴数据")
    parser.add_argument("--landmark_root", default="data/raw")
    parser.add_argument("--train_split", default="data/splits/train.json")
    parser.add_argument("--val_split", default="data/splits/val.json")
    parser.add_argument("--test_split", default="data/splits/test.json")
    parser.add_argument("--split_key", choices=["landmark", "axis"], default="landmark")
    parser.add_argument("--feature_source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--feature_set", choices=["coords", "coords_geom"], default="coords")
    parser.add_argument("--model_type", choices=["ridge", "poly_ridge", "pls", "mlp"], default="ridge")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=[1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0])
    parser.add_argument("--poly_degree", type=int, default=2)
    parser.add_argument("--pls_components", type=int, default=8)
    parser.add_argument("--pls_components_grid", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--mlp_hidden_sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--mlp_alpha", type=float, default=1.0e-3)
    parser.add_argument("--mlp_alpha_grid", type=float, nargs="+", default=[1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    parser.add_argument("--mlp_max_iter", type=int, default=2000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--selection_metric", choices=["endpoint_mae_norm"], default="endpoint_mae_norm")
    parser.add_argument("--use_trainval_cv", action="store_true")
    parser.add_argument("--ridge_solver", choices=["svd", "cholesky", "lsqr", "auto"], default="svd")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/axis_regression_ridge")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model = None
    input_size = None
    if args.feature_source == "pred":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --feature_source=pred")
        model, input_size = load_hrnet_model(args.config, args.checkpoint, device)

    train_paths = load_split_paths(args.train_split)
    val_paths = load_split_paths(args.val_split)
    test_paths = load_split_paths(args.test_split)

    train_set, train_records, train_names = build_dataset(
        axis_dir=args.axis_dir,
        landmark_root=args.landmark_root,
        split_paths=train_paths,
        feature_source=args.feature_source,
        feature_set=args.feature_set,
        split_key=args.split_key,
        model=model,
        device=device,
        input_size=input_size,
    )
    val_set, val_records, val_names = build_dataset(
        axis_dir=args.axis_dir,
        landmark_root=args.landmark_root,
        split_paths=val_paths,
        feature_source=args.feature_source,
        feature_set=args.feature_set,
        split_key=args.split_key,
        model=model,
        device=device,
        input_size=input_size,
    )
    test_set, test_records, test_names = build_dataset(
        axis_dir=args.axis_dir,
        landmark_root=args.landmark_root,
        split_paths=test_paths,
        feature_source=args.feature_source,
        feature_set=args.feature_set,
        split_key=args.split_key,
        model=model,
        device=device,
        input_size=input_size,
    )

    if len(train_set["features"]) == 0:
        raise RuntimeError("No matched training samples were found between landmark and axis annotations")

    trainval_set = concat_datasets(train_set, val_set)
    trainval_records = train_records + val_records
    trainval_names = train_names + val_names

    if args.model_type == "pls":
        selected_value: float | int = args.pls_components
    elif args.model_type == "mlp":
        selected_value = args.mlp_alpha
    else:
        selected_value = args.alpha
    cv_results = []
    if args.use_trainval_cv:
        selected_value, cv_results = select_alpha_via_cv(
            dataset=trainval_set,
            names=trainval_names,
            search_values=get_search_values(args),
            cv_folds=args.cv_folds,
            args=args,
        )

    fit_set = trainval_set if args.use_trainval_cv else train_set
    regressor = build_regressor(args, selected_value)
    regressor.fit(fit_set["features"], fit_set["targets"])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = make_axis_feature_names(args.feature_set)
    regressor.save(output_dir / "axis_model.joblib")
    write_json(output_dir / "axis_model_equation.json", regressor.export_coefficients(feature_names))
    if args.model_type == "ridge":
        regressor.save(output_dir / "axis_ridge.joblib")
        write_json(output_dir / "axis_ridge_equation.json", regressor.export_coefficients(feature_names))

    dataset_payload = {
        "train": (train_set, train_records, train_names),
        "val": (val_set, val_records, val_names),
        "trainval": (trainval_set, trainval_records, trainval_names),
        "test": (test_set, test_records, test_names),
    }
    metrics = {
        "feature_source": args.feature_source,
        "feature_set": args.feature_set,
        "split_key": args.split_key,
        "model_type": args.model_type,
        "alpha": args.alpha,
        "selected_alpha": float(selected_value) if args.model_type != "pls" else None,
        "pls_components": args.pls_components,
        "selected_pls_components": int(selected_value) if args.model_type == "pls" else None,
        "poly_degree": args.poly_degree,
        "mlp_hidden_sizes": args.mlp_hidden_sizes,
        "mlp_alpha": args.mlp_alpha,
        "selected_mlp_alpha": float(selected_value) if args.model_type == "mlp" else None,
        "mlp_max_iter": args.mlp_max_iter,
        "random_state": args.random_state,
        "ridge_solver": args.ridge_solver,
        "use_trainval_cv": bool(args.use_trainval_cv),
        "cv_folds": args.cv_folds,
        "selection_metric": args.selection_metric,
        "cv_results": cv_results,
        "splits": {},
    }

    for split_name, (dataset, records, names) in dataset_payload.items():
        if len(dataset["features"]) == 0:
            metrics["splits"][split_name] = {"n_samples": 0}
            write_json(output_dir / f"{split_name}_matches.json", records)
            continue

        summary, sample_metrics = evaluate_dataset(regressor, dataset, names)
        metrics["splits"][split_name] = summary
        write_json(output_dir / f"{split_name}_matches.json", records)
        write_json(output_dir / f"{split_name}_sample_metrics.json", sample_metrics)

    write_json(output_dir / "metrics_summary.json", metrics)

    print("=" * 60)
    print("Axis Regression")
    print("=" * 60)
    print(f"  Feature source: {args.feature_source}")
    print(f"  Feature set:    {args.feature_set}")
    print(f"  Split key:      {args.split_key}")
    print(f"  Model type:     {args.model_type}")
    if args.model_type == "pls":
        print(f"  Components:     {int(selected_value)}")
    else:
        print(f"  Alpha:          {float(selected_value):.4f}")
        if args.model_type in {"ridge", "poly_ridge"}:
            print(f"  Solver:         {args.ridge_solver}")
    if args.model_type == "poly_ridge":
        print(f"  Poly degree:    {args.poly_degree}")
    if args.model_type == "mlp":
        print(f"  Hidden sizes:   {', '.join(str(v) for v in args.mlp_hidden_sizes)}")
        print(f"  Max iter:       {args.mlp_max_iter}")
    if args.use_trainval_cv:
        print(f"  CV folds:       {args.cv_folds}")
        if args.model_type == "pls":
            print(f"  Components grid:{', '.join(str(v) for v in args.pls_components_grid)}")
        elif args.model_type == "mlp":
            print(f"  Alpha grid:     {', '.join(f'{alpha:.4g}' for alpha in args.mlp_alpha_grid)}")
        else:
            print(f"  Alpha grid:     {', '.join(f'{alpha:.4g}' for alpha in args.alpha_grid)}")
    for split_name in ["train", "val", "trainval", "test"]:
        summary = metrics["splits"][split_name]
        print(f"\n[{split_name}]")
        print(f"  Samples:        {summary.get('n_samples', 0)}")
        if summary.get("n_samples", 0) == 0:
            continue
        print(f"  Endpoint MAE:   {summary['endpoint_mae_px']:.3f} px")
        print(f"  Endpoint NMAE:  {summary['endpoint_mae_norm']:.4f}")
        print(f"  Midpoint X MAE: {summary['midpoint_x_mae_px']:.3f} px")
        print(f"  Midpoint NMAE:  {summary['midpoint_x_mae_norm']:.4f}")
        print(f"  Angle MAE:      {summary['angle_mae_deg']:.3f} deg")
        if summary.get("target_r2") is not None:
            print(f"  Target R2:      {summary['target_r2']:.4f}")


if __name__ == "__main__":
    main()
