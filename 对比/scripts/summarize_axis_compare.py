#!/usr/bin/env python3
"""Summarize axis-model comparison (Table 5 metrics) from test_sample_metrics.json.

Computes per-model: angular deviation (mean±SD, median IQR, 95% CI, signed bias, 95% LoA),
position deviation at yc, normalized position error (%IPD), mean horizontal distance.
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


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(stat_fn(sample))
    stats = np.array(stats)
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def mean_horizontal_distance(gt_points: np.ndarray, pred_slope: float, pred_intercept: float) -> float:
    """Average |pred_x - gt_x| over 100 longitudinal samples along the axis."""
    pts = np.asarray(gt_points, dtype=np.float64)
    y_vals = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 100)
    gt_x = np.interp(y_vals, pts[:, 1], pts[:, 0])
    pred_x = pred_slope * y_vals + pred_intercept
    return float(np.mean(np.abs(pred_x - gt_x)))


def summarize_model(model_dir: Path) -> dict:
    with open(model_dir / "test_sample_metrics.json") as f:
        samples = json.load(f)

    angle_diff = np.array([s["signed_angle_diff_deg"] for s in samples], dtype=np.float64)
    pos_diff = np.array([s["signed_position_diff_px"] for s in samples], dtype=np.float64)
    ipd = np.array([s["ipd_px"] for s in samples], dtype=np.float64)
    yc = np.array([s["yc"] for s in samples], dtype=np.float64)
    abs_angle = np.abs(angle_diff)
    abs_pos = np.abs(pos_diff)

    # Position deviation at yc: pred_x(yc) - gt_x(yc)
    pos_at_yc = []
    for s in samples:
        pred_x = s["pred_slope_xy"] * s["yc"] + s["pred_intercept_x"]
        gt_x = s["gt_slope_xy"] * s["yc"] + s["gt_intercept_x"]
        pos_at_yc.append(pred_x - gt_x)
    pos_at_yc = np.array(pos_at_yc, dtype=np.float64)
    abs_pos_yc = np.abs(pos_at_yc)

    # Mean horizontal distance
    mhd = np.array([mean_horizontal_distance(s["gt_axis_points"], s["pred_slope_xy"], s["pred_intercept_x"]) for s in samples], dtype=np.float64)

    sd_angle = float(angle_diff.std(ddof=1)) if len(angle_diff) > 1 else 0.0
    sd_pos = float(pos_at_yc.std(ddof=1)) if len(pos_at_yc) > 1 else 0.0
    sd_mhd = float(mhd.std(ddof=1)) if len(mhd) > 1 else 0.0

    return {
        "n_samples": len(samples),
        "angular": {
            "mean_deg": float(angle_diff.mean()),
            "sd_deg": sd_angle,
            "abs_mean_deg": float(abs_angle.mean()),
            "abs_sd_deg": float(abs_angle.std(ddof=1)) if len(abs_angle) > 1 else 0.0,
            "median_deg": float(np.median(abs_angle)),
            "iqr_deg": [float(np.percentile(abs_angle, 25)), float(np.percentile(abs_angle, 75))],
            "ci95_deg": bootstrap_ci(abs_angle, np.mean),
            "signed_bias_deg": float(angle_diff.mean()),
            "loa_deg": [float(angle_diff.mean() - 1.96 * sd_angle), float(angle_diff.mean() + 1.96 * sd_angle)],
        },
        "position": {
            "mean_px": float(pos_at_yc.mean()),
            "sd_px": sd_pos,
            "abs_mean_px": float(abs_pos_yc.mean()),
            "abs_sd_px": float(abs_pos_yc.std(ddof=1)) if len(abs_pos_yc) > 1 else 0.0,
            "median_px": float(np.median(abs_pos_yc)),
            "iqr_px": [float(np.percentile(abs_pos_yc, 25)), float(np.percentile(abs_pos_yc, 75))],
            "ci95_px": bootstrap_ci(abs_pos_yc, np.mean),
            "signed_bias_px": float(pos_at_yc.mean()),
            "loa_px": [float(pos_at_yc.mean() - 1.96 * sd_pos), float(pos_at_yc.mean() + 1.96 * sd_pos)],
            "mean_pct_ipd": float((abs_pos_yc / ipd * 100).mean()),
        },
        "mean_horizontal_distance": {
            "mean_px": float(mhd.mean()),
            "sd_px": sd_mhd,
            "mean_pct_ipd": float((mhd / ipd * 100).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {}
    for spec in args.models:
        name, path = spec.split(":")
        results[name] = summarize_model(Path(path))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "axis_comparison_table5.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'Model':<22} {'AngMean±SD':>14} {'AngMed':>8} {'PosMean±SD':>14} {'PosMed':>8} {'MHD(px)':>10}")
    print("-" * 82)
    for name, r in results.items():
        a = r["angular"]
        p = r["position"]
        m = r["mean_horizontal_distance"]
        print(f"{name:<22} {a['abs_mean_deg']:>7.2f}±{a['abs_sd_deg']:.2f} {a['median_deg']:>8.2f} "
              f"{p['abs_mean_px']:>8.2f}±{p['abs_sd_px']:.2f} {p['median_px']:>8.2f} {m['mean_px']:>10.2f}")

    print(f"\nSaved to {out_dir / 'axis_comparison_table5.json'}")


if __name__ == "__main__":
    main()
