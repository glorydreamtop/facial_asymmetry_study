#!/usr/bin/env python3
"""Generate comparison report for 4 landmark models (HRNet-W48 vs 3 comparison models).

Reads each model's paper_metrics_summary.json + paper_error_details.json and produces:
  - comparison_summary.json (Table 2: overall metrics per model)
  - comparison_region.json (Table 3: per-region MRE/SDR per model)
  - comparison_per_point.json (Table 4: per-landmark MRE per model)
  - comparison_metrics.csv (flat table for spreadsheets)
  - figures: sdr_curves.png, mre_boxplot.png, region_mre.png, per_point_heatmap.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "对比/scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.landmark.schema import POINT_NAMES_31, REGIONS, NUM_POINTS

SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
REGION_DISPLAY = {"jaw": "Facial contour"}


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(stat_fn(sample))
    stats = np.array(stats)
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def load_model(model_dir: Path) -> dict:
    with open(model_dir / "paper_metrics_summary.json") as f:
        summary = json.load(f)
    with open(model_dir / "paper_error_details.json") as f:
        details = json.load(f)["samples"]
    return {"summary": summary, "samples": details}


def per_sample_mre(samples: list[dict]) -> np.ndarray:
    mres = []
    for s in samples:
        errs = list(s["per_point_error_px"].values())
        if errs:
            mres.append(np.mean(errs))
    return np.array(mres)


def per_sample_region_mre(samples: list[dict]) -> dict[str, np.ndarray]:
    out = {r: [] for r in REGIONS}
    for s in samples:
        for r, indices in REGIONS.items():
            vals = [s["per_point_error_px"][str(i)] for i in indices if str(i) in s["per_point_error_px"]]
            if vals:
                out[r].append(np.mean(vals))
    return {r: np.array(v) for r, v in out.items()}


def per_point_matrix(samples: list[dict]) -> np.ndarray:
    n = len(samples)
    mat = np.full((n, NUM_POINTS), np.nan)
    for i, s in enumerate(samples):
        for k, v in s["per_point_error_px"].items():
            mat[i, int(k)] = v
    return mat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="pairs of name:path_to_model_dir")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    model_specs = []
    for item in args.models:
        name, path = item.split(":")
        model_specs.append((name, Path(path)))

    data = {name: load_model(path) for name, path in model_specs}
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============ Table 2: overall metrics ============
    table2 = []
    for name in data:
        d = data[name]
        summary = d["summary"]
        mres = per_sample_mre(d["samples"])
        nme = np.array([s["nme"] for s in d["samples"]])
        sdr = summary["sdr_px"]

        # AUC of SDR curve (trapezoidal, thresholds 5..40)
        th = np.array(SDR_THRESHOLDS, dtype=float)
        y = np.array([sdr[str(t)] for t in SDR_THRESHOLDS], dtype=float)
        auc = float(np.trapezoid(y, th))

        row = {
            "model": name,
            "n_samples": summary["n_samples"],
            "mre_mean_px": float(mres.mean()),
            "mre_sd_px": float(mres.std(ddof=1)),
            "mre_median_px": float(np.median(mres)),
            "mre_iqr": [float(np.percentile(mres, 25)), float(np.percentile(mres, 75))],
            "mre_95ci": bootstrap_ci(mres, np.mean),
            "nme_mean_pct": float(nme.mean() * 100),
            "nme_sd_pct": float(nme.std(ddof=1) * 100),
            "sdr_px": {str(t): float(sdr[str(t)]) for t in SDR_THRESHOLDS},
            "sdr_auc": auc,
            "per_sample_mre": mres.tolist(),
            "per_sample_nme": (nme * 100).tolist(),
        }
        table2.append(row)

    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump({"models": table2}, f, indent=2, ensure_ascii=False)

    # ============ Table 3: per-region metrics ============
    table3 = {}
    for name in data:
        region_mres = per_sample_region_mre(data[name]["samples"])
        summary = data[name]["summary"]
        table3[name] = {}
        for r in REGIONS:
            vals = region_mres[r]
            sdr_r = {}
            for t in SDR_THRESHOLDS:
                sdr_r[str(t)] = float(np.mean(vals <= t))
            table3[name][r] = {
                "mre_mean_px": float(vals.mean()) if len(vals) else None,
                "mre_sd_px": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                "sdr_px": sdr_r,
            }

    with open(out_dir / "comparison_region.json", "w", encoding="utf-8") as f:
        json.dump(table3, f, indent=2, ensure_ascii=False)

    # ============ Table 4: per-point metrics ============
    table4 = {}
    for name in data:
        mat = per_point_matrix(data[name]["samples"])
        table4[name] = {}
        for i in range(NUM_POINTS):
            col = mat[:, i]
            col = col[~np.isnan(col)]
            table4[name][POINT_NAMES_31[i]] = float(col.mean()) if len(col) else None

    with open(out_dir / "comparison_per_point.json", "w", encoding="utf-8") as f:
        json.dump(table4, f, indent=2, ensure_ascii=False)

    # ============ CSV flat table ============
    import csv
    with open(out_dir / "comparison_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Metric", "Value"])
        for row in table2:
            for k in ["mre_mean_px", "mre_sd_px", "mre_median_px", "nme_mean_pct", "sdr_auc"]:
                writer.writerow([row["model"], k, row[k]])
            for t in SDR_THRESHOLDS:
                writer.writerow([row["model"], f"sdr@{t}px", row["sdr_px"][str(t)]])

    # ============ Figures ============
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # SDR curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in enumerate(table2):
        ax.plot(SDR_THRESHOLDS, [row["sdr_px"][str(t)] * 100 for t in SDR_THRESHOLDS],
                marker="o", label=row["model"], color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Distance threshold (px)")
    ax.set_ylabel("Success Detection Rate (%)")
    ax.set_title("SDR Curves for 4 Landmark Models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_dir / "sdr_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # MRE boxplot (per-sample)
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [row["per_sample_mre"] for row in table2]
    ax.boxplot(box_data, labels=[row["model"] for row in table2], patch_artist=True,
               boxprops=dict(facecolor="#DD8452", alpha=0.6))
    ax.set_ylabel("Mean Radial Error (px)")
    ax.set_title("Per-Image MRE Distribution by Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mre_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Region MRE grouped bar
    fig, ax = plt.subplots(figsize=(10, 6))
    regions = list(REGIONS.keys())
    x = np.arange(len(regions))
    width = 0.2
    for i, (name, _) in enumerate(model_specs):
        means = [table3[name][r]["mre_mean_px"] or 0 for r in regions]
        ax.bar(x + (i - 1.5) * width, means, width, label=name, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_DISPLAY.get(r, r) for r in regions])
    ax.set_ylabel("Mean Error (px)")
    ax.set_title("Per-Region Landmark Error by Model")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "region_mre.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Per-point heatmap (4 models × 29 points)
    names = [row["model"] for row in table2]
    pt_mat = np.zeros((len(names), NUM_POINTS))
    for j, name in enumerate(names):
        for i in range(NUM_POINTS):
            pt_mat[j, i] = table4[name][POINT_NAMES_31[i]] or 0
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pt_mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(NUM_POINTS))
    ax.set_xticklabels(POINT_NAMES_31, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for j in range(len(names)):
        for i in range(NUM_POINTS):
            ax.text(i, j, f"{pt_mat[j, i]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Per-Landmark MRE (px) Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_point_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ============ Console summary ============
    print(f"\n{'Model':<28} {'MRE(px)':>10} {'Median':>8} {'NME%':>8} {'SDR@15':>8} {'SDR@30':>8} {'AUC':>6}")
    print("-" * 78)
    for row in table2:
        print(f"{row['model']:<28} {row['mre_mean_px']:>8.2f}±{row['mre_sd_px']:.2f} {row['mre_median_px']:>8.2f} "
              f"{row['nme_mean_pct']:>7.2f}% {row['sdr_px']['15']*100:>7.1f}% {row['sdr_px']['30']*100:>7.1f}% {row['sdr_auc']:>6.1f}")

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
