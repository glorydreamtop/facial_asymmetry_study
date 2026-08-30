#!/usr/bin/env python3
"""Generate landmark-comparison figures + raw-data CSVs for version A/B reports.

Figures: sdr_curves, mre_boxplot, region_mre, per_point_heatmap
CSVs:    fig1_sdr, fig2_mre_boxplot (per-sample), fig3_region, fig4_per_point
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.landmark.schema import POINT_NAMES_31, REGIONS, NUM_POINTS

SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
REGION_ORDER = ["eyebrow", "eye", "nose", "mouth", "jaw"]
REGION_DISPLAY = {"jaw": "Facial contour", "eyebrow": "Eyebrow", "eye": "Eye", "nose": "Nose", "mouth": "Mouth"}


def load_model(mdir: Path) -> tuple[dict, list[dict]]:
    with open(mdir / "paper_metrics_summary.json") as f:
        summary = json.load(f)
    with open(mdir / "paper_error_details.json") as f:
        samples = json.load(f)["samples"]
    return summary, samples


def point_error_matrix(samples: list[dict]) -> np.ndarray:
    n = len(samples)
    mat = np.full((n, NUM_POINTS), np.nan)
    for i, s in enumerate(samples):
        for k, v in s["per_point_error_px"].items():
            mat[i, int(k)] = v
    return mat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name:path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dirs = {spec.split(":")[0]: Path(spec.split(":")[1]) for spec in args.models}
    out = Path(args.output)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out / "figures_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    samples = {}
    mats = {}
    mres = {}
    for name, mdir in model_dirs.items():
        s, samp = load_model(mdir)
        summaries[name] = s
        samples[name] = samp
        mats[name] = point_error_matrix(samp)
        mres[name] = [np.nanmean(mats[name][i]) for i in range(len(samp))]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    names = list(model_dirs.keys())

    # ---- Fig 1: SDR curves ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        sdr = summaries[name]["sdr_px"]
        ax.plot(SDR_THRESHOLDS, [sdr[str(t)] * 100 for t in SDR_THRESHOLDS],
                marker="o", label=name, color=colors[i % 4], linewidth=2)
    ax.set_xlabel("Distance threshold (px)")
    ax.set_ylabel("Success Detection Rate (%)")
    ax.set_title("SDR Curves (4 Landmark Models)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(fig_dir / "sdr_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    with open(data_dir / "fig1_sdr_curves.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + [f"SDR@{t}px(%)" for t in SDR_THRESHOLDS])
        for name in names:
            sdr = summaries[name]["sdr_px"]
            w.writerow([name] + [f"{sdr[str(t)]*100:.2f}" for t in SDR_THRESHOLDS])

    # ---- Fig 2: MRE boxplot ----
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([mres[name] for name in names], labels=names, patch_artist=True)
    for patch, i in zip(bp["boxes"], range(4)):
        patch.set_facecolor(colors[i % 4]); patch.set_alpha(0.6)
    ax.set_ylabel("Mean Radial Error (px)")
    ax.set_title("Per-Image MRE by Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "mre_boxplot.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    with open(data_dir / "fig2_mre_boxplot.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Model", "SampleIndex", "PerImageMRE_px"])
        for name in names:
            for i, v in enumerate(mres[name]):
                w.writerow([name, i + 1, f"{v:.4f}"])

    # ---- Fig 3: Region MRE ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(REGION_ORDER)); width = 0.2
    region_means = {}
    for j, name in enumerate(names):
        means = []
        for reg in REGION_ORDER:
            idx = REGIONS[reg]
            col = mats[name][:, idx]
            means.append(np.nanmean(col))
        region_means[name] = means
        ax.bar(x + (j - 1.5) * width, means, width, label=name, color=colors[j % 4])
    ax.set_xticks(x); ax.set_xticklabels([REGION_DISPLAY[r] for r in REGION_ORDER])
    ax.set_ylabel("Mean Error (px)")
    ax.set_title("Per-Region Landmark Error by Model")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "region_mre.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    with open(data_dir / "fig3_region_mre.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Region"] + names)
        for i, reg in enumerate(REGION_ORDER):
            w.writerow([REGION_DISPLAY[reg]] + [f"{region_means[name][i]:.4f}" for name in names])

    # ---- Fig 4: Per-point heatmap ----
    pt_mat = np.zeros((len(names), NUM_POINTS))
    for j, name in enumerate(names):
        for i in range(NUM_POINTS):
            col = mats[name][:, i]
            pt_mat[j, i] = np.nanmean(col) if len(col[~np.isnan(col)]) else 0
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pt_mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(NUM_POINTS)); ax.set_xticklabels(POINT_NAMES_31, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for j in range(len(names)):
        for i in range(NUM_POINTS):
            ax.text(i, j, f"{pt_mat[j, i]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Per-Landmark MRE (px) Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(fig_dir / "per_point_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    with open(data_dir / "fig4_per_point_heatmap.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Landmark"] + names)
        for i in range(NUM_POINTS):
            w.writerow([POINT_NAMES_31[i]] + [f"{pt_mat[j, i]:.4f}" for j in range(len(names))])

    print(f"Figures -> {fig_dir}")
    print(f"CSVs -> {data_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
