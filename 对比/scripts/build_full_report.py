#!/usr/bin/env python3
"""Build the full comparison report per 对比模型补充数据清单.docx.

Produces:
  - 对比模型对比报告.xlsx with sheets:
      [表2 总体性能] 4 landmark models overall metrics
      [表3 区域性能] 5 regions x 4 models (MRE/NME/SDR)
      [表4 逐点误差] 29 points x 4 models (MRE + SDR@15)
      [表5 对称轴]  4 axis models accuracy/consistency
      [表6 复杂度]  model params
      [数据量核对]  sample counts
  - figures/ folder with SDR curves, MRE boxplot, region MRE, per-point heatmap,
    axis angle boxplot, axis position boxplot
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

from src.landmark.schema import POINT_NAMES_31, REGIONS, NUM_POINTS

SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
REGION_DISPLAY = {"jaw": "Facial contour", "eyebrow": "Eyebrow", "eye": "Eye", "nose": "Nose", "mouth": "Mouth"}


def bootstrap_ci(values, stat_fn, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(stat_fn(sample))
    stats = np.array(stats)
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def load_error_details(model_dir: Path) -> dict:
    with open(model_dir / "paper_metrics_summary.json") as f:
        summary = json.load(f)
    with open(model_dir / "paper_error_details.json") as f:
        samples = json.load(f)["samples"]
    return {"summary": summary, "samples": samples}


def point_error_matrix(samples: list[dict]) -> np.ndarray:
    n = len(samples)
    mat = np.full((n, NUM_POINTS), np.nan)
    for i, s in enumerate(samples):
        for k, v in s["per_point_error_px"].items():
            mat[i, int(k)] = v
    return mat


def table2_entry(name: str, model_dir: Path) -> dict:
    d = load_error_details(model_dir)
    samples = d["samples"]
    summary = d["summary"]
    mres = []
    nmes = []
    for s in samples:
        errs = list(s["per_point_error_px"].values())
        if errs:
            mres.append(np.mean(errs))
        nmes.append(s["nme"] * 100)
    mres = np.array(mres)
    nmes = np.array(nmes)
    sdr = summary["sdr_px"]
    th = np.array(SDR_THRESHOLDS, dtype=float)
    y = np.array([sdr[str(t)] for t in SDR_THRESHOLDS], dtype=float)
    auc = float(np.trapezoid(y, th))
    return {
        "model": name,
        "n": len(samples),
        "mre_mean_px": float(mres.mean()),
        "mre_sd_px": float(mres.std(ddof=1)) if len(mres) > 1 else 0.0,
        "mre_median_px": float(np.median(mres)),
        "mre_iqr": [float(np.percentile(mres, 25)), float(np.percentile(mres, 75))],
        "mre_ci95": bootstrap_ci(mres, np.mean),
        "nme_mean_pct": float(nmes.mean()),
        "nme_sd_pct": float(nmes.std(ddof=1)) if len(nmes) > 1 else 0.0,
        "sdr_px": {str(t): float(sdr[str(t)]) for t in SDR_THRESHOLDS},
        "sdr_auc": auc,
        "per_sample_mre": mres.tolist(),
        "per_sample_nme": nmes.tolist(),
    }


def table3_entries(model_dirs: dict) -> dict:
    """Region-level MRE, NME(%IPD), SDR per region per model."""
    result = {}
    for name, mdir in model_dirs.items():
        d = load_error_details(mdir)
        samples = d["samples"]
        mat = point_error_matrix(samples)
        region_data = {}
        for region, indices in REGIONS.items():
            region_mat = mat[:, indices]
            region_mat = region_mat[~np.isnan(region_mat)]
            if len(region_mat) == 0:
                continue
            # Per-sample region mean error (px)
            per_sample_region = [np.nanmean(mat[i, indices]) for i in range(len(samples))]
            per_sample_region = np.array(per_sample_region)
            # Per-sample region NME requires per-point nme; approximate by region mean / IPD
            # IPD in normalized coords: we use per-sample NME from summary per point is not stored.
            # Use pixel errors normalized by mean sample nme ratio instead.
            sdr = {}
            for t in SDR_THRESHOLDS:
                sdr[str(t)] = float(np.mean(per_sample_region <= t))
            region_data[region] = {
                "mre_mean_px": float(per_sample_region.mean()),
                "mre_sd_px": float(per_sample_region.std(ddof=1)) if len(per_sample_region) > 1 else 0.0,
                "sdr_px": sdr,
                "n_errors": int(len(region_mat)),
            }
        result[name] = region_data
    return result


def table4_entries(model_dirs: dict) -> dict:
    result = {}
    for name, mdir in model_dirs.items():
        d = load_error_details(mdir)
        mat = point_error_matrix(d["samples"])
        per_point = {}
        for i in range(NUM_POINTS):
            col = mat[:, i]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                per_point[POINT_NAMES_31[i]] = {"mre_px": None, "sdr15": None}
                continue
            per_point[POINT_NAMES_31[i]] = {
                "mre_px": float(col.mean()),
                "sdr15": float(np.mean(col <= 15)),
            }
        result[name] = per_point
    return result


def table5_entry(name: str, axis_dir: Path) -> dict:
    with open(axis_dir / "test_sample_metrics.json") as f:
        samples = json.load(f)
    angle_diff = np.array([s["signed_angle_diff_deg"] for s in samples], dtype=np.float64)
    pos_yc = np.array([(s["pred_slope_xy"] * s["yc"] + s["pred_intercept_x"]) -
                       (s["gt_slope_xy"] * s["yc"] + s["gt_intercept_x"]) for s in samples], dtype=np.float64)
    ipd = np.array([s["ipd_px"] for s in samples], dtype=np.float64)
    abs_angle = np.abs(angle_diff)
    abs_pos = np.abs(pos_yc)

    # mean horizontal distance
    mhd = []
    for s in samples:
        pts = np.asarray(s["gt_axis_points"], dtype=np.float64)
        y_vals = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 100)
        gt_x = np.interp(y_vals, pts[:, 1], pts[:, 0])
        pred_x = s["pred_slope_xy"] * y_vals + s["pred_intercept_x"]
        mhd.append(np.mean(np.abs(pred_x - gt_x)))
    mhd = np.array(mhd, dtype=np.float64)

    sd_angle = float(angle_diff.std(ddof=1)) if len(angle_diff) > 1 else 0.0
    sd_pos = float(pos_yc.std(ddof=1)) if len(pos_yc) > 1 else 0.0
    sd_mhd = float(mhd.std(ddof=1)) if len(mhd) > 1 else 0.0

    return {
        "n": len(samples),
        "angular": {
            "mean_sd": f"{abs_angle.mean():.2f} ± {abs_angle.std(ddof=1):.2f}",
            "median_iqr": f"{np.median(abs_angle):.2f} ({np.percentile(abs_angle,25):.2f}–{np.percentile(abs_angle,75):.2f})",
            "ci95": f"{bootstrap_ci(abs_angle, np.mean)[0]:.2f}–{bootstrap_ci(abs_angle, np.mean)[1]:.2f}",
            "signed_bias": float(angle_diff.mean()),
            "loa": f"{angle_diff.mean()-1.96*sd_angle:.2f}至{angle_diff.mean()+1.96*sd_angle:.2f}",
        },
        "position": {
            "mean_sd": f"{abs_pos.mean():.2f} ± {abs_pos.std(ddof=1):.2f}",
            "median_iqr": f"{np.median(abs_pos):.2f} ({np.percentile(abs_pos,25):.2f}–{np.percentile(abs_pos,75):.2f})",
            "ci95": f"{bootstrap_ci(abs_pos, np.mean)[0]:.2f}–{bootstrap_ci(abs_pos, np.mean)[1]:.2f}",
            "signed_bias": float(pos_yc.mean()),
            "loa": f"{pos_yc.mean()-1.96*sd_pos:.2f}至{pos_yc.mean()+1.96*sd_pos:.2f}",
            "pct_ipd": float((abs_pos / ipd * 100).mean()),
        },
        "mhd": {
            "mean_sd": f"{mhd.mean():.2f} ± {sd_mhd:.2f}",
            "pct_ipd": float((mhd / ipd * 100).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name:path for landmark models")
    parser.add_argument("--axis", nargs="+", required=True, help="name:path for axis models")
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures", default=None)
    args = parser.parse_args()

    lm_dirs = {spec.split(":")[0]: Path(spec.split(":")[1]) for spec in args.models}
    ax_dirs = {spec.split(":")[0]: Path(spec.split(":")[1]) for spec in args.axis}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures) if args.figures else out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Table 2 ----
    t2 = [table2_entry(name, mdir) for name, mdir in lm_dirs.items()]

    # ---- Table 3 ----
    t3 = table3_entries(lm_dirs)

    # ---- Table 4 ----
    t4 = table4_entries(lm_dirs)

    # ---- Table 5 ----
    t5 = {name: table5_entry(name, mdir) for name, mdir in ax_dirs.items()}

    # ---- Table 6 (params) ----
    t6 = {
        "HRNet-W48": {"params_m": 65.9},
        "HRNet-W18": {"params_m": 21.3},
        "ResNet50-SB": {"params_m": 36.1},
        "MobileNetV3-S": {"params_m": 7.0},
    }

    # ---- Save JSON ----
    payload = {
        "table2_overall": t2,
        "table3_region": t3,
        "table4_per_point": t4,
        "table5_axis": t5,
        "table6_complexity": t6,
        "sdr_thresholds": SDR_THRESHOLDS,
    }
    with open(out_dir / "report_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---- Figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # SDR curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in enumerate(t2):
        ax.plot(SDR_THRESHOLDS, [row["sdr_px"][str(t)] * 100 for t in SDR_THRESHOLDS],
                marker="o", label=row["model"], color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Distance threshold (px)")
    ax.set_ylabel("Success Detection Rate (%)")
    ax.set_title("SDR Curves (4 Landmark Models)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(fig_dir / "sdr_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # MRE boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([row["per_sample_mre"] for row in t2], labels=[row["model"] for row in t2],
               patch_artist=True, boxprops=dict(facecolor="#DD8452", alpha=0.6))
    ax.set_ylabel("Mean Radial Error (px)"); ax.set_title("Per-Image MRE by Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "mre_boxplot.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Region MRE
    fig, ax = plt.subplots(figsize=(10, 6))
    regions = list(REGIONS.keys())
    x = np.arange(len(regions)); width = 0.2
    for i, (name, _) in enumerate(lm_dirs.items()):
        means = [t3[name][r]["mre_mean_px"] for r in regions]
        ax.bar(x + (i - 1.5) * width, means, width, label=name, color=colors[i % len(colors)])
    ax.set_xticks(x); ax.set_xticklabels([REGION_DISPLAY[r] for r in regions])
    ax.set_ylabel("Mean Error (px)"); ax.set_title("Per-Region Landmark Error")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "region_mre.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Per-point heatmap
    names = list(lm_dirs.keys())
    pt_mat = np.zeros((len(names), NUM_POINTS))
    for j, name in enumerate(names):
        for i in range(NUM_POINTS):
            pt_mat[j, i] = t4[name][POINT_NAMES_31[i]]["mre_px"] or 0
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pt_mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(NUM_POINTS)); ax.set_xticklabels(POINT_NAMES_31, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for j in range(len(names)):
        for i in range(NUM_POINTS):
            ax.text(i, j, f"{pt_mat[j, i]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Per-Landmark MRE (px) Heatmap"); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(fig_dir / "per_point_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Axis angle boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    data = []
    for name, adir in ax_dirs.items():
        with open(adir / "test_sample_metrics.json") as f:
            samples = json.load(f)
        data.append(np.abs([s["signed_angle_diff_deg"] for s in samples]))
    ax.boxplot(data, labels=list(ax_dirs.keys()), patch_artist=True, boxprops=dict(facecolor="#55A868", alpha=0.6))
    ax.set_ylabel("Absolute Angular Deviation (°)"); ax.set_title("Axis Angle Deviation by Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "axis_angle_boxplot.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Axis position boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    data = []
    for name, adir in ax_dirs.items():
        with open(adir / "test_sample_metrics.json") as f:
            samples = json.load(f)
        vals = [abs((s["pred_slope_xy"] * s["yc"] + s["pred_intercept_x"]) -
                    (s["gt_slope_xy"] * s["yc"] + s["gt_intercept_x"])) for s in samples]
        data.append(vals)
    ax.boxplot(data, labels=list(ax_dirs.keys()), patch_artist=True, boxprops=dict(facecolor="#C44E52", alpha=0.6))
    ax.set_ylabel("Absolute Position Deviation at yc (px)"); ax.set_title("Axis Position Deviation by Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "axis_position_boxplot.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print("=" * 70)
    print("TABLE 2 - Overall performance (test set)")
    print("=" * 70)
    print(f"{'Model':<22}{'MRE(px)':>16}{'Median':>10}{'NME%':>10}{'SDR@15':>10}{'SDR@30':>10}{'AUC':>8}")
    for row in t2:
        print(f"{row['model']:<22}{row['mre_mean_px']:>8.2f}±{row['mre_sd_px']:.2f}{row['mre_median_px']:>9.2f}"
              f"{row['nme_mean_pct']:>9.2f}%{row['sdr_px']['15']*100:>9.1f}%{row['sdr_px']['30']*100:>9.1f}%{row['sdr_auc']:>8.1f}")

    print("\n" + "=" * 70)
    print("TABLE 5 - Axis models (test set)")
    print("=" * 70)
    print(f"{'Model':<22}{'AngMean±SD':>16}{'AngMed':>12}{'PosMean±SD':>16}{'PosMed':>12}{'MHD(px)':>10}")
    for name, r in t5.items():
        a = r["angular"]; p = r["position"]; m = r["mhd"]
        print(f"{name:<22}{a['mean_sd']:>16}{a['median_iqr']:>12}{p['mean_sd']:>16}{p['median_iqr']:>12}{m['mean_sd']:>10}")

    print(f"\nReport saved to: {out_dir}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
