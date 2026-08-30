#!/usr/bin/env python3
"""Export raw plotting data for all comparison figures to CSV files.

Each CSV matches one figure, so the user can re-plot with their own colors.
Outputs to <outdir>/figures_data/:
  fig1_sdr_curves.csv
  fig2_mre_boxplot.csv
  fig3_region_mre.csv
  fig4_per_point_heatmap.csv
  fig5_axis_angle_boxplot.csv
  fig6_axis_position_boxplot.csv
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

from src.landmark.schema import POINT_NAMES_31, REGIONS

SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
REGION_DISPLAY = {"jaw": "Facial contour", "eyebrow": "Eyebrow", "eye": "Eye", "nose": "Nose", "mouth": "Mouth"}
REGION_ORDER = ["eyebrow", "eye", "nose", "mouth", "jaw"]


def write_csv(path: Path, headers: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  {path.name} ({len(rows)} rows)")


def export_fig1_sdr(out: Path, report: dict):
    rows = []
    for model in report["table2_overall"]:
        row = [model["model"]]
        for t in SDR_THRESHOLDS:
            row.append(model["sdr_px"][str(t)] * 100)
        rows.append(row)
    write_csv(out / "fig1_sdr_curves.csv",
              ["Model"] + [f"SDR@{t}px(%)" for t in SDR_THRESHOLDS], rows)


def export_fig2_mre_boxplot(out: Path, report: dict):
    # Long format: Model, sample_index, per_image_mre_px
    rows = []
    for model in report["table2_overall"]:
        for i, mre in enumerate(model["per_sample_mre"]):
            rows.append([model["model"], i + 1, f"{mre:.4f}"])
    write_csv(out / "fig2_mre_boxplot.csv",
              ["Model", "SampleIndex", "PerImageMRE_px"], rows)


def export_fig3_region_mre(out: Path, report: dict):
    t3 = report["table3_region"]
    models = [m["model"] for m in report["table2_overall"]]
    rows = []
    for region in REGION_ORDER:
        for model in models:
            rd = t3[model].get(region, {})
            sdr15 = rd.get("sdr_px", {}).get("15", 0)
            sdr30 = rd.get("sdr_px", {}).get("30", 0)
            rows.append([REGION_DISPLAY[region], model,
                         f"{rd.get('mre_mean_px', 0):.4f}",
                         f"{rd.get('mre_sd_px', 0):.4f}",
                         f"{float(sdr15)*100:.2f}",
                         f"{float(sdr30)*100:.2f}"])
    write_csv(out / "fig3_region_mre.csv",
              ["Region", "Model", "MRE_mean_px", "MRE_sd_px", "SDR@15%", "SDR@30%"], rows)


def export_fig4_heatmap(out: Path, report: dict):
    t4 = report["table4_per_point"]
    models = [m["model"] for m in report["table2_overall"]]
    rows = []
    for region in REGION_ORDER:
        for idx in REGIONS[region]:
            name = POINT_NAMES_31[idx]
            row = [REGION_DISPLAY[region], name]
            for model in models:
                pt = t4[model][name]
                row.append(f"{pt['mre_px']:.4f}" if pt["mre_px"] is not None else "")
                row.append(f"{pt['sdr15']*100:.2f}" if pt["sdr15"] is not None else "")
            rows.append(row)
    headers = ["Region", "Landmark"]
    for model in models:
        headers += [f"{model}_MRE_px", f"{model}_SDR@15%"]
    write_csv(out / "fig4_per_point_heatmap.csv", headers, rows)


def export_fig5_axis_angle(out: Path, ax_dirs: dict):
    rows = []
    for name, adir in ax_dirs.items():
        with open(adir / "test_sample_metrics.json") as f:
            samples = json.load(f)
        for s in samples:
            rows.append([name, s["file_name"], f"{s['signed_angle_diff_deg']:.4f}",
                         f"{abs(s['signed_angle_diff_deg']):.4f}"])
    write_csv(out / "fig5_axis_angle_boxplot.csv",
              ["Model", "SampleName", "SignedAngleDiff_deg", "AbsAngleDiff_deg"], rows)


def export_fig6_axis_position(out: Path, ax_dirs: dict):
    rows = []
    for name, adir in ax_dirs.items():
        with open(adir / "test_sample_metrics.json") as f:
            samples = json.load(f)
        for s in samples:
            pred_x = s["pred_slope_xy"] * s["yc"] + s["pred_intercept_x"]
            gt_x = s["gt_slope_xy"] * s["yc"] + s["gt_intercept_x"]
            diff = pred_x - gt_x
            rows.append([name, s["file_name"], f"{diff:.4f}", f"{abs(diff):.4f}", f"{s['ipd_px']:.4f}"])
    write_csv(out / "fig6_axis_position_boxplot.csv",
              ["Model", "SampleName", "SignedPosDiff_px", "AbsPosDiff_px", "IPD_px"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="path to report_data.json")
    parser.add_argument("--axis", nargs="+", required=True, help="name:path axis model dirs")
    parser.add_argument("--output", required=True, help="output dir (figures_data will be created inside)")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)
    ax_dirs = {spec.split(":")[0]: Path(spec.split(":")[1]) for spec in args.axis}

    out = Path(args.output)
    data_dir = out / "figures_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print("Exporting figure raw data to:", data_dir)

    export_fig1_sdr(data_dir, report)
    export_fig2_mre_boxplot(data_dir, report)
    export_fig3_region_mre(data_dir, report)
    export_fig4_heatmap(data_dir, report)
    export_fig5_axis_angle(data_dir, ax_dirs)
    export_fig6_axis_position(data_dir, ax_dirs)
    print("Done.")


if __name__ == "__main__":
    main()
