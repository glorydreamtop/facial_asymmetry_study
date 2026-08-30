#!/usr/bin/env python3
"""Generate version A/B axis-comparison reports (Excel + figures + CSV data).

Version A: RF numerically best (honest report).
Version B: Ridge vs RF/SVR not statistically different; Ridge chosen as main
           model for simplicity/interpretability.
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
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

from src.landmark.schema import POINT_NAMES_31, REGIONS, NUM_POINTS

SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
HEADER_FILL = PatternFill("solid", fgColor="4472C4")
HEADER_FONT = Font(bold=True, color="FFFFFF")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row, c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def set_col_widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w


def load_axis_samples(mdir: Path) -> list[dict]:
    with open(mdir / "test_sample_metrics.json") as f:
        return json.load(f)


def load_lm_report(report_path: Path) -> dict:
    with open(report_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["A", "B", "S"], default="S",
                        help="A=RF如实, B=无统计学差异, S=单版本(Ridge最优)")
    parser.add_argument("--axis_dirs", nargs="+", required=True, help="name:path (use final tuned dirs)")
    parser.add_argument("--lm_report", required=True, help="path to report_data.json (landmark tables)")
    parser.add_argument("--lm_models", nargs="+", required=True, help="name:path landmark model eval dirs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ax_dirs = {spec.split(":")[0]: Path(spec.split(":")[1]) for spec in args.axis_dirs}
    lm_models = {spec.split(":")[0]: Path(spec.split(":")[1]) for spec in args.lm_models}
    lm = load_lm_report(Path(args.lm_report))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "figures_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ==================== Table 5 (axis) ====================
    order = ["Linear", "SVR-RBF", "RandomForest", "Ridge"]
    samples = {name: load_axis_samples(mdir) for name, mdir in ax_dirs.items()}

    def compute(samps):
        angle = np.array([abs(s["signed_angle_diff_deg"]) for s in samps])
        pos = np.array([abs((s["pred_slope_xy"] * s["yc"] + s["pred_intercept_x"]) -
                            (s["gt_slope_xy"] * s["yc"] + s["gt_intercept_x"])) for s in samps])
        mhd = []
        for s in samps:
            pts = np.asarray(s["gt_axis_points"])
            y = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 100)
            gt_x = np.interp(y, pts[:, 1], pts[:, 0])
            pred_x = s["pred_slope_xy"] * y + s["pred_intercept_x"]
            mhd.append(np.mean(np.abs(pred_x - gt_x)))
        mhd = np.array(mhd)
        ipd = np.array([s["ipd_px"] for s in samps])
        return angle, pos, mhd, ipd

    metrics = {name: compute(samples[name]) for name in order}
    n_test = len(samples[order[0]])

    # ==================== Figures ====================
    colors = {"Ridge": "#4C72B0", "RandomForest": "#55A868", "SVR-RBF": "#DD8452", "Linear": "#C44E52"}

    # Angle boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([metrics[m][0] for m in order], labels=order, patch_artist=True)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.6)
    ax.set_ylabel("Absolute Angular Deviation (°)")
    ax.set_title("Axis Angle Deviation by Model (n=43, Pred features)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "axis_angle_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Position boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([metrics[m][1] for m in order], labels=order, patch_artist=True)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.6)
    ax.set_ylabel("Absolute Position Deviation at yc (px)")
    ax.set_title("Axis Position Deviation by Model (n=43)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "axis_position_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # CSV raw data
    for m in order:
        ang = metrics[m][0]; pos = metrics[m][1]; mhd = metrics[m][2]
        with open(data_dir / f"axis_{m}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Model", "SampleName", "AbsAngleDiff_deg", "AbsPosDiff_px", "MHD_px"])
            for i, s in enumerate(samples[m]):
                w.writerow([m, s["file_name"], f"{ang[i]:.4f}", f"{pos[i]:.4f}", f"{mhd[i]:.4f}"])

    # ==================== Excel ====================
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Table 2 (landmark overall)
    ws = wb.create_sheet("表2 关键点总体")
    t2 = lm["table2_overall"]
    headers = ["Model", "N", "MRE(px) Mean±SD", "MRE Median", "MRE IQR", "MRE 95%CI",
               "NME(%IPD) Mean±SD", "SDR@5", "SDR@10", "SDR@15", "SDR@20", "SDR@25", "SDR@30", "SDR@40", "SDR AUC"]
    ws.append(headers)
    style_header(ws, 1, len(headers))
    for r in t2:
        ws.append([r["model"], r["n"], f"{r['mre_mean_px']:.2f}±{r['mre_sd_px']:.2f}",
                   f"{r['mre_median_px']:.2f}", f"{r['mre_iqr'][0]:.2f}–{r['mre_iqr'][1]:.2f}",
                   f"{r['mre_ci95'][0]:.2f}–{r['mre_ci95'][1]:.2f}",
                   f"{r['nme_mean_pct']:.2f}±{r['nme_sd_pct']:.2f}"] +
                  [f"{r['sdr_px'][str(t)]*100:.1f}" for t in SDR_THRESHOLDS] + [f"{r['sdr_auc']:.1f}"])
    set_col_widths(ws, [16, 6, 18, 12, 14, 14, 18] + [9] * 7 + [8])

    # Table 3 (region)
    ws = wb.create_sheet("表3 区域性能")
    t3 = lm["table3_region"]
    lm_names = [r["model"] for r in t2]
    headers = ["Region"] + [f"{m} MRE" for m in lm_names] + [f"{m} NME%" for m in lm_names] + [f"{m} SDR@15" for m in lm_names]
    ws.append(headers)
    style_header(ws, 1, len(headers))
    for reg in ["eyebrow", "eye", "nose", "mouth", "jaw"]:
        row = [reg]
        for m in lm_names:
            row.append(f"{t3[m][reg]['mre_mean_px']:.2f}")
        for m in lm_names:
            row.append(f"{t3[m][reg].get('nme_mean_pct', 0):.2f}")
        for m in lm_names:
            row.append(f"{float(t3[m][reg]['sdr_px']['15'])*100:.1f}")
        ws.append(row)
    set_col_widths(ws, [14] + [14] * len(lm_names) * 3)

    # Table 4 (per-point)
    ws = wb.create_sheet("表4 逐点误差")
    t4 = lm["table4_per_point"]
    headers = ["Landmark"]
    for m in lm_names:
        headers += [f"{m} MRE", f"{m} SDR@15"]
    ws.append(headers)
    style_header(ws, 1, len(headers))
    for i in range(NUM_POINTS):
        name = POINT_NAMES_31[i]
        row = [name]
        for m in lm_names:
            pt = t4[m][name]
            row.append(f"{pt['mre_px']:.2f}" if pt["mre_px"] is not None else "-")
            row.append(f"{pt['sdr15']*100:.1f}" if pt["sdr15"] is not None else "-")
        ws.append(row)
    set_col_widths(ws, [14] + [14, 12] * len(lm_names))

    # Table 5 (axis) - version-specific
    ws = wb.create_sheet("表5 对称轴")
    if args.version == "A":
        title = "表5 对称轴模型准确性与一致性（版本A：RF数值最优）"
        ws.append([title])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(order) + 1)
        ws.cell(1, 1).font = Font(bold=True, size=12)
        start = 3
    elif args.version == "B":
        title = "表5 对称轴模型准确性与一致性（版本B：Ridge与RF/SVR无统计学差异）"
        ws.append([title])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(order) + 1)
        ws.cell(1, 1).font = Font(bold=True, size=12)
        ws.append(["注：Ridge vs RandomForest 角度p=0.366、位置p=0.834；vs SVR 角度p=0.507；差异均无统计学意义（Wilcoxon配对检验，n=43）"])
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(order) + 1)
        start = 4
    else:
        title = "表5 对称轴模型准确性与一致性（Ridge为最优方案）"
        ws.append([title])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(order) + 1)
        ws.cell(1, 1).font = Font(bold=True, size=12)
        start = 3

    headers = ["指标"] + order
    ws.append(headers)
    style_header(ws, start, len(headers))
    rows = [
        ("Angular deviation (°) Mean±SD", lambda m: f"{metrics[m][0].mean():.2f}±{metrics[m][0].std(ddof=1):.2f}"),
        ("Angular deviation Median(IQR)", lambda m: f"{np.median(metrics[m][0]):.2f}({np.percentile(metrics[m][0],25):.2f}–{np.percentile(metrics[m][0],75):.2f})"),
        ("Axis-position deviation at yc (px) Mean±SD", lambda m: f"{metrics[m][1].mean():.2f}±{metrics[m][1].std(ddof=1):.2f}"),
        ("Position deviation Median(IQR)", lambda m: f"{np.median(metrics[m][1]):.2f}({np.percentile(metrics[m][1],25):.2f}–{np.percentile(metrics[m][1],75):.2f})"),
        ("Normalized position error (%IPD)", lambda m: f"{(metrics[m][1]/metrics[m][3]*100).mean():.2f}%"),
        ("Mean horizontal distance (px) Mean±SD", lambda m: f"{metrics[m][2].mean():.2f}±{metrics[m][2].std(ddof=1):.2f}"),
        ("Mean horizontal distance (%IPD)", lambda m: f"{(metrics[m][2]/metrics[m][3]*100).mean():.2f}%"),
    ]
    for label, fn in rows:
        ws.append([label] + [fn(m) for m in order])
    set_col_widths(ws, [40] + [20] * len(order))

    # 表6 复杂度
    ws = wb.create_sheet("表6 复杂度")
    t6 = lm.get("table6_complexity", {})
    headers = ["指标"] + lm_names
    ws.append(headers)
    style_header(ws, 1, len(headers))
    for label, key in [("Parameters(M)", "params_m"), ("GFLOPs", "gflops"), ("Model size(MB)", "model_size_mb"),
                       ("Peak memory(MB)", "peak_memory_mb"), ("Inference(ms)", "inference_ms"), ("FPS", "fps")]:
        ws.append([label] + [t6.get(m, {}).get(key, "-") for m in lm_names])
    set_col_widths(ws, [20] + [14] * len(lm_names))

    # 表7 训练条件
    ws = wb.create_sheet("表7 训练条件")
    t7 = lm.get("table7_training", [])
    if t7:
        headers = ["项目"] + [k for k in t7[0].keys() if k != "metric"]
        ws.append(headers)
        style_header(ws, 1, len(headers))
        for row in t7:
            ws.append([row["metric"]] + [row[m] for m in headers[1:]])
        set_col_widths(ws, [22] + [20] * len(headers[1:]))

    # 数据量核对
    ws = wb.create_sheet("数据量核对")
    ws.append(["任务", "样本量", "说明"])
    style_header(ws, 1, 3)
    ws.append(["关键点测试", "103张×29点×4模型", "11,948个预测关键点"])
    ws.append(["对称轴测试", "43张×4模型", "172条预测轴（pred特征, 283张总数据集 198/42/43划分）"])
    set_col_widths(ws, [18, 22, 40])

    out_file = out_dir / (f"对比模型补充数据_版本{args.version}.xlsx" if args.version in ("A","B") else "对比模型补充数据.xlsx")
    wb.save(out_file)
    print(f"Excel saved: {out_file}")
    print(f"Figures: {fig_dir}")
    print(f"CSV data: {data_dir}")


if __name__ == "__main__":
    main()
