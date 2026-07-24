#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro


REGION_ORDER = [
    ("overall", "Overall"),
    ("eyebrow", "Eyebrow"),
    ("eye", "Eye"),
    ("nose", "Nose"),
    ("mouth", "Mouth"),
    ("jaw", "Facial Contour"),
]

FIGURE_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
TABLE_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
RGB_COLORS = [
    (68, 4, 90),
    (48, 104, 141),
    (31, 146, 139),
    (53, 183, 119),
    (145, 213, 66),
    (248, 230, 32),
]
COLORS = [tuple(channel / 255.0 for channel in rgb) for rgb in RGB_COLORS]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_sample_level_mre(details: dict) -> dict[str, list[float]]:
    values = {key: [] for key, _ in REGION_ORDER}
    for sample in details.get("samples", []):
        point_errors = [float(v) for v in sample.get("per_point_error_px", {}).values()]
        if point_errors:
            values["overall"].append(float(np.mean(point_errors)))
        for region_key, _ in REGION_ORDER[1:]:
            region_errors = sample.get("per_region_error_px", {}).get(region_key, [])
            if region_errors:
                values[region_key].append(float(np.mean(region_errors)))
    return values


def collect_point_level_errors(details: dict) -> dict[str, list[float]]:
    values = {key: [] for key, _ in REGION_ORDER}
    for sample in details.get("samples", []):
        point_errors = [float(v) for v in sample.get("per_point_error_px", {}).values()]
        values["overall"].extend(point_errors)
        for region_key, _ in REGION_ORDER[1:]:
            region_errors = [float(v) for v in sample.get("per_region_error_px", {}).get(region_key, [])]
            values[region_key].extend(region_errors)
    return values


def collect_sample_level_sdr(details: dict, thresholds: list[int]) -> dict[str, dict[int, list[float]]]:
    values = {key: {threshold: [] for threshold in thresholds} for key, _ in REGION_ORDER}
    for sample in details.get("samples", []):
        point_errors = np.asarray([float(v) for v in sample.get("per_point_error_px", {}).values()], dtype=np.float64)
        if len(point_errors):
            for threshold in thresholds:
                values["overall"][threshold].append(float(np.mean(point_errors <= threshold) * 100.0))

        for region_key, _ in REGION_ORDER[1:]:
            region_errors = np.asarray(sample.get("per_region_error_px", {}).get(region_key, []), dtype=np.float64)
            if len(region_errors):
                for threshold in thresholds:
                    values[region_key][threshold].append(float(np.mean(region_errors <= threshold) * 100.0))
    return values


def is_normal_distribution(values: np.ndarray) -> bool:
    if len(values) < 3:
        return True
    if np.allclose(values, values[0]):
        return True
    try:
        _, p_value = shapiro(values)
    except Exception:
        return False
    return bool(p_value > 0.05)


def format_distribution(values: np.ndarray) -> tuple[str, str]:
    if is_normal_distribution(values):
        return f"{np.mean(values):.2f}±{np.std(values):.2f}", "mean±sd"
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return f"{median:.2f} ({q1:.2f}, {q3:.2f})", "median(q1,q3)"


def build_table_rows(summary: dict, details: dict, sdr_mode: str = "sample_distribution") -> list[dict[str, str]]:
    sample_level = collect_sample_level_mre(details)
    sample_level_sdr = collect_sample_level_sdr(details, TABLE_THRESHOLDS)
    point_level = collect_point_level_errors(details)
    rows = []
    for region_key, region_label in REGION_ORDER:
        region_values = np.asarray(sample_level[region_key], dtype=np.float64)
        sdr_source = np.asarray(point_level[region_key], dtype=np.float64)
        if region_key == "overall":
            mre_display = f"{summary['mre_px']:.2f}±{summary['std_mre_px']:.2f}"
            distribution_type = "mean±sd"
        else:
            mre_display, distribution_type = format_distribution(region_values)

        row = {
            "Region": region_label,
            "MRE(px)": mre_display,
            "distribution": distribution_type,
        }
        for threshold in TABLE_THRESHOLDS:
            if sdr_mode == "pooled":
                row[f"SDR<=${threshold}px"] = f"{np.mean(sdr_source <= threshold) * 100:.2f}"
            else:
                sample_sdr_values = np.asarray(sample_level_sdr[region_key][threshold], dtype=np.float64)
                if len(sample_sdr_values):
                    row[f"SDR<=${threshold}px"] = format_distribution(sample_sdr_values)[0]
                else:
                    row[f"SDR<=${threshold}px"] = f"{np.mean(sdr_source <= threshold) * 100:.2f}±0.00"
        rows.append(row)
    return rows


def write_table_files(rows: list[dict[str, str]], output_dir: Path, file_stem: str, note_lines: list[str] | None = None) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / f"{file_stem}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Region", "MRE(px)", "distribution", *[f"SDR<=${threshold}px" for threshold in TABLE_THRESHOLDS]],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = tables_dir / f"{file_stem}.md"
    headers = ["Region", "MRE(px)", *[f"SDR(≤{threshold}px, %)" for threshold in TABLE_THRESHOLDS]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Region"],
                    row["MRE(px)"],
                    *[row[f"SDR<=${threshold}px"] for threshold in TABLE_THRESHOLDS],
                ]
            )
            + " |"
        )
    if note_lines:
        lines.append("")
        lines.extend(note_lines)
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = tables_dir / f"{file_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def save_figure3(details: dict, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    point_level = collect_point_level_errors(details)
    fig, ax = plt.subplots(figsize=(9, 6))
    for (region_key, region_label), color in zip(REGION_ORDER, COLORS):
        values = np.asarray(point_level[region_key], dtype=np.float64)
        y = [np.mean(values <= threshold) * 100 for threshold in FIGURE_THRESHOLDS]
        ax.plot(FIGURE_THRESHOLDS, y, color=color, linewidth=2.4, linestyle="-", label=region_label)

    ax.set_title("SDR Curve of All Regions")
    ax.set_xlabel("Threshold (px)")
    ax.set_ylabel("SDR (%)")
    ax.set_xticks(FIGURE_THRESHOLDS)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure3_sdr_curve_of_all_regions_hrnet.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "figure3_sdr_curve_of_all_regions_hrnet.pdf", bbox_inches="tight")
    plt.close(fig)


def save_figure4(details: dict, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sample_level = collect_sample_level_mre(details)
    labels = [label for _, label in REGION_ORDER]
    data = [np.asarray(sample_level[key], dtype=np.float64) for key, _ in REGION_ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "#333333", "linewidth": 1.2},
        capprops={"color": "#333333", "linewidth": 1.2},
        boxprops={"edgecolor": "#333333", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    ax.set_xlabel("Region")
    ax.set_ylabel("MRE (px)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure4_hrnet_only_mre_boxplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "figure4_hrnet_only_mre_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HRNet-only final report assets")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--table4_pooled_only", action="store_true")
    args = parser.parse_args()

    summary = load_json(Path(args.summary))
    details = load_json(Path(args.details))
    output_dir = Path(args.output_dir)

    if args.table4_pooled_only:
        pooled_rows = build_table_rows(summary, details, sdr_mode="pooled")
        write_table_files(
            pooled_rows,
            output_dir,
            file_stem="table4_hrnet_only_pooled_sdr",
            note_lines=[
                "旧版统计口径说明：本表中的 SDR 采用 pooled point-level 统计方式，即将某一区域内所有关键点误差直接汇总后，计算其在给定阈值内的总体比例。因此，每个阈值对应一个总体百分比，不包含样本间分布的 `±SD` 或 `M(QR)` 表示。",
            ],
        )
        return

    rows = build_table_rows(summary, details, sdr_mode="sample_distribution")
    write_table_files(
        rows,
        output_dir,
        file_stem="table4_hrnet_only",
        note_lines=["MRE/SDR formatting rule: normal distributions use X±SD; non-normal distributions use M(QR)."],
    )
    save_figure3(details, output_dir)
    save_figure4(details, output_dir)


if __name__ == "__main__":
    main()
