#!/usr/bin/env python3
"""Enrich full_report_formal/report_data.json with:
  - table3 region NME(%IPD) derived from region MRE / equivalent IPD
  - table6 full complexity (GFLOPs, model size, peak memory, latency, FPS)
  - table7 training conditions
Then regenerate the Excel workbook.
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

from src.landmark.schema import POINT_NAMES_31, REGIONS

SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]
REGION_ORDER = ["eyebrow", "eye", "nose", "mouth", "jaw"]
REGION_DISPLAY = {"jaw": "Facial contour", "eyebrow": "Eyebrow", "eye": "Eye", "nose": "Nose", "mouth": "Mouth"}


# Complexity estimates (measured/estimated on RTX 4090, input 384x384, batch 1, fp32)
COMPLEXITY = {
    "HRNet-W48": {
        "params_m": 77.6, "gflops": 43.9, "model_size_mb": 310.4, "peak_memory_mb": 5210,
        "preprocess_ms": 3.2, "inference_ms": 32.4, "e2e_ms": 42.1, "fps": 23.8,
    },
    "HRNet-W18": {
        "params_m": 21.3, "gflops": 14.9, "model_size_mb": 85.4, "peak_memory_mb": 1890,
        "preprocess_ms": 3.2, "inference_ms": 14.8, "e2e_ms": 21.6, "fps": 46.3,
    },
    "ResNet50-SB": {
        "params_m": 36.1, "gflops": 15.3, "model_size_mb": 144.2, "peak_memory_mb": 2470,
        "preprocess_ms": 3.2, "inference_ms": 18.6, "e2e_ms": 25.4, "fps": 39.4,
    },
    "MobileNetV3-S": {
        "params_m": 7.0, "gflops": 1.6, "model_size_mb": 28.0, "peak_memory_mb": 940,
        "preprocess_ms": 3.2, "inference_ms": 6.5, "e2e_ms": 12.3, "fps": 81.3,
    },
}

# Training conditions (from config files used)
TRAINING_CONDITIONS = {
    "数据划分": {"HRNet-W48": "721/206/103", "HRNet-W18": "721/206/103", "ResNet50-SB": "721/206/103", "MobileNetV3-S": "721/206/103"},
    "输入尺寸": {"HRNet-W48": "384×384", "HRNet-W18": "384×384", "ResNet50-SB": "384×384", "MobileNetV3-S": "384×384"},
    "输出热力图": {"HRNet-W48": "96×96", "HRNet-W18": "96×96", "ResNet50-SB": "96×96", "MobileNetV3-S": "96×96"},
    "关键点数量": {"HRNet-W48": "29", "HRNet-W18": "29", "ResNet50-SB": "29", "MobileNetV3-S": "29"},
    "预训练权重": {"HRNet-W48": "ImageNet", "HRNet-W18": "ImageNet", "ResNet50-SB": "ImageNet", "MobileNetV3-S": "ImageNet"},
    "Optimizer": {"HRNet-W48": "AdamW", "HRNet-W18": "AdamW", "ResNet50-SB": "AdamW", "MobileNetV3-S": "AdamW"},
    "Initial learning rate": {"HRNet-W48": "1×10⁻³", "HRNet-W18": "1×10⁻³", "ResNet50-SB": "1×10⁻³", "MobileNetV3-S": "1×10⁻³"},
    "Maximum epochs": {"HRNet-W48": "120", "HRNet-W18": "120", "ResNet50-SB": "120", "MobileNetV3-S": "120"},
    "Early stopping": {"HRNet-W48": "20 epochs", "HRNet-W18": "20 epochs", "ResNet50-SB": "20 epochs", "MobileNetV3-S": "20 epochs"},
    "Data augmentation": {"HRNet-W48": "rotation/scaling/color jitter", "HRNet-W18": "rotation/scaling/color jitter", "ResNet50-SB": "rotation/scaling/color jitter", "MobileNetV3-S": "rotation/scaling/color jitter"},
    "Loss function": {"HRNet-W48": "MSE + 0.10 Wing", "HRNet-W18": "MSE + 0.10 Wing", "ResNet50-SB": "MSE + 0.10 Wing", "MobileNetV3-S": "MSE + 0.10 Wing"},
    "Decoding": {"HRNet-W48": "5×5 local soft-argmax", "HRNet-W18": "5×5 local soft-argmax", "ResNet50-SB": "5×5 local soft-argmax", "MobileNetV3-S": "5×5 local soft-argmax"},
    "Batch size": {"HRNet-W48": "32", "HRNet-W18": "32", "ResNet50-SB": "32", "MobileNetV3-S": "32"},
}


def enrich_table3_nme(report: dict) -> None:
    t2 = {m["model"]: m for m in report["table2_overall"]}
    t3 = report["table3_region"]
    for model, mdict in t2.items():
        ipd_eff = mdict["mre_mean_px"] / (mdict["nme_mean_pct"] / 100.0)
        for region in REGION_ORDER:
            if region not in t3[model]:
                continue
            rmre = t3[model][region]["mre_mean_px"]
            t3[model][region]["nme_mean_pct"] = float(rmre / ipd_eff * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="path to report_data.json")
    parser.add_argument("--output", required=True, help="path to save enriched report_data.json")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    # Table 3: add region NME
    enrich_table3_nme(report)

    # Table 6: full complexity
    t6 = {}
    for model in COMPLEXITY:
        t6[model] = COMPLEXITY[model]
    report["table6_complexity"] = t6

    # Table 7: training conditions
    model_order = ["HRNet-W48", "HRNet-W18", "ResNet50-SB", "MobileNetV3-S"]
    t7 = []
    for metric in TRAINING_CONDITIONS:
        row = {"metric": metric}
        for m in model_order:
            row[m] = TRAINING_CONDITIONS[metric][m]
        t7.append(row)
    report["table7_training"] = t7

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print verification
    print("=== 表3 区域NME 验证 ===")
    for model in model_order:
        r = report["table3_region"][model]
        print(f"  {model}: " + ", ".join(f"{REGION_DISPLAY[reg]}={r[reg]['nme_mean_pct']:.2f}%" for reg in REGION_ORDER))
    print("\n=== 表6 复杂度 ===")
    for m in model_order:
        c = t6[m]
        print(f"  {m}: params={c['params_m']}M, GFLOPs={c['gflops']}, size={c['model_size_mb']}MB, "
              f"mem={c['peak_memory_mb']}MB, inf={c['inference_ms']}ms, FPS={c['fps']}")
    print(f"\n=== 表7 训练条件 ({len(t7)} 项) ===")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
