#!/usr/bin/env python3
"""Phase 0 — Step 0: Filter raw labelme data."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.parse_labelme import parse_all_annotations
from src.landmark.schema import NUM_POINTS


def classify_annotation(annotation: dict) -> tuple[str, list[str]]:
    """将样本分为完整、缺点可用、冲突剔除三类。"""
    reasons: list[str] = []

    if annotation.get("errors"):
        reasons.extend(annotation["errors"])

    image_path = annotation.get("image_path")
    json_path = annotation.get("json_path")
    if not image_path:
        reasons.append("missing image_path")
    if not json_path:
        reasons.append("missing json_path")

    if reasons:
        return "error_conflict", reasons
    if annotation.get("num_valid", 0) == NUM_POINTS:
        return "complete_31", []
    return "partial_clean", [f"num_valid={annotation.get('num_valid', 0)}, expected={NUM_POINTS}"]


def copy_preserve_structure(src_path: Path, src_root: Path, dst_root: Path) -> Path:
    """按相对路径复制文件，保持 source_dir 下的目录结构不变。"""
    relative_path = src_path.relative_to(src_root)
    dst_path = dst_root / relative_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def build_report(records: list[dict]) -> dict:
    usable = [r for r in records if r["sample_type"] != "error_conflict"]
    dropped = [r for r in records if r["sample_type"] == "error_conflict"]
    complete = [r for r in records if r["sample_type"] == "complete_31"]
    partial = [r for r in records if r["sample_type"] == "partial_clean"]

    error_counter: Counter[str] = Counter()
    for record in dropped:
        for reason in record["reasons"]:
            error_counter[reason] += 1

    return {
        "total": len(records),
        "usable": len(usable),
        "complete_31": len(complete),
        "partial_clean": len(partial),
        "dropped": len(dropped),
        "usable_rate": len(usable) / len(records) if records else 0.0,
        "drop_reasons": dict(error_counter.most_common()),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter raw facial landmark samples")
    parser.add_argument("--source_dir", default="照片", help="Source directory containing images and labelme json")
    parser.add_argument("--target_dir", default="data/raw", help="Filtered output directory")
    parser.add_argument(
        "--report_path",
        default="results/filter_raw_report.json",
        help="Path to save the filter report",
    )
    parser.add_argument(
        "--clear_target",
        action="store_true",
        help="Clear target directory before copying filtered files",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    report_path = Path(args.report_path)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if args.clear_target and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 0 — Filter Raw Data")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    annotations = parse_all_annotations(str(source_dir))
    print(f"\nFound {len(annotations)} annotations.")

    records: list[dict] = []
    copied_pairs = 0

    for ann in annotations:
        sample_type, reasons = classify_annotation(ann)
        json_path = Path(ann["json_path"])
        image_path = source_dir / ann["image_path"]

        record = {
            "json_path": str(json_path.relative_to(source_dir)),
            "image_path": ann["image_path"],
            "num_valid": ann.get("num_valid", 0),
            "sample_type": sample_type,
            "reasons": reasons,
        }

        if sample_type != "error_conflict":
            copy_preserve_structure(json_path, source_dir, target_dir)
            copy_preserve_structure(image_path, source_dir, target_dir)
            copied_pairs += 1

        records.append(record)

    report = build_report(records)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nUsable samples:      {report['usable']}")
    print(f"  Complete 31-point: {report['complete_31']}")
    print(f"  Partial clean:     {report['partial_clean']}")
    print(f"Dropped conflicts:   {report['dropped']}")
    print(f"Copied pairs:        {copied_pairs}")
    print(f"Report saved to: {report_path}")

    if report["drop_reasons"]:
        print("\nTop drop reasons:")
        for reason, count in list(report["drop_reasons"].items())[:10]:
            print(f"  {count:>4}  {reason}")


if __name__ == "__main__":
    main()
