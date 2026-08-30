#!/usr/bin/env python3
"""Generate per-point landmark raw-data CSV (per docx section 6).

For each model x sample x landmark: Euclidean error + Success_5..40 flags.
These are derived from the real per-point errors stored in paper_error_details.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.landmark.schema import POINT_NAMES_31, NUM_POINTS

SDR_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name:path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dirs = {spec.split(":")[0]: Path(spec.split(":")[1]) for spec in args.models}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for name, mdir in model_dirs.items():
        with open(mdir / "paper_error_details.json") as f:
            data = json.load(f)["samples"]

        rows = []
        for s in data:
            img_id = s["image_id"]
            for i in range(NUM_POINTS):
                err = s["per_point_error_px"].get(str(i))
                if err is None:
                    continue
                row = [name, img_id, POINT_NAMES_31[i], f"{err:.4f}"]
                for t in SDR_THRESHOLDS:
                    row.append(1 if err <= t else 0)
                rows.append(row)

        csv_path = out / f"landmark_raw_{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            headers = ["Model", "Image_ID", "Landmark", "Euclidean_error_px"] + \
                      [f"Success_{t}" for t in SDR_THRESHOLDS]
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"  {csv_path.name}: {len(rows)} rows")

    # Also generate a per-image MRE table (data count check: 412 per-image MREs)
    summary_rows = []
    for name, mdir in model_dirs.items():
        with open(mdir / "paper_error_details.json") as f:
            data = json.load(f)["samples"]
        for s in data:
            errs = list(s["per_point_error_px"].values())
            if errs:
                summary_rows.append([name, s["image_id"], f"{sum(errs)/len(errs):.4f}"])
    csv_path = out / "per_image_mre.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Image_ID", "PerImageMRE_px"])
        writer.writerows(summary_rows)
    print(f"  {csv_path.name}: {len(summary_rows)} rows")


if __name__ == "__main__":
    main()
