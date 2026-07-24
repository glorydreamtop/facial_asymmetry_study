#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import setup_project_paths

setup_project_paths()

from src.symmetry.axis_regression import parse_axis_labelme_json, sort_axis_points_by_y


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_label_box(image: np.ndarray, endpoint_mae_px: float) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (20, 20), (360, 120), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, dst=image)

    cv2.line(image, (36, 50), (96, 50), (0, 255, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "GT axis", (112, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.line(image, (36, 84), (96, 84), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(image, "Pred axis", (112, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.putText(
        image,
        f"Endpoint MAE: {endpoint_mae_px:.2f}px",
        (36, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def clip_and_draw_segment(image: np.ndarray, p1: tuple[float, float], p2: tuple[float, float], color: tuple[int, int, int], thickness: int) -> None:
    h, w = image.shape[:2]
    ok, q1, q2 = cv2.clipLine((0, 0, w, h), (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))))
    if ok:
        cv2.line(image, q1, q2, color, thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test-set GT/pred axis overlay images")
    parser.add_argument("--result_dir", default="results/axis_regression_ridge_pred_formal")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    project_root = Path.cwd()
    result_dir = (project_root / args.result_dir).resolve()
    output_dir = (result_dir / "test_axis_overlays") if args.output_dir is None else (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matches = read_json(result_dir / "test_matches.json")
    sample_metrics = read_json(result_dir / "test_sample_metrics.json")
    metrics_by_name = {item["file_name"]: item for item in sample_metrics}

    manifest = []
    for match in matches:
        file_name = match["file_name"]
        metric = metrics_by_name.get(file_name)
        if metric is None:
            continue

        image_path = (project_root / match["image_path"]).resolve()
        axis_json_path = (project_root / match["axis_json"]).resolve()
        axis_ann = parse_axis_labelme_json(str(axis_json_path))
        if axis_ann is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        gt_axis_points = sort_axis_points_by_y(axis_ann.axis_points)
        y0 = float(gt_axis_points[0, 1])
        y1 = float(gt_axis_points[1, 1])
        pred_slope = float(metric["pred_slope_xy"])
        pred_intercept = float(metric["pred_intercept_x"])
        pred_points = np.array(
            [
                [pred_slope * y0 + pred_intercept, y0],
                [pred_slope * y1 + pred_intercept, y1],
            ],
            dtype=np.float32,
        )

        clip_and_draw_segment(image, tuple(gt_axis_points[0]), tuple(gt_axis_points[1]), (0, 255, 0), 4)
        clip_and_draw_segment(image, tuple(pred_points[0]), tuple(pred_points[1]), (0, 0, 255), 4)

        for point in gt_axis_points:
            cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 6, (0, 255, 0), -1, cv2.LINE_AA)
        for point in pred_points:
            x = int(round(point[0]))
            y = int(round(point[1]))
            cv2.line(image, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(image, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 255), 2, cv2.LINE_AA)

        draw_label_box(image, float(metric["endpoint_mae_px"]))

        save_path = output_dir / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)

        manifest.append(
            {
                "file_name": file_name,
                "output_path": str(save_path.relative_to(project_root)),
                "endpoint_mae_px": float(metric["endpoint_mae_px"]),
                "midpoint_x_error_px": float(metric["midpoint_x_error_px"]),
                "angle_error_deg": float(metric["angle_error_deg"]),
            }
        )

    manifest.sort(key=lambda item: item["endpoint_mae_px"])
    with open(output_dir / "overlay_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(manifest)} overlay images to {output_dir}")


if __name__ == "__main__":
    main()
