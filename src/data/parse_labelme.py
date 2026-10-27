"""
Parse labelme JSON annotation files and convert to unified COCO-style format.

Each labelme JSON contains:
  - shapes: list of {label, points[[x,y]], shape_type}
  - imagePath, imageWidth, imageHeight

Bilateral points (e.g. ebe, ebs, os, ...) appear twice (left/right).
Midline points (e.g. Na, prn, Sn, ...) appear once.
We resolve left/right by x-coordinate: smaller x to left, larger x to right.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.landmark.schema import (
    BILATERAL_TAGS,
    LABELME_TAG_MAP,
    MIDLINE_TAGS,
    NAME_TO_IDX,
    POINT_NAMES_31,
    NUM_POINTS,
    SYMMETRY_PAIRS,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _match_existing_path(candidate: Path) -> Path | None:
    parent = candidate.parent
    if not parent.exists():
        return None

    target_name = candidate.name.casefold()
    for child in parent.iterdir():
        if child.name.casefold() == target_name:
            return child
    return None


def _resolve_image_path(json_path: Path, image_path: str, raw_dir: Path | None = None) -> str:
    """
    解析与 json 配对的图片路径。

    labelme 的 imagePath 往往只保存文件名，因此这里优先：
    1. 使用 json 同目录下的同名文件
    2. 使用 imagePath 指定的文件名
    3. 返回相对于 raw_dir 的路径，便于后续统一拼接
    """
    candidates: List[Path] = []
    if image_path:
        candidates.append(json_path.parent / image_path)

    for ext in IMAGE_EXTENSIONS:
        candidates.append(json_path.with_suffix(ext))
        candidates.append(json_path.with_suffix(ext.upper()))

    for candidate in candidates:
        matched = _match_existing_path(candidate)
        if matched is not None:
            if raw_dir is not None:
                return str(matched.relative_to(raw_dir))
            return str(matched)

    if image_path:
        fallback = json_path.parent / image_path
        if raw_dir is not None:
            try:
                return str(fallback.relative_to(raw_dir))
            except ValueError:
                return str(fallback)
        return str(fallback)

    return json_path.stem


def parse_labelme_json(json_path: str, raw_dir: str | None = None) -> Optional[Dict]:
    """
    Parse a single labelme JSON file.

    Returns a dict with keys:
      - image_path: str (relative path to image)
      - image_width: int
      - image_height: int
      - landmarks: np.ndarray of shape (31, 2) or None if incomplete
      - valid_mask: np.ndarray of shape (31,) - True if point was found
      - raw_labels: list of label strings for debugging
    """
    json_path_obj = Path(json_path)
    raw_dir_obj = Path(raw_dir) if raw_dir is not None else None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = data.get("imagePath", "")
    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)
    resolved_image_path = _resolve_image_path(json_path_obj, image_path, raw_dir_obj)

    tag_points: Dict[str, List[Tuple[float, float]]] = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in LABELME_TAG_MAP:
            continue
        pts = shape.get("points", [])
        if not pts:
            continue
        x, y = pts[0]
        tag = LABELME_TAG_MAP[label]
        tag_points.setdefault(tag, []).append((x, y))

    landmarks = np.full((NUM_POINTS, 2), np.nan, dtype=np.float32)
    valid_mask = np.zeros(NUM_POINTS, dtype=bool)
    errors = []

    for tag, pts in tag_points.items():
        if tag in MIDLINE_TAGS:
            if len(pts) == 1:
                idx = NAME_TO_IDX[tag]
                landmarks[idx] = pts[0]
                valid_mask[idx] = True
            elif len(pts) > 1:
                errors.append(f"Midline tag '{tag}' has {len(pts)} points, expected 1")
        elif tag in BILATERAL_TAGS:
            if len(pts) == 2:
                pts_sorted = sorted(pts, key=lambda p: p[0])
                idx_l = NAME_TO_IDX[f"{tag}_L"]
                idx_r = NAME_TO_IDX[f"{tag}_R"]
                landmarks[idx_l] = pts_sorted[0]
                landmarks[idx_r] = pts_sorted[1]
                valid_mask[idx_l] = True
                valid_mask[idx_r] = True
            elif len(pts) == 1:
                errors.append(f"Bilateral tag '{tag}' has only 1 point, expected 2")
            else:
                errors.append(f"Bilateral tag '{tag}' has {len(pts)} points, expected 2")
        else:
            errors.append(f"Unknown tag '{tag}'")

    return {
        "image_path": resolved_image_path,
        "image_width": img_w,
        "image_height": img_h,
        "landmarks": landmarks,
        "valid_mask": valid_mask,
        "num_valid": int(valid_mask.sum()),
        "errors": errors,
    }


def parse_all_annotations(raw_dir: str) -> List[Dict]:
    """
    Scan raw_dir for labelme JSON files and parse all annotations.

    Returns a list of annotation dicts.
    """
    raw_dir = Path(raw_dir)
    # 递归扫描，兼容“照片/子文件夹/图片+同名json”的组织方式
    json_files = sorted(raw_dir.rglob("*.json"))

    results = []
    for jf in json_files:
        parsed = parse_labelme_json(str(jf), raw_dir=str(raw_dir))
        if parsed is None:
            print(f"  [SKIP] Failed to parse {jf.name}")
            continue
        parsed["json_path"] = str(jf)
        # 目前按图片随机划分，person_id 仅保留作元信息
        parsed["person_id"] = jf.stem
        results.append(parsed)

    return results


def coco_annotation_dict(
    annotations: List[Dict],
    output_path: str,
    point_names: List[str] = POINT_NAMES_31,
):
    """
    Convert parsed annotations to a unified JSON file (COCO keypoint style).

    Output format:
    {
      "categories": [{"id": 1, "name": "face", "keypoints": [...], "skeleton": [...]}],
      "images": [{"id": ..., "file_name": ..., "width": ..., "height": ...}],
      "annotations": [{"id": ..., "image_id": ..., "category_id": 1,
                       "keypoints": [x1,y1,v1, ...]}]
    }
    """
    skeleton = [[l + 1, r + 1] for l, r in SYMMETRY_PAIRS]

    output = {
        "info": {
            "description": "Facial Symmetry Landmarks",
            "version": "1.0",
            "year": 2026,
        },
        "categories": [
            {
                "id": 1,
                "name": "face",
                "keypoints": point_names,
                "skeleton": skeleton,
            }
        ],
        "images": [],
        "annotations": [],
    }

    for i, ann in enumerate(annotations):
        img_id = i + 1
        output["images"].append({
            "id": img_id,
            "file_name": ann["image_path"],
            "width": ann["image_width"],
            "height": ann["image_height"],
        })

        kp = []
        num_valid = 0
        for j in range(NUM_POINTS):
            if ann["valid_mask"][j]:
                x, y = ann["landmarks"][j].tolist()
                v = 1
                num_valid += 1
            else:
                x, y = 0.0, 0.0
                v = 0
            kp.extend([float(x), float(y), v])

        output["annotations"].append({
            "id": img_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": kp,
            "num_keypoints": num_valid,
            "area": float(ann["image_width"] * ann["image_height"]),
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(annotations)} annotations to {output_path}")
    return output
