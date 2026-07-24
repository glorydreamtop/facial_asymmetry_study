#!/usr/bin/env python3
"""Phase 0 — Step 2: Data preprocessing."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.parse_labelme import parse_all_annotations, coco_annotation_dict
from src.data.split import save_split, split_with_partial_in_train
from src.detection.detector import RetinaFaceDetector
from src.landmark.schema import NUM_POINTS


def classify_annotation(annotation: dict) -> str:
    """Classify one sample for downstream usage.

    - error_conflict: parsing conflicts, excluded for now
    - complete_full: gold-standard sample, can be used anywhere
    - partial_clean: missing some points but no conflicts, train only
    """
    if annotation.get("errors"):
        return "error_conflict"
    if annotation.get("num_valid", 0) == NUM_POINTS:
        return "complete_full"
    return "partial_clean"


def filter_usable_annotations(annotations):
    usable = []
    complete_indices = []
    partial_indices = []
    error_annotations = []

    for ann in annotations:
        sample_type = classify_annotation(ann)
        ann["sample_type"] = sample_type

        if sample_type == "error_conflict":
            error_annotations.append(ann)
            continue

        usable_idx = len(usable)
        usable.append(ann)
        if sample_type == "complete_full":
            complete_indices.append(usable_idx)
        else:
            partial_indices.append(usable_idx)

    return usable, complete_indices, partial_indices, error_annotations


def load_force_train_paths(manifest_path: str | None) -> set[str]:
    if not manifest_path:
        return set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return {
        item["image_path"]
        for item in manifest.get("items", [])
        if item.get("image_path")
    }


def run_face_detection(annotations, raw_dir, detector, crop_dir, target_size=(256, 256)):
    """
    Detect faces and save crops for all valid annotations.

    Also adds face bbox info to each annotation dict.
    """
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir)

    for ann in tqdm(annotations, desc="Detecting faces"):
        img_path = raw_dir / ann["image_path"]
        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_path}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face = detector.detect_single(image_rgb)
        if face is None:
            print(f"  [WARN] No face detected: {ann['image_path']}")
            ann["face_bbox"] = None
            continue

        bbox = face["bbox"]
        ann["face_bbox"] = bbox
        ann["face_confidence"] = float(face["confidence"])
        ann["face_landmarks_5"] = face["landmarks_5"].tolist() if face["landmarks_5"] is not None else None

        # Crop face with margin
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        margin = 0.2
        fw, fh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - margin * fw))
        y1 = max(0, int(y1 - margin * fh))
        x2 = min(w, int(x2 + margin * fw))
        y2 = min(h, int(y2 + margin * fh))

        crop = image_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (target_size[1], target_size[0]))

        # Save crop
        crop_name = Path(ann["image_path"]).stem + "_crop.png"
        crop_path = crop_dir / crop_name
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
        ann["crop_path"] = str(crop_path)

    return annotations


def main():
    parser = argparse.ArgumentParser(description="Preprocess facial landmark data")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--ann_dir", default="data/annotations")
    parser.add_argument("--split_dir", default="data/splits")
    parser.add_argument("--crop_dir", default="data/crops")
    parser.add_argument(
        "--force_train_manifest",
        default=None,
        help="Optional manifest JSON whose samples must stay in the training split",
    )
    parser.add_argument("--detect_faces", action="store_true", help="Run face detection")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Target size H W for face crops")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 0 — Preprocessing")
    print("=" * 60)

    # Step 1: Parse labelme JSON
    print("\n[Step 1] Parsing labelme annotations...")
    annotations = parse_all_annotations(args.raw_dir)
    print(f"  Parsed {len(annotations)} annotations.")

    usable_annotations, complete_indices, partial_indices, error_annotations = filter_usable_annotations(annotations)
    print(f"  Complete full-point samples: {len(complete_indices)}")
    print(f"  Partial clean samples:     {len(partial_indices)}")
    print(f"  Error/conflict samples:    {len(error_annotations)}")

    # Step 2: Save unified COCO format
    print("\n[Step 2] Saving unified COCO annotations...")
    ann_dir = Path(args.ann_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    usable_coco_path = ann_dir / "annotations_usable.json"
    complete_coco_path = ann_dir / "annotations_complete.json"
    partial_coco_path = ann_dir / "annotations_partial.json"
    error_coco_path = ann_dir / "annotations_error.json"

    coco_annotation_dict(usable_annotations, str(usable_coco_path))
    coco_annotation_dict([usable_annotations[i] for i in complete_indices], str(complete_coco_path))
    coco_annotation_dict([usable_annotations[i] for i in partial_indices], str(partial_coco_path))
    coco_annotation_dict(error_annotations, str(error_coco_path))

    # Step 3: Split usable data.
    # Partial samples are forced into train; val/test stay gold-standard only.
    print("\n[Step 3] Splitting usable dataset...")
    force_train_paths = load_force_train_paths(args.force_train_manifest)
    force_train_indices = [
        idx for idx, ann in enumerate(usable_annotations) if ann["image_path"] in force_train_paths
    ]
    if force_train_paths:
        matched_paths = {usable_annotations[idx]["image_path"] for idx in force_train_indices}
        missing_paths = sorted(force_train_paths - matched_paths)
        print(f"  Force-train manifest samples: {len(force_train_paths)}")
        print(f"  Matched usable samples:       {len(matched_paths)}")
        if missing_paths:
            print(f"  [WARN] Not usable after parsing: {len(missing_paths)}")

    train_idx, val_idx, test_idx = split_with_partial_in_train(
        complete_indices,
        partial_indices,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        force_train_indices=force_train_indices,
    )
    save_split(usable_annotations, train_idx, val_idx, test_idx, args.split_dir)

    # Step 4: (Optional) Face detection
    if args.detect_faces:
        print("\n[Step 4] Running face detection...")
        detector = RetinaFaceDetector(device="auto", conf_thresh=0.8)
        usable_annotations = run_face_detection(
            usable_annotations, args.raw_dir, detector, args.crop_dir,
            target_size=tuple(args.target_size),
        )
        # Re-save with face info
        coco_annotation_dict(usable_annotations, str(usable_coco_path))
        coco_annotation_dict([usable_annotations[i] for i in complete_indices], str(complete_coco_path))
        coco_annotation_dict([usable_annotations[i] for i in partial_indices], str(partial_coco_path))
    else:
        print("\n[Step 4] Skipped (use --detect_faces to run)")

    print("\nPreprocessing complete!")
    print(f"  Usable annotations:   {usable_coco_path}")
    print(f"  Complete annotations: {complete_coco_path}")
    print(f"  Partial annotations:  {partial_coco_path}")
    print(f"  Error annotations:    {error_coco_path}")
    print(f"  Splits:      {args.split_dir}/")
    print(f"  Crops:       {args.crop_dir}/")


if __name__ == "__main__":
    main()
