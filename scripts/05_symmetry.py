#!/usr/bin/env python3
"""
Phase 3 — Symmetry Analysis

Full symmetry analysis pipeline: detect landmarks -> fit symmetry axis -> quantify.

Usage:
    # Use custom model
    python scripts/06_symmetry.py \
        --checkpoint checkpoints/best_model.pth \
        --ann_file data/annotations/annotations.json \
        --img_root data/raw \
        --output results/symmetry \
        --batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.detection.detector import RetinaFaceDetector
from src.data.dataset import normalize_image_tensor
from src.landmark.model import build_model
from src.landmark.schema import NUM_POINTS
from src.symmetry.pairs import compute_midpoints
from src.symmetry.axis import fit_symmetry_axis
from src.symmetry.metrics import analyze_symmetry, summarize_report
from src.visualize.plot import plot_symmetry_analysis


def run_custom_pipeline(
    image_rgb: np.ndarray,
    detector: RetinaFaceDetector,
    model: torch.nn.Module,
    device: torch.device,
    input_size: tuple = (256, 256),
):
    """
    Run full pipeline with custom trained model.
    """
    face = detector.detect_single(image_rgb)
    if face is None:
        return None, None

    x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
    h, w = image_rgb.shape[:2]
    margin = 0.2
    fw, fh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * fw))
    y1 = max(0, int(y1 - margin * fh))
    x2 = min(w, int(x2 + margin * fw))
    y2 = min(h, int(y2 + margin * fh))

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, (input_size[1], input_size[0]))
    crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
    crop_tensor = normalize_image_tensor(crop_tensor).unsqueeze(0)
    crop_tensor = crop_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(crop_tensor).cpu().numpy()[0]

    pred_crop = output.reshape(-1, 2)
    # Map to original coords
    landmarks = pred_crop.copy()
    landmarks[:, 0] = pred_crop[:, 0] * (x2 - x1) + x1
    landmarks[:, 1] = pred_crop[:, 1] * (y2 - y1) + y1

    return landmarks, face


def analyze_single_image(
    image_path: str,
    output_dir: str,
    detector: RetinaFaceDetector,
    pipeline_fn,
    image_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Analyze a single image for facial symmetry.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERR] Cannot read image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks, face_info = pipeline_fn(image_rgb)
    if landmarks is None:
        print(f"  [WARN] No face detected: {image_path}")
        return None

    valid_mask = np.ones(NUM_POINTS, dtype=bool)

    # Compute midpoints and fit symmetry axis
    midpoints, pair_labels = compute_midpoints(landmarks, valid_mask)
    axis = fit_symmetry_axis(
        midpoints, use_ransac=True, residual_threshold=5.0
    )

    # Full symmetry analysis
    report = analyze_symmetry(landmarks, axis, valid_mask)

    # Generate summary
    summary = summarize_report(report)
    summary["image"] = image_name or Path(image_path).name
    summary["image_path"] = str(image_path)

    # Visualization
    plot_symmetry_analysis(
        image_rgb,
        landmarks,
        axis,
        report,
        valid_mask,
        title=f"Symmetry Analysis — {summary['image']}",
        save_path=str(Path(output_dir) / f"{Path(image_path).stem}_symmetry.png"),
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Facial symmetry analysis")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--ann_file", help="COCO annotation file for batch mode")
    parser.add_argument("--img_root", default="data/raw", help="Image root")
    parser.add_argument("--output", default="results/symmetry", help="Output directory")
    parser.add_argument("--checkpoint", help="Custom model checkpoint (optional)")
    parser.add_argument("--backbone", default="mobilenet_v2", choices=["mobilenet_v2"])
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Limit images in batch")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3 — Facial Symmetry Analysis")
    print("=" * 60)

    # Init models
    print("\nInitializing models...")
    detector = RetinaFaceDetector(device="auto")

    if not args.checkpoint:
        print("Provide --checkpoint for custom model inference")
        sys.exit(1)

    device = args.device if args.device != "auto" else \
             "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(backbone=args.backbone, num_points=NUM_POINTS, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(torch.device(device))
    pipeline_fn = lambda img: run_custom_pipeline(img, detector, model, torch.device(device))
    print(f"  Using custom model: {args.checkpoint}")

    if args.image:
        # Single image mode
        print(f"\nAnalyzing: {args.image}")
        summary = analyze_single_image(
            args.image, str(output_dir), detector, pipeline_fn
        )
        if summary:
            print(f"\n  Overall Symmetry Score: {summary['overall_score']:.1f}/100")
            print(f"  Axis Angle: {summary['axis_angle']:.2f}°")
            for region, score in summary["region_scores"].items():
                print(f"  {region}: {score:.4f}")

            # Save summary
            summary_path = output_dir / f"{Path(args.image).stem}_symmetry.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved: {summary_path}")

    elif args.batch and args.ann_file:
        # Batch mode
        print(f"\nBatch analysis from: {args.ann_file}")
        with open(args.ann_file, "r") as f:
            coco = json.load(f)

        images_meta = {img["id"]: img for img in coco["images"]}
        annotations = coco["annotations"]

        if args.limit:
            annotations = annotations[:args.limit]

        all_summaries = []
        for ann in tqdm(annotations, desc="Analyzing"):
            img_info = images_meta[ann["image_id"]]
            img_path = Path(args.img_root) / img_info["file_name"]
            if not img_path.exists():
                continue

            summary = analyze_single_image(
                str(img_path), str(output_dir), detector, pipeline_fn,
                image_name=str(img_info["file_name"]),
            )
            if summary:
                all_summaries.append(summary)

        # Aggregate results
        scores = [s["overall_score"] for s in all_summaries]
        print(f"\n{'='*60}")
        print(f"Batch Summary ({len(scores)} faces)")
        print(f"{'='*60}")
        print(f"  Mean Score:     {np.mean(scores):.2f} / 100")
        print(f"  Median Score:   {np.median(scores):.2f} / 100")
        print(f"  Std:            {np.std(scores):.2f}")
        print(f"  Min:            {min(scores):.2f}")
        print(f"  Max:            {max(scores):.2f}")

        if len(scores) > 1:
            print(f"\n  Distribution:")
            bins = [0, 20, 40, 60, 80, 100]
            for i in range(len(bins) - 1):
                count = sum(1 for s in scores if bins[i] <= s < bins[i + 1])
                print(f"    {bins[i]:>3d}-{bins[i + 1]:<3d}: {'█' * count} {count}")

        # Save batch summary
        batch_summary = {
            "n_faces": len(scores),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "results": all_summaries,
        }
        batch_path = output_dir / "batch_symmetry_results.json"
        with open(batch_path, "w") as f:
            json.dump(batch_summary, f, indent=2)
        print(f"\nBatch results saved: {batch_path}")

    else:
        print("Provide --image for single mode or --batch --ann_file for batch mode")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
