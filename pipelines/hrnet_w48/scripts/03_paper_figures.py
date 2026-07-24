#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from _bootstrap import setup_project_paths

setup_project_paths()

from pipelines.hrnet_w48.hrnet_w48_pipeline.model import build_model
from pipelines.hrnet_w48.hrnet_w48_pipeline.reporting import (
    find_event_file,
    load_scalar_history,
    render_ranked_samples,
    save_sample_grid,
    save_training_curves,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready HRNet-W48 figures")
    parser.add_argument("--config", default="pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    model_cfg = config["model"]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_samples_dir = output_dir / "paper_samples"
    paper_samples_dir.mkdir(parents=True, exist_ok=True)

    event_file = find_event_file(Path(args.log_dir))
    history = load_scalar_history(event_file)
    save_training_curves(history, figures_dir / "training_curves.png")
    with open(figures_dir / "training_scalar_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(data_cfg["ann_file"], "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {int(img["id"]): img for img in coco["images"]}
    annotations_by_image_id = {int(ann["image_id"]): ann for ann in coco["annotations"]}

    model = build_model(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device_t, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device_t)
    model.eval()

    best_rendered = render_ranked_samples(
        summary.get("best_samples", [])[: args.top_k],
        annotations_by_image_id,
        images_by_id,
        Path(data_cfg["img_root"]),
        model,
        device_t,
        paper_samples_dir / "best",
        input_size=tuple(model_cfg["input_size"]),
    )
    worst_rendered = render_ranked_samples(
        list(reversed(summary.get("worst_samples", [])[-args.top_k:])),
        annotations_by_image_id,
        images_by_id,
        Path(data_cfg["img_root"]),
        model,
        device_t,
        paper_samples_dir / "worst",
        input_size=tuple(model_cfg["input_size"]),
    )

    save_sample_grid(best_rendered, figures_dir / "best_samples_grid.png", "Best Test Samples")
    save_sample_grid(worst_rendered, figures_dir / "worst_samples_grid.png", "Worst Test Samples")

    manifest = {"event_file": str(event_file), "best_samples": best_rendered, "worst_samples": worst_rendered}
    with open(output_dir / "paper_sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
