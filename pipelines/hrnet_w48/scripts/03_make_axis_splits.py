#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _bootstrap import setup_project_paths

setup_project_paths()

from src.symmetry.axis_regression import parse_axis_labelme_json


def write_split(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create split files directly from axis annotation data")
    parser.add_argument("--axis_dir", default="对称轴数据")
    parser.add_argument("--output_dir", default="data/splits_axis")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    axis_dir = Path(args.axis_dir)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    samples = []
    for axis_json in sorted(axis_dir.rglob("*.json")):
        ann = parse_axis_labelme_json(str(axis_json))
        if ann is None:
            continue
        image_path = Path(ann.image_path)
        try:
            rel_image_path = str(image_path.relative_to(axis_dir)).replace("\\", "/")
        except ValueError:
            rel_image_path = image_path.name
        samples.append(
            {
                "image_path": rel_image_path,
                "axis_json": str(axis_json.relative_to(axis_dir)).replace("\\", "/"),
            }
        )

    rng.shuffle(samples)
    n_total = len(samples)
    n_train = int(round(n_total * args.train_ratio))
    n_val = int(round(n_total * args.val_ratio))
    n_train = min(n_train, n_total)
    n_val = min(n_val, max(0, n_total - n_train))
    n_test = n_total - n_train - n_val

    train_items = samples[:n_train]
    val_items = samples[n_train:n_train + n_val]
    test_items = samples[n_train + n_val:]

    write_split(output_dir / "train.json", train_items)
    write_split(output_dir / "val.json", val_items)
    write_split(output_dir / "test.json", test_items)

    print(f"Saved axis-only splits to {output_dir}")
    print(f"  total: {n_total}")
    print(f"  train: {len(train_items)}")
    print(f"  val:   {len(val_items)}")
    print(f"  test:  {len(test_items)}")


if __name__ == "__main__":
    main()
