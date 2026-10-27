"""Dataset splitting utilities."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def random_split(
    n_total: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Randomly split indices into train/val/test.

    Args:
        n_total: Total number of samples.
        train_ratio, val_ratio, test_ratio: Split ratios.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:n_train + n_val])
    test_idx = sorted(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def split_with_partial_in_train(
    complete_indices: List[int],
    partial_indices: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    force_train_indices: List[int] | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Split usable samples while forcing all partial labels into train.

    The overall ratio is computed on all usable samples. Validation and test
    sets are drawn only from fully annotated samples so that evaluation keeps
    using gold-standard annotations.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n_complete = len(complete_indices)
    n_partial = len(partial_indices)
    n_total = n_complete + n_partial
    force_train_set = set(force_train_indices or [])

    all_indices = set(complete_indices) | set(partial_indices)
    unknown_force_indices = force_train_set - all_indices
    if unknown_force_indices:
        raise ValueError(
            "Force-train indices contain items outside the usable annotation set: "
            f"{sorted(unknown_force_indices)[:10]}"
        )

    force_train_complete = [idx for idx in complete_indices if idx in force_train_set]

    n_val = round(n_total * val_ratio)
    n_test = round(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    fixed_train_count = n_partial + len(force_train_complete)
    if fixed_train_count > n_train:
        raise ValueError(
            "Forced train samples exceed the train split capacity. "
            f"fixed_train={fixed_train_count}, train_capacity={n_train}"
        )

    available_complete = [idx for idx in complete_indices if idx not in force_train_set]
    n_complete_train = n_train - fixed_train_count
    if n_complete_train < 0 or n_complete_train + n_val + n_test > len(available_complete):
        raise ValueError(
            "Not enough complete samples to satisfy the requested split. "
            f"available_complete={len(available_complete)}, needed={n_complete_train + n_val + n_test}"
        )

    rng = random.Random(seed)
    complete_shuffled = available_complete.copy()
    partial_shuffled = partial_indices.copy()
    rng.shuffle(complete_shuffled)
    rng.shuffle(partial_shuffled)

    train_complete = complete_shuffled[:n_complete_train]
    val_idx = complete_shuffled[n_complete_train:n_complete_train + n_val]
    test_idx = complete_shuffled[n_complete_train + n_val:n_complete_train + n_val + n_test]
    train_idx = partial_shuffled + force_train_complete + train_complete

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx


def save_split(
    annotations: List[Dict],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    output_dir: str,
):
    """Save train/val/test split information to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _indices_to_list(indices):
        return [
            {
                "idx": i,
                "image_id": i + 1,
                "image_path": annotations[i]["image_path"],
                "sample_type": annotations[i].get("sample_type", "unknown"),
                "num_valid": annotations[i].get("num_valid"),
            }
            for i in indices
        ]

    splits = {
        "train": _indices_to_list(train_idx),
        "val": _indices_to_list(val_idx),
        "test": _indices_to_list(test_idx),
    }

    for name, data in splits.items():
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  {name}: {len(data)} samples -> {path}")

    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"\nTotal: {total} samples")
    print(f"  Train: {len(train_idx)} ({100*len(train_idx)/total:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({100*len(val_idx)/total:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({100*len(test_idx)/total:.1f}%)")
