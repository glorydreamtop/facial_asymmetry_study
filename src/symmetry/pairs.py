"""
Symmetric pair definitions and midpoint computation.

For facial symmetry analysis, we:
  1. Take each bilateral point pair (L, R)
  2. Compute the midpoint of each pair
  3. Use these midpoints as anchor points for the symmetry axis

We also handle midline points (self-symmetric).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.landmark.schema import (
    ALL_SYMMETRY_GROUPS,
    MIDLINE_INDICES,
    POINT_NAMES_31,
    REGIONS,
    SYMMETRY_PAIRS,
    NUM_POINTS,
)


def compute_midpoints(
    landmarks: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute midpoints for all symmetry pairs.

    Args:
        landmarks: (N, 2) array of landmark coordinates (N=31).
        valid_mask: (N,) boolean mask, True for valid points.

    Returns:
        (M, 2) array of midpoint coordinates, where M = len(symmetry_pairs) for
        bilateral pairs + len(midline_indices) for self-paired midline points.
        Returns NaN for pairs where either point is invalid.
        Also returns list of pair labels.
    """
    if valid_mask is None:
        valid_mask = np.ones(NUM_POINTS, dtype=bool)

    midpoints = []
    pair_labels = []

    # 双侧对称点：取左右坐标的平均值作为中点
    # 例如 ebe_L 和 ebe_R 的中点就是眉外侧端的对称中心
    for l_idx, r_idx in SYMMETRY_PAIRS:
        both_valid = valid_mask[l_idx] and valid_mask[r_idx]
        if both_valid:
            mid = (landmarks[l_idx] + landmarks[r_idx]) / 2.0
        else:
            mid = np.array([np.nan, np.nan])
        midpoints.append(mid)
        pair_labels.append(f"{POINT_NAMES_31[l_idx]}_{POINT_NAMES_31[r_idx]}")

    # 正中线点：自身就是"中点"（与自己对称）
    for idx in MIDLINE_INDICES:
        if valid_mask[idx]:
            mid = landmarks[idx].copy()
        else:
            mid = np.array([np.nan, np.nan])
        midpoints.append(mid)
        pair_labels.append(f"{POINT_NAMES_31[idx]}_midline")

    return np.array(midpoints), pair_labels


def get_landmark_pairs() -> List[Tuple[str, str, str]]:
    """Get human-readable landmark pair definitions."""
    pairs = []
    for l_idx, r_idx in SYMMETRY_PAIRS:
        pairs.append((
            f"{POINT_NAMES_31[l_idx]}_{POINT_NAMES_31[r_idx]}",
            POINT_NAMES_31[l_idx],
            POINT_NAMES_31[r_idx],
        ))
    return pairs


def region_symmetry_pairs() -> Dict[str, List[Tuple[int, int]]]:
    """Group symmetry pairs by anatomical region."""
    result = {}
    for region, indices in REGIONS.items():
        pairs = []
        for i in indices:
            if i in MIDLINE_INDICES:
                pairs.append((i, i))
            else:
                for l, r in SYMMETRY_PAIRS:
                    if i == l:
                        pairs.append((l, r))
                    elif i == r:
                        pairs.append((l, r))
        if pairs:
            result[region] = pairs
    return result


def compute_midpoints_by_region(
    landmarks: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute average midpoint per anatomical region.

    Args:
        landmarks: (N, 2) array.
        valid_mask: (N,) boolean mask.

    Returns:
        {region_name: (2,) midpoint array} - average of all pair midpoints in that region.
    """
    if valid_mask is None:
        valid_mask = np.ones(NUM_POINTS, dtype=bool)

    region_pairs = region_symmetry_pairs()
    result = {}

    for region, pairs in region_pairs.items():
        region_mids = []
        for l, r in pairs:
            if valid_mask[l] and valid_mask[r]:
                mid = (landmarks[l] + landmarks[r]) / 2.0
                region_mids.append(mid)
        if region_mids:
            result[region] = np.mean(region_mids, axis=0)
        else:
            result[region] = np.array([np.nan, np.nan])

    return result
