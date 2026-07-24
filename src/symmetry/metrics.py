"""
Symmetry quantification metrics.

Computes various symmetry scores from detected landmarks and fitted axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.landmark.schema import (
    MIDLINE_INDICES,
    NUM_POINTS,
    POINT_NAMES_31,
    REGIONS,
    SYMMETRY_PAIRS,
)
from src.symmetry.axis import SymmetryAxis, compute_symmetric_deviation
from src.symmetry.pairs import compute_midpoints


@dataclass
class SymmetryReport:
    """
    Per-image symmetry analysis report.
    """
    image_name: str = ""
    axis: Optional[SymmetryAxis] = None
    deviations: Optional[np.ndarray] = None
    region_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    pair_asymmetry: Dict[str, float] = field(default_factory=dict)
    midline_deviation: Dict[str, float] = field(default_factory=dict)


def _normalize_deviations(
    deviations: np.ndarray,
    landmarks: np.ndarray,
    method: str = "interocular",
) -> np.ndarray:
    """
    Normalize deviations by a facial reference distance.

    Args:
        deviations: (N,) raw pixel deviations.
        landmarks: (N, 2) landmark coordinates.
        method: "interocular" (eye distance) or "face_width" (face width).

    Returns:
        Normalized deviations (unitless).
    """
    # 以内眼角间距（interocular distance）为基准归一化
    # 这样不同分辨率的图片之间可以比较对称性分数
    if method == "interocular":
        en_l = landmarks[8]
        en_r = landmarks[9]
        ref_dist = np.linalg.norm(en_l - en_r)
    elif method == "face_width":
        go_l = landmarks[28]
        go_r = landmarks[29]
        ref_dist = np.linalg.norm(go_l - go_r)
    else:
        ref_dist = 1.0

    if ref_dist < 1e-6:
        return deviations

    return deviations / ref_dist


def analyze_symmetry(
    landmarks: np.ndarray,
    axis: SymmetryAxis,
    valid_mask: Optional[np.ndarray] = None,
    normalize_by: str = "interocular",
) -> SymmetryReport:
    """
    Perform full symmetry analysis.

    Args:
        landmarks: (N, 2) landmark coordinates.
        axis: Fitted symmetry axis.
        valid_mask: (31,) boolean mask.
        normalize_by: Normalization method.

    Returns:
        SymmetryReport with all scores.
    """
    if valid_mask is None:
        valid_mask = np.ones(NUM_POINTS, dtype=bool)

    report = SymmetryReport()
    report.axis = axis

    raw_deviations = compute_symmetric_deviation(landmarks, axis, valid_mask)
    report.deviations = raw_deviations.copy()

    norm_deviations = _normalize_deviations(raw_deviations, landmarks, normalize_by)

    for region, indices in REGIONS.items():
        valid_in_region = [i for i in indices if valid_mask[i]]
        if valid_in_region:
            rms = np.sqrt(np.nanmean(norm_deviations[valid_in_region] ** 2))
            report.region_scores[region] = float(rms)
        else:
            report.region_scores[region] = np.nan

    for l_idx, r_idx in SYMMETRY_PAIRS:
        if valid_mask[l_idx] and valid_mask[r_idx]:
            pair_name = f"{POINT_NAMES_31[l_idx]}_{POINT_NAMES_31[r_idx]}"
            m = axis.slope
            b = axis.intercept
            denom = np.sqrt(m ** 2 + 1)
            d_l = (landmarks[l_idx, 1] - (m * landmarks[l_idx, 0] + b)) / denom
            d_r = (landmarks[r_idx, 1] - (m * landmarks[r_idx, 0] + b)) / denom
            report.pair_asymmetry[pair_name] = float(abs(d_l) + abs(d_r))

    for idx in MIDLINE_INDICES:
        if valid_mask[idx]:
            report.midline_deviation[POINT_NAMES_31[idx]] = float(
                norm_deviations[idx]
            )

    # 各区域对称性权重配置
    # 眼睛和鼻子区域权重较高（对面部对称性影响大）
    weights = {
        "eyebrow": 1.0, "eye": 1.5, "nose": 1.5,
        "mouth": 1.0, "jaw": 0.5,
    }

    weighted_sum = 0.0
    weight_sum = 0.0
    for region, score in report.region_scores.items():
        w = weights.get(region, 1.0)
        if not np.isnan(score):
            weighted_sum += w * score
            weight_sum += w

    # 总体对称性评分（0-100分）
    # 使用指数映射：score = 100 * exp(-alpha * avg_deviation)
    # alpha=20 意味着平均归一化偏差0.05 → 评分约37分（中等对称）
    # 平均归一化偏差0.01 → 评分约82分（高度对称）
    avg_norm_deviation = weighted_sum / max(weight_sum, 1e-8)
    alpha = 20.0
    report.overall_score = float(100.0 * np.exp(-alpha * avg_norm_deviation))

    return report


def summarize_report(report: SymmetryReport) -> Dict:
    """Convert a SymmetryReport to a JSON-serializable dict."""
    return {
        "overall_score": round(report.overall_score, 2),
        "axis_angle": round(report.axis.angle_deg, 2) if report.axis else None,
        "axis_center": [round(c, 1) for c in report.axis.center] if report.axis else None,
        "region_scores": {
            k: round(v, 4) for k, v in report.region_scores.items()
            if not np.isnan(v)
        },
        "pair_asymmetry": {
            k: round(v, 4) for k, v in report.pair_asymmetry.items()
        },
        "midline_deviation": {
            k: round(v, 4) for k, v in report.midline_deviation.items()
        },
        "n_inliers": report.axis.n_inliers if report.axis else 0,
        "n_total_midpoints": report.axis.n_total if report.axis else 0,
    }
