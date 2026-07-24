"""
Symmetry axis fitting.

Given facial landmark midpoints, fits the facial symmetry axis
using RANSAC for robustness against outliers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import RANSACRegressor


@dataclass
class SymmetryAxis:
    slope: float
    intercept: float
    angle_deg: float
    center: Tuple[float, float]
    inlier_mask: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    n_inliers: int = 0
    n_total: int = 0


def _fit_line_ransac(
    points: np.ndarray,
    min_samples: int = 2,
    residual_threshold: float = 5.0,
    max_trials: int = 1000,
) -> Tuple[float, float, np.ndarray]:
    """
    Fit a line using RANSAC: y = slope * x + intercept.

    Args:
        points: (N, 2) array of (x, y) coordinates.
        min_samples: Minimum samples for RANSAC.
        residual_threshold: Max residual for inlier classification.
        max_trials: Maximum RANSAC iterations.

    Returns:
        (slope, intercept, inlier_mask)
    """
    if len(points) < 2:
        return 0.0, 0.0, np.ones(len(points), dtype=bool)

    # Remove NaN rows
    nan_mask = ~np.any(np.isnan(points), axis=1)
    clean_points = points[nan_mask]

    if len(clean_points) < 2:
        return 0.0, 0.0, np.ones(len(points), dtype=bool)

    X = clean_points[:, 0].reshape(-1, 1)
    y = clean_points[:, 1]

    reg = RANSACRegressor(
        min_samples=min(min_samples, len(clean_points)),
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=42,
    )
    reg.fit(X, y)

    slope = reg.estimator_.coef_[0]
    intercept = reg.estimator_.intercept_

    # Build full inlier mask (including NaN positions as outliers)
    full_mask = np.zeros(len(points), dtype=bool)
    full_mask[nan_mask] = reg.inlier_mask_

    return float(slope), float(intercept), full_mask


def fit_symmetry_axis(
    midpoints: np.ndarray,
    use_ransac: bool = True,
    min_samples: int = 2,
    residual_threshold: float = 5.0,
    max_trials: int = 1000,
    weights: Optional[np.ndarray] = None,
) -> SymmetryAxis:
    """
    Fit the facial symmetry axis from landmark midpoints.

    Args:
        midpoints: (M, 2) array of midpoint coordinates.
        use_ransac: If True, use RANSAC; otherwise use least squares.
        min_samples: Minimum samples for RANSAC.
        residual_threshold: Max residual for RANSAC inliers.
        max_trials: Max RANSAC iterations.
        weights: (M,) optional weights for each midpoint (not used in RANSAC,
                 but if not using RANSAC, used for weighted least squares).

    Returns:
        SymmetryAxis object with fitted parameters.
    """
    # Remove NaN
    nan_mask = ~np.any(np.isnan(midpoints), axis=1)
    clean_pts = midpoints[nan_mask]

    if len(clean_pts) < 2:
        return SymmetryAxis(
            slope=0.0, intercept=0.0, angle_deg=0.0,
            center=(0.0, 0.0), n_inliers=0, n_total=len(midpoints),
        )

    # 策略一：RANSAC（默认）—— 对离群点鲁棒，适合标注有噪声的情况
    # 策略二：加权最小二乘 —— 通过weights参数控制各点权重
    if use_ransac:
        slope, intercept, inlier_mask = _fit_line_ransac(
            midpoints, min_samples, residual_threshold, max_trials
        )
    else:
        if weights is not None and nan_mask.sum() == len(weights):
            w = weights[nan_mask]
        else:
            w = None
        A = np.vstack([clean_pts[:, 0], np.ones(len(clean_pts))]).T
        if w is not None:
            W = np.diag(w)
            coeffs = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ clean_pts[:, 1]
        else:
            coeffs = np.linalg.lstsq(A, clean_pts[:, 1], rcond=None)[0]
        slope, intercept = coeffs[0], coeffs[1]
        inlier_mask = np.ones(len(midpoints), dtype=bool)

    # 将斜率转换为角度（度），便于观察对称轴偏转
    angle_deg = float(np.degrees(np.arctan(slope)))

    # 对称轴中心点 = 所有中点的平均位置
    center = float(np.nanmean(midpoints[:, 0])), float(np.nanmean(midpoints[:, 1]))

    # 计算每个中点到拟合直线的垂直距离（残差）
    # 残差越大说明该对称点对越不对称
    residuals = np.full(len(midpoints), np.nan)
    valid = ~np.any(np.isnan(midpoints), axis=1)
    if valid.any():
        y_pred = slope * midpoints[valid, 0] + intercept
        residuals[valid] = np.abs(midpoints[valid, 1] - y_pred)

    n_inliers = int(inlier_mask.sum()) if use_ransac else len(clean_pts)

    return SymmetryAxis(
        slope=slope,
        intercept=intercept,
        angle_deg=angle_deg,
        center=center,
        inlier_mask=inlier_mask,
        residuals=residuals,
        n_inliers=n_inliers,
        n_total=len(midpoints),
    )


def compute_symmetric_deviation(
    landmarks: np.ndarray,
    axis: SymmetryAxis,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute perpendicular distance from each landmark to the symmetry axis.

    For a line y = mx + b, the perpendicular distance from point (x0, y0) is:
      d = |mx0 - y0 + b| / sqrt(m^2 + 1)

    Args:
        landmarks: (N, 2) array.
        axis: Fitted SymmetryAxis.
        valid_mask: (N,) boolean mask.

    Returns:
        (N,) array of perpendicular distances. NaN for invalid points.
    """
    if valid_mask is None:
        valid_mask = np.ones(len(landmarks), dtype=bool)

    m = axis.slope
    b = axis.intercept
    denom = np.sqrt(m ** 2 + 1)

    x = landmarks[:, 0]
    y = landmarks[:, 1]
    dist = np.abs(m * x - y + b) / denom
    dist[~valid_mask] = np.nan

    return dist


def axis_from_two_points(p1: np.ndarray, p2: np.ndarray) -> SymmetryAxis:
    """
    Create a symmetry axis from two points (e.g. glabella and subnasale).
    """
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-8)
    intercept = p1[1] - slope * p1[0]
    angle = float(np.degrees(np.arctan(slope)))
    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    return SymmetryAxis(
        slope=slope, intercept=intercept, angle_deg=angle,
        center=center, n_inliers=2, n_total=2,
    )
