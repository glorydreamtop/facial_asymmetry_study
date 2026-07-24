"""
Visualization utilities for facial landmarks and symmetry analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.landmark.schema import (
    MIDLINE_INDICES,
    POINT_NAMES_31,
    REGIONS,
    SYMMETRY_PAIRS,
    NUM_POINTS,
)
from src.symmetry.axis import SymmetryAxis
from src.symmetry.metrics import SymmetryReport

COLOR_BILATERAL = {
    "eyebrow": "#4CAF50",
    "eye": "#2196F3",
    "nose": "#FF9800",
    "mouth": "#E91E63",
    "jaw": "#9C27B0",
}
COLOR_MIDLINE = "#F44336"
COLOR_AXIS = "#FF0000"
COLOR_MIDPOINT = "#FFD700"


def _get_point_color(idx: int) -> str:
    if idx in MIDLINE_INDICES:
        return COLOR_MIDLINE
    for region, indices in REGIONS.items():
        if idx in indices:
            return COLOR_BILATERAL.get(region, "#666666")
    return "#666666"


def plot_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    point_size: int = 8,
    show_labels: bool = False,
    title: str = "Landmarks",
) -> plt.Figure:
    """
    Plot facial landmarks on image.

    Args:
        image: (H, W, 3) RGB image.
        landmarks: (N, 2) array.
        valid_mask: (N,) boolean.
        ax: matplotlib axis (creates new if None).
        point_size: Size of scatter points.
        show_labels: Whether to draw point names.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    if valid_mask is None:
        valid_mask = np.ones(len(landmarks), dtype=bool)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    else:
        fig = ax.figure

    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    for i in range(len(landmarks)):
        if not valid_mask[i]:
            continue
        color = _get_point_color(i)
        ax.scatter(
            landmarks[i, 0], landmarks[i, 1],
            c=color, s=point_size, zorder=5, edgecolors="white", linewidths=0.5,
        )
        if show_labels:
            ax.annotate(
                POINT_NAMES_31[i],
                (landmarks[i, 0], landmarks[i, 1]),
                fontsize=6, color="white",
                xytext=(3, 3), textcoords="offset points",
            )

    for l_idx, r_idx in SYMMETRY_PAIRS:
        if valid_mask[l_idx] and valid_mask[r_idx]:
            ax.plot(
                [landmarks[l_idx, 0], landmarks[r_idx, 0]],
                [landmarks[l_idx, 1], landmarks[r_idx, 1]],
                "--", color="gray", alpha=0.4, linewidth=1,
            )

    return fig


def plot_symmetry_analysis(
    image: np.ndarray,
    landmarks: np.ndarray,
    axis: SymmetryAxis,
    report: Optional[SymmetryReport] = None,
    valid_mask: Optional[np.ndarray] = None,
    title: str = "Symmetry Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Comprehensive symmetry analysis visualization.

    Layout:
      - Left: Image with landmarks + symmetry axis
      - Right: Deviation heatmap or bar chart
    """
    if valid_mask is None:
        valid_mask = np.ones(len(landmarks), dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title("Landmarks + Symmetry Axis")
    ax1.axis("off")

    for i in range(NUM_POINTS):
        if not valid_mask[i]:
            continue
        color = _get_point_color(i)
        ax1.scatter(landmarks[i, 0], landmarks[i, 1], c=color, s=6, zorder=5)

    for l_idx, r_idx in SYMMETRY_PAIRS:
        if valid_mask[l_idx] and valid_mask[r_idx]:
            mid = (landmarks[l_idx] + landmarks[r_idx]) / 2
            ax1.plot(
                [landmarks[l_idx, 0], landmarks[r_idx, 0]],
                [landmarks[l_idx, 1], landmarks[r_idx, 1]],
                "-", color="gray", alpha=0.3, linewidth=1,
            )
            ax1.scatter(mid[0], mid[1], c=COLOR_MIDPOINT, s=20, marker="x", zorder=6)

    if axis is not None:
        h, w = image.shape[:2]
        x_vals = np.array([0, w])
        y_vals = axis.slope * x_vals + axis.intercept
        ax1.plot(x_vals, y_vals, "-", color=COLOR_AXIS, linewidth=2, label=f"Axis ({axis.angle_deg:.1f}deg)")
        ax1.legend()

    ax2 = axes[1]
    if report is not None and report.deviations is not None:
        valid_deviations = []
        labels = []
        colors = []
        for i in range(NUM_POINTS):
            if valid_mask[i] and not np.isnan(report.deviations[i]):
                valid_deviations.append(report.deviations[i])
                labels.append(POINT_NAMES_31[i])
                colors.append(_get_point_color(i))

        bars = ax2.barh(range(len(valid_deviations)), valid_deviations, color=colors)
        ax2.set_yticks(range(len(valid_deviations)))
        ax2.set_yticklabels(labels, fontsize=6)
        ax2.set_xlabel("Deviation (pixels)")
        ax2.set_title("Per-Landmark Asymmetry")
        ax2.invert_yaxis()

    ax3 = axes[2]
    if report is not None and report.region_scores:
        regions = list(report.region_scores.keys())
        scores = [report.region_scores[r] for r in regions]
        colors_reg = [COLOR_BILATERAL.get(r, "#666") for r in regions]

        bars = ax3.barh(regions, scores, color=colors_reg)
        ax3.set_xlabel("Normalized Deviation")
        ax3.set_title("Regional Symmetry Scores")

        overall = report.overall_score
        ax3.text(
            0.95, 0.05, f"Overall: {overall:.1f}/100",
            transform=ax3.transAxes, fontsize=12, fontweight="bold",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_batch_results(
    images: List[np.ndarray],
    landmarks_list: List[np.ndarray],
    axes_list: List[SymmetryAxis],
    scores: List[float],
    max_cols: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a grid of faces with their symmetry axes.
    """
    n = len(images)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i in range(n):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(images[i])
            ax.axis("off")

            if axes_list[i] is not None:
                h, w = images[i].shape[:2]
                x_vals = np.array([0, w])
                axis = axes_list[i]
                y_vals = axis.slope * x_vals + axis.intercept
                ax.plot(x_vals, y_vals, "-", color=COLOR_AXIS, linewidth=2)

            if i < len(scores):
                ax.set_title(f"Score: {scores[i]:.1f}", fontsize=10)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_data_distribution(
    landmarks_list: List[np.ndarray],
    valid_masks: List[np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the distribution of all landmarks across the dataset.

    Useful for understanding the spatial distribution and detecting annotation outliers.
    """
    all_points = {name: [] for name in POINT_NAMES_31}

    for lm, mask in zip(landmarks_list, valid_masks):
        for i in range(NUM_POINTS):
            if mask[i]:
                all_points[POINT_NAMES_31[i]].append(lm[i])

    fig, axes = plt.subplots(7, 5, figsize=(20, 28))
    axes = axes.flatten()

    for i, name in enumerate(POINT_NAMES_31):
        if i < len(axes):
            pts = np.array(all_points[name])
            if len(pts) > 0:
                axes[i].scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.3, c=_get_point_color(i))
            axes[i].set_title(name, fontsize=8)
            axes[i].set_aspect("equal")
            axes[i].tick_params(labelsize=6)

    for i in range(NUM_POINTS, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
