from __future__ import annotations

import numpy as np
import torch


def _build_coordinate_grid(heat_h: int, heat_w: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.5 / heat_w, 1.0 - 0.5 / heat_w, heat_w, device=device, dtype=dtype)
    ys = torch.linspace(0.5 / heat_h, 1.0 - 0.5 / heat_h, heat_h, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def decode_heatmaps_argmax(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_points, heat_h, heat_w = heatmaps.shape
    flat = heatmaps.view(batch, num_points, -1)
    scores, indices = flat.max(dim=-1)
    xs = (indices % heat_w).float()
    ys = torch.div(indices, heat_w, rounding_mode="floor").float()

    coords = torch.zeros(batch, num_points, 2, device=heatmaps.device, dtype=torch.float32)
    coords[..., 0] = (xs + 0.5) / heat_w
    coords[..., 1] = (ys + 0.5) / heat_h
    return coords, scores


def decode_heatmaps_local_softargmax(
    heatmaps: torch.Tensor,
    window_size: int = 5,
    temperature: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_points, heat_h, heat_w = heatmaps.shape
    coords, scores = decode_heatmaps_argmax(heatmaps)

    if window_size <= 1:
        return coords, scores

    radius = max(1, window_size // 2)
    xx, yy = _build_coordinate_grid(heat_h, heat_w, heatmaps.device, heatmaps.dtype)
    xx = xx.view(1, 1, heat_h, heat_w)
    yy = yy.view(1, 1, heat_h, heat_w)

    flat = heatmaps.view(batch, num_points, -1)
    peak_x = (flat.argmax(dim=-1) % heat_w).view(batch, num_points, 1, 1)
    peak_y = torch.div(flat.argmax(dim=-1), heat_w, rounding_mode="floor").view(batch, num_points, 1, 1)

    grid_x = torch.arange(heat_w, device=heatmaps.device).view(1, 1, 1, heat_w)
    grid_y = torch.arange(heat_h, device=heatmaps.device).view(1, 1, heat_h, 1)
    local_mask = (grid_x - peak_x).abs() <= radius
    local_mask = local_mask & ((grid_y - peak_y).abs() <= radius)

    masked = heatmaps.masked_fill(~local_mask, float("-inf"))
    probs = torch.softmax(masked.view(batch, num_points, -1) * temperature, dim=-1)
    probs = probs.view(batch, num_points, heat_h, heat_w)

    coords = torch.zeros(batch, num_points, 2, device=heatmaps.device, dtype=torch.float32)
    coords[..., 0] = (probs * xx).sum(dim=(-2, -1)).float()
    coords[..., 1] = (probs * yy).sum(dim=(-2, -1)).float()
    return coords, scores


def spatial_expectation2d(heatmaps: torch.Tensor) -> torch.Tensor:
    batch, num_points, heat_h, heat_w = heatmaps.shape
    flat = heatmaps.view(batch, num_points, -1)
    probs = torch.softmax(flat, dim=-1)

    xx, yy = _build_coordinate_grid(heat_h, heat_w, heatmaps.device, heatmaps.dtype)
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return torch.matmul(probs, grid)


def compute_masked_nme(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    ref_point_indices: tuple[int, int] = (4, 5),
) -> np.ndarray:
    nme = np.full(pred.shape[0], np.nan, dtype=np.float32)
    left_idx, right_idx = ref_point_indices

    for i in range(pred.shape[0]):
        valid = mask[i] > 0.5
        if valid.sum() == 0:
            continue
        if not (valid[left_idx] and valid[right_idx]):
            continue

        ref_dist = np.linalg.norm(target[i, left_idx] - target[i, right_idx])
        if ref_dist <= 1e-8:
            continue

        point_errors = np.linalg.norm(pred[i, valid] - target[i, valid], axis=1)
        nme[i] = point_errors.mean() / ref_dist

    return nme


def detection_rate_at_threshold(
    nme: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[float, float]:
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    valid = nme[~np.isnan(nme)]
    if len(valid) == 0:
        return {t: 0.0 for t in thresholds}
    return {t: float((valid < t).mean()) for t in thresholds}


def compute_pixel_metrics(per_point_errors, thresholds=None):
    if thresholds is None:
        thresholds = [5, 10, 15, 20, 25, 30, 40]
    all_errors = []
    for errors in per_point_errors:
        all_errors.extend(errors)

    if not all_errors:
        return None

    all_errors = np.asarray(all_errors, dtype=np.float32)
    return {
        "mre_px": float(np.mean(all_errors)),
        "std_px": float(np.std(all_errors)),
        "sdr_px": {str(th): float(np.mean(all_errors <= th)) for th in thresholds},
        "n_points": int(len(all_errors)),
    }
