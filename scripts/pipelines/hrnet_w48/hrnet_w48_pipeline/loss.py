from __future__ import annotations

import math

import torch
import torch.nn as nn

from .metrics import spatial_expectation2d


def wing_loss(diff: torch.Tensor, w: float, epsilon: float) -> torch.Tensor:
    abs_diff = diff.abs()
    constant = w - w * math.log1p(w / epsilon)
    return torch.where(
        abs_diff < w,
        w * torch.log1p(abs_diff / epsilon),
        abs_diff - constant,
    )


class CombinedHeatmapLoss(nn.Module):
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        coord_weight: float = 0.05,
        wing_w: float = 10.0,
        wing_epsilon: float = 2.0,
    ) -> None:
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.wing_w = wing_w
        self.wing_epsilon = wing_epsilon

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask_4d = mask[:, :, None, None].float()
        heatmap_residual = (pred_heatmaps - target_heatmaps) ** 2
        heatmap_loss = (heatmap_residual * mask_4d).sum() / mask_4d.sum().clamp_min(1.0)

        coord_loss = torch.tensor(0.0, device=pred_heatmaps.device)
        if self.coord_weight > 0:
            pred_coords = spatial_expectation2d(pred_heatmaps)
            coord_residual = wing_loss(pred_coords - target_coords, self.wing_w, self.wing_epsilon)
            coord_loss = (coord_residual * mask[:, :, None]).sum() / mask.sum().clamp_min(1.0)

        total = self.heatmap_weight * heatmap_loss + self.coord_weight * coord_loss
        return {
            "loss": total,
            "heatmap_loss": heatmap_loss.detach(),
            "coord_loss": coord_loss.detach(),
        }
