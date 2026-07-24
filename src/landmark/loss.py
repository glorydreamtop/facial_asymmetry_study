from __future__ import annotations

import math

import torch
import torch.nn as nn


def wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w: float = 10.0,
    epsilon: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Wing loss for facial landmark regression.

    The loss function is:
        if |x| < w:  w * log(1 + |x| / epsilon)
        else:        |x| - C
    where C = w - w * log(1 + w / epsilon).

    Parameters
    ----------
    pred : torch.Tensor  (B, N, 2) or (B, N*2)
    target : torch.Tensor  same shape as pred
    w : float  nonlinear region width
    epsilon : float  curvature scale
    reduction : str  'mean' | 'sum' | 'none'

    Returns
    -------
    torch.Tensor  scalar loss if reduction != 'none'
    """
    if pred.ndim == 2 and target.ndim == 3:
        pred = pred.view(pred.shape[0], -1, 2)
    elif pred.ndim == 3 and target.ndim == 2:
        target = target.view(target.shape[0], -1, 2)
    elif pred.ndim == 2 and target.ndim == 2:
        pred = pred.view(pred.shape[0], -1, 2)
        target = target.view(target.shape[0], -1, 2)

    diff = pred - target
    abs_diff = diff.abs()

    C = w - w * math.log1p(w / epsilon)

    inside = abs_diff < w
    outside = ~inside

    loss = torch.where(
        inside,
        w * torch.log1p(abs_diff / epsilon),
        abs_diff - C,
    )

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class WeightedWingLoss(nn.Module):
    """Wing loss weighted by a per-landmark visibility mask."""

    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = wing_loss(pred, target, self.w, self.epsilon, reduction="none")
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.float()
            loss = loss * mask
            normalizer = mask.expand_as(loss).sum().clamp_min(1.0)
            return loss.sum() / normalizer
        return loss.mean()


class CombinedLandmarkLoss(nn.Module):
    """Wing loss + L2 weight regularisation on the output layer."""

    def __init__(
        self,
        w: float = 10.0,
        epsilon: float = 2.0,
        l2_reg: float = 1e-5,
    ):
        super().__init__()
        self.wing = WeightedWingLoss(w=w, epsilon=epsilon)
        self.l2_reg = l2_reg

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        wing = self.wing(pred, target, mask)

        l2 = torch.tensor(0.0, device=pred.device)
        if self.l2_reg > 0 and model is not None:
            for param in model.parameters():
                l2 = l2 + param.pow(2).sum()
            l2 = self.l2_reg * l2

        return wing + l2
