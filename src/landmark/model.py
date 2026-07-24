from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights

from src.landmark.schema import NUM_POINTS

OUTPUT_DIM = NUM_POINTS * 2  # 62


class LandmarkHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


class MobileNetLandmark(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.backbone = backbone.features
        self.head = LandmarkHead(in_features=1280)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


def build_model(
    backbone: str = "mobilenet_v2",
    num_points: int = NUM_POINTS,
    pretrained: bool = True,
) -> nn.Module:
    if num_points != NUM_POINTS:
        raise ValueError(f"This model currently supports {NUM_POINTS} points, got {num_points}")
    if backbone == "mobilenet_v2":
        return MobileNetLandmark(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'mobilenet_v2'.")
