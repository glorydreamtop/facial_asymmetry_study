from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file

from src.landmark.schema import NUM_POINTS


class HRNetW48HeatmapModel(nn.Module):
    def __init__(
        self,
        num_points: int = NUM_POINTS,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        head_channels: int = 256,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        use_timm_pretrained = pretrained and not pretrained_path
        self.backbone = timm.create_model("hrnet_w48", pretrained=use_timm_pretrained)
        if pretrained_path:
            weight_path = Path(pretrained_path)
            if weight_path.suffix == ".safetensors":
                state_dict = load_file(weight_path)
            else:
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=True)
        self.low_level_frozen = False

        self.head = nn.Sequential(
            nn.Conv2d(48, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(head_channels, num_points, kernel_size=1),
        )

    def extract_high_res_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        branches = self.backbone.stages(x)
        return branches[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.extract_high_res_feature(x)
        return self.head(feature)

    def set_low_level_frozen(self, frozen: bool) -> None:
        self.low_level_frozen = frozen
        modules = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.conv2,
            self.backbone.bn2,
            self.backbone.layer1,
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = not frozen

    def apply_freeze_state(self) -> None:
        if not self.low_level_frozen:
            return
        modules = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.conv2,
            self.backbone.bn2,
            self.backbone.layer1,
        ]
        for module in modules:
            module.eval()


def build_model(config: dict) -> HRNetW48HeatmapModel:
    return HRNetW48HeatmapModel(
        num_points=int(config.get("num_points", NUM_POINTS)),
        pretrained=bool(config.get("pretrained", True)),
        pretrained_path=config.get("pretrained_path"),
        head_channels=int(config.get("head_channels", 256)),
        head_dropout=float(config.get("head_dropout", 0.2)),
    )
