#!/usr/bin/env python3
"""Generic heatmap landmark models for comparison vs HRNet-W48.

Supports:
  - HRNet-W18 / HRNet-W32 / HRNet-W48: high-res branch + conv head (same as main pipeline)
  - ResNet50-SimpleBaseline / MobileNetV3-Small: 1/32 features + deconv head (SimpleBaseline style)
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

from src.landmark.schema import NUM_POINTS


class HRNetStyleModel(nn.Module):
    """HRNet family: extract high-res branch (1/4 resolution) + conv heatmap head."""

    def __init__(
        self,
        backbone_name: str = "hrnet_w18",
        num_points: int = NUM_POINTS,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        head_channels: int = 256,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        use_timm_pretrained = pretrained and not pretrained_path
        self.backbone = timm.create_model(backbone_name, pretrained=use_timm_pretrained)
        if pretrained_path:
            weight_path = Path(pretrained_path)
            if weight_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(weight_path)
            else:
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=True)

        # Probe high-res branch channels
        in_channels = self._probe_branch_channels()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(head_channels, num_points, kernel_size=1),
        )

    def _probe_branch_channels(self) -> int:
        self.backbone.eval()
        x = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            f = self.backbone.conv1(x)
            f = self.backbone.bn1(f)
            f = self.backbone.act1(f)
            f = self.backbone.conv2(f)
            f = self.backbone.bn2(f)
            f = self.backbone.act2(f)
            branches = self.backbone.stages(f)
        return int(branches[0].shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone.conv1(x)
        f = self.backbone.bn1(f)
        f = self.backbone.act1(f)
        f = self.backbone.conv2(f)
        f = self.backbone.bn2(f)
        f = self.backbone.act2(f)
        branches = self.backbone.stages(f)
        return self.head(branches[0])

    def set_low_level_frozen(self, frozen: bool) -> None:
        for module in (self.backbone.conv1, self.backbone.bn1, self.backbone.conv2, self.backbone.bn2):
            for param in module.parameters():
                param.requires_grad = not frozen

    def apply_freeze_state(self) -> None:
        pass


class SimpleBaselineModel(nn.Module):
    """ResNet/MobileNet family: forward_features (1/32) + deconv head (SimpleBaseline style)."""

    def __init__(
        self,
        backbone_name: str,
        num_points: int = NUM_POINTS,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        head_channels: int = 256,
        num_deconv: int = 3,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        use_timm_pretrained = pretrained and not pretrained_path
        self.backbone = timm.create_model(backbone_name, pretrained=use_timm_pretrained)
        if pretrained_path:
            weight_path = Path(pretrained_path)
            if weight_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(weight_path)
            else:
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=True)

        in_channels = self._probe_features_channels()

        deconv = []
        prev = in_channels
        for _ in range(num_deconv):
            deconv.append(nn.ConvTranspose2d(prev, head_channels, kernel_size=4, stride=2, padding=1, output_padding=0))
            deconv.append(nn.BatchNorm2d(head_channels))
            deconv.append(nn.ReLU(inplace=True))
            prev = head_channels
        self.deconv_layers = nn.Sequential(*deconv)

        self.final_layer = nn.Conv2d(head_channels, num_points, kernel_size=1)

    def _probe_features_channels(self) -> int:
        self.backbone.eval()
        x = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            f = self.backbone.forward_features(x)
        return int(f.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        out = self.deconv_layers(features)
        return self.final_layer(out)

    def set_low_level_frozen(self, frozen: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not frozen

    def apply_freeze_state(self) -> None:
        pass


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", config)
    backbone = str(model_cfg.get("backbone", "hrnet_w48"))
    model_type = str(model_cfg.get("model_type", "hrnet"))

    num_points = int(model_cfg.get("num_points", NUM_POINTS))
    pretrained = bool(model_cfg.get("pretrained", True))
    pretrained_path = model_cfg.get("pretrained_path")
    head_channels = int(model_cfg.get("head_channels", 256))
    head_dropout = float(model_cfg.get("head_dropout", 0.2))

    if model_type == "simplebaseline":
        return SimpleBaselineModel(
            backbone_name=backbone,
            num_points=num_points,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            head_channels=head_channels,
            num_deconv=int(model_cfg.get("num_deconv", 3)),
            head_dropout=head_dropout,
        )
    return HRNetStyleModel(
        backbone_name=backbone,
        num_points=num_points,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        head_channels=head_channels,
        head_dropout=head_dropout,
    )
