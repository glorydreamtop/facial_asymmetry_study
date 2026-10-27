"""
Data augmentation for facial landmark training.

Applies geometric and photometric transformations
while keeping landmark coordinates consistent.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
from torchvision import transforms as T


class RandomRotation:
    """
    Random rotation with landmark coordinate adjustment.

    Args:
        degrees: Max rotation angle in degrees (plus or minus).
    """

    def __init__(self, degrees: float = 15.0):
        self.degrees = degrees

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        landmarks = sample["landmarks"].clone()
        mask = sample["mask"]

        C, H, W = image.shape
        angle = np.random.uniform(-self.degrees, self.degrees)

        cx, cy = 0.5, 0.5
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        lm = landmarks.numpy()
        lm_centered = lm - np.array([cx, cy])
        lm_rotated = np.zeros_like(lm_centered)
        lm_rotated[:, 0] = lm_centered[:, 0] * cos_a - lm_centered[:, 1] * sin_a
        lm_rotated[:, 1] = lm_centered[:, 0] * sin_a + lm_centered[:, 1] * cos_a
        lm_rotated += np.array([cx, cy])
        sample["landmarks"] = torch.from_numpy(lm_rotated).float()

        img_pil = T.ToPILImage()(image)
        img_rotated = T.functional.rotate(img_pil, angle, interpolation=T.InterpolationMode.BILINEAR)
        sample["image"] = T.ToTensor()(img_rotated)

        return sample


class RandomBrightnessContrast:
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)

        img = T.ToPILImage()(image)
        img = T.functional.adjust_brightness(img, brightness_factor)
        img = T.functional.adjust_contrast(img, contrast_factor)
        sample["image"] = T.ToTensor()(img)
        return sample


class RandomOcclusion:
    """
    Random rectangular occlusion for face-occlusion augmentation.

    Args:
        probability: Probability of applying occlusion.
        max_cover: Maximum fraction of image to occlude.
    """

    def __init__(self, probability: float = 0.1, max_cover: float = 0.15):
        self.probability = probability
        self.max_cover = max_cover

    def __call__(self, sample: Dict) -> Dict:
        if np.random.random() > self.probability:
            return sample

        image = sample["image"]
        C, H, W = image.shape

        occ_h = int(H * np.random.uniform(0.05, self.max_cover))
        occ_w = int(W * np.random.uniform(0.05, self.max_cover))
        x1 = np.random.randint(0, W - occ_w)
        y1 = np.random.randint(0, H - occ_h)

        image[:, y1:y1 + occ_h, x1:x1 + occ_w] = torch.rand(C, occ_h, occ_w) * 0.5
        sample["image"] = image
        return sample


class LandmarkAugmentation:
    """
    Composes all augmentations for landmark training.

    Args:
        rotation: Max rotation degrees (plus or minus).
        scale: (min, max) scale factor.
        brightness: Max brightness adjustment.
        contrast: Max contrast adjustment.
        occlusion_prob: Probability of random occlusion.
    """

    def __init__(
        self,
        rotation: float = 15.0,
        scale: Tuple[float, float] = (0.85, 1.15),
        brightness: float = 0.2,
        contrast: float = 0.2,
        occlusion_prob: float = 0.1,
    ):
        self.transforms = [
            RandomRotation(degrees=rotation),
            RandomBrightnessContrast(brightness=brightness, contrast=contrast),
            RandomOcclusion(probability=occlusion_prob),
        ]

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample
