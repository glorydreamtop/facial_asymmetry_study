from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HeatmapAugmentation:
    rotation: float = 30.0
    translate: float = 0.10
    scale: tuple[float, float] = (0.9, 1.1)
    hflip_prob: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    blur_prob: float = 0.15
    blur_kernel: int = 5
    occlusion_prob: float = 0.05
    occlusion_max_cover: float = 0.12
    flip_pairs: list[tuple[int, int]] | None = None

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        landmarks = sample["landmarks"].copy()
        mask = sample["mask"].copy()

        image, landmarks, mask = self._random_horizontal_flip(image, landmarks, mask)
        image, landmarks, mask = self._random_affine(image, landmarks, mask)
        image = self._random_brightness_contrast(image)
        image = self._random_blur(image)
        image = self._random_occlusion(image)

        sample["image"] = image
        sample["landmarks"] = landmarks
        sample["mask"] = mask
        return sample

    def _random_horizontal_flip(self, image, landmarks, mask):
        if np.random.random() >= self.hflip_prob:
            return image, landmarks, mask

        h, w = image.shape[:2]
        image = np.ascontiguousarray(image[:, ::-1])
        landmarks[:, 0] = (w - 1) - landmarks[:, 0]

        for left, right in self.flip_pairs or []:
            landmarks[[left, right]] = landmarks[[right, left]]
            mask[[left, right]] = mask[[right, left]]

        return image, landmarks, mask

    def _random_affine(self, image, landmarks, mask):
        h, w = image.shape[:2]
        angle = np.random.uniform(-self.rotation, self.rotation)
        scale = np.random.uniform(self.scale[0], self.scale[1])
        tx = np.random.uniform(-self.translate, self.translate) * w
        ty = np.random.uniform(-self.translate, self.translate) * h

        center = (w * 0.5, h * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([landmarks.astype(np.float32), ones], axis=1)
        transformed = (matrix @ points_h.T).T

        inside = (
            (transformed[:, 0] >= 0)
            & (transformed[:, 0] <= (w - 1))
            & (transformed[:, 1] >= 0)
            & (transformed[:, 1] <= (h - 1))
        )
        mask = mask * inside.astype(np.float32)
        transformed[:, 0] = np.clip(transformed[:, 0], 0, w - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, h - 1)
        return image, transformed.astype(np.float32), mask.astype(np.float32)

    def _random_brightness_contrast(self, image):
        alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        beta = 255.0 * np.random.uniform(-self.brightness, self.brightness)
        image = image.astype(np.float32) * alpha + beta
        return np.clip(image, 0, 255).astype(np.uint8)

    def _random_blur(self, image):
        if np.random.random() >= self.blur_prob:
            return image
        kernel = max(3, int(self.blur_kernel))
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0)

    def _random_occlusion(self, image):
        if np.random.random() >= self.occlusion_prob:
            return image

        h, w = image.shape[:2]
        occ_h = max(4, int(h * np.random.uniform(0.05, self.occlusion_max_cover)))
        occ_w = max(4, int(w * np.random.uniform(0.05, self.occlusion_max_cover)))
        x1 = np.random.randint(0, max(1, w - occ_w))
        y1 = np.random.randint(0, max(1, h - occ_h))
        fill = np.random.randint(0, 128, size=(occ_h, occ_w, 3), dtype=np.uint8)
        image = image.copy()
        image[y1:y1 + occ_h, x1:x1 + occ_w] = fill
        return image
