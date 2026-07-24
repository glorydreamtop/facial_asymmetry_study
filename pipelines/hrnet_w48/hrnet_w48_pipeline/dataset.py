from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import normalize_image_tensor
from src.landmark.schema import NUM_POINTS, SYMMETRY_PAIRS

from .augment import HeatmapAugmentation


def build_gaussian_heatmaps(
    landmarks_norm: np.ndarray,
    mask: np.ndarray,
    heatmap_size: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    heat_h, heat_w = heatmap_size
    heatmaps = np.zeros((NUM_POINTS, heat_h, heat_w), dtype=np.float32)
    radius = max(1, int(3 * sigma))

    for idx in range(NUM_POINTS):
        if mask[idx] <= 0:
            continue

        cx = landmarks_norm[idx, 0] * (heat_w - 1)
        cy = landmarks_norm[idx, 1] * (heat_h - 1)
        mu_x = int(round(cx))
        mu_y = int(round(cy))

        x0 = max(0, mu_x - radius)
        x1 = min(heat_w - 1, mu_x + radius)
        y0 = max(0, mu_y - radius)
        y1 = min(heat_h - 1, mu_y + radius)

        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        heatmaps[idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(
            heatmaps[idx, y0:y1 + 1, x0:x1 + 1],
            gaussian,
        )

    return heatmaps


class HRNetHeatmapDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        img_root: str,
        input_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        sigma: float,
        augment: HeatmapAugmentation | None = None,
    ) -> None:
        self.img_root = Path(img_root)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {int(img["id"]): img for img in coco["images"]}
        self.annotations = coco["annotations"]

    @staticmethod
    def default_train_augment(config: dict) -> HeatmapAugmentation:
        return HeatmapAugmentation(
            rotation=float(config.get("rotation", 30.0)),
            translate=float(config.get("translate", 0.10)),
            scale=tuple(config.get("scale", [0.9, 1.1])),
            hflip_prob=float(config.get("hflip_prob", 0.5)),
            brightness=float(config.get("brightness", 0.15)),
            contrast=float(config.get("contrast", 0.15)),
            blur_prob=float(config.get("blur_prob", 0.15)),
            blur_kernel=int(config.get("blur_kernel", 5)),
            occlusion_prob=float(config.get("occlusion_prob", 0.05)),
            occlusion_max_cover=float(config.get("occlusion_max_cover", 0.12)),
            flip_pairs=SYMMETRY_PAIRS,
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        img_info = self.images[int(ann["image_id"])]

        img_path = self.img_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        landmarks = kp[:, :2].copy()
        mask = (kp[:, 2] > 0).astype(np.float32)

        input_h, input_w = self.input_size
        image = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        landmarks[:, 0] *= input_w / orig_w
        landmarks[:, 1] *= input_h / orig_h

        sample = {
            "image": image,
            "landmarks": landmarks,
            "mask": mask,
        }
        if self.augment is not None:
            sample = self.augment(sample)

        image = sample["image"]
        landmarks = sample["landmarks"].astype(np.float32)
        mask = sample["mask"].astype(np.float32)

        landmarks_norm = landmarks.copy()
        landmarks_norm[:, 0] /= input_w
        landmarks_norm[:, 1] /= input_h

        heatmaps = build_gaussian_heatmaps(
            landmarks_norm,
            mask,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = normalize_image_tensor(image_tensor)

        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(heatmaps).float(),
            "landmarks": torch.from_numpy(landmarks_norm).float(),
            "landmarks_px": torch.from_numpy(landmarks).float(),
            "mask": torch.from_numpy(mask).float(),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "image_id": int(ann["image_id"]),
            "file_name": img_info["file_name"],
        }
