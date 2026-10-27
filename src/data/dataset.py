"""
PyTorch Dataset for facial landmark detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.landmark.schema import NUM_POINTS


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    return (image - IMAGENET_MEAN) / IMAGENET_STD


class FaceLandmarkDataset(Dataset):
    """
    PyTorch Dataset for facial landmark detection.

    Args:
        ann_file: Path to COCO-style annotation JSON.
        img_root: Root directory of images.
        transform: Optional callable for image/landmark augmentation.
        target_size: (H, W) to resize images to. If None, keep original size.
        normalize: Whether to normalize landmarks to [0, 1] range.
    """

    def __init__(
        self,
        ann_file: str,
        img_root: str = "",
        transform: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        normalize_image: bool = True,
    ):
        self.img_root = Path(img_root)
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.normalize_image = normalize_image

        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.annotations = self.coco["annotations"]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        ann = self.annotations[idx]
        img_info = self.images[ann["image_id"]]

        img_path = self.img_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        landmarks = kp[:, :2].copy()
        visible = kp[:, 2] > 0

        if self.target_size is not None:
            t_h, t_w = self.target_size
            image = cv2.resize(image, (t_w, t_h))
            scale_x = t_w / orig_w
            scale_y = t_h / orig_h
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y

        if self.normalize:
            h, w = image.shape[:2]
            norm_landmarks = landmarks.copy()
            norm_landmarks[:, 0] /= w
            norm_landmarks[:, 1] /= h
        else:
            norm_landmarks = landmarks

        mask = visible.astype(np.float32)

        sample = {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "landmarks": torch.from_numpy(norm_landmarks).float(),
            "mask": torch.from_numpy(mask).float(),
            "orig_landmarks": torch.from_numpy(landmarks).float(),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "image_id": ann["image_id"],
        }

        if self.transform:
            sample = self.transform(sample)

        if self.normalize_image:
            sample["image"] = normalize_image_tensor(sample["image"])

        return sample


class FaceCropDataset(Dataset):
    """
    Dataset that first detects and crops the face, then returns the crop.

    Args:
        ann_file: Path to COCO-style annotation JSON.
        img_root: Root directory of images.
        detector: Face detector instance with .detect(image) to list of bboxes.
        crop_size: (H, W) to resize face crops to.
        transform: Optional augmentation.
    """

    def __init__(
        self,
        ann_file: str,
        img_root: str = "",
        detector=None,
        crop_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        normalize_image: bool = True,
    ):
        self.img_root = Path(img_root)
        self.detector = detector
        self.crop_size = crop_size
        self.transform = transform
        self.normalize_image = normalize_image

        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.annotations = self.coco["annotations"]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        ann = self.annotations[idx]
        img_info = self.images[ann["image_id"]]

        img_path = self.img_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        if self.detector is not None:
            faces = self.detector.detect(image)
            if faces:
                bbox = faces[0]
            else:
                bbox = (0, 0, orig_w, orig_h)
        else:
            bbox = (0, 0, orig_w, orig_h)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        margin = 0.2
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - margin * w))
        y1 = max(0, int(y1 - margin * h))
        x2 = min(orig_w, int(x2 + margin * w))
        y2 = min(orig_h, int(y2 + margin * h))

        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.crop_size[1], self.crop_size[0]))

        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        landmarks = kp[:, :2].copy()
        visible = kp[:, 2] > 0

        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1
        scale_x = self.crop_size[1] / (x2 - x1)
        scale_y = self.crop_size[0] / (y2 - y1)
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y

        h_c, w_c = crop.shape[:2]
        norm_landmarks = landmarks.copy()
        norm_landmarks[:, 0] /= w_c
        norm_landmarks[:, 1] /= h_c

        mask = visible.astype(np.float32)

        sample = {
            "image": torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0,
            "landmarks": torch.from_numpy(norm_landmarks).float(),
            "mask": torch.from_numpy(mask).float(),
            "orig_landmarks": torch.from_numpy(landmarks).float(),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "image_id": ann["image_id"],
        }

        if self.transform:
            sample = self.transform(sample)

        if self.normalize_image:
            sample["image"] = normalize_image_tensor(sample["image"])

        return sample
