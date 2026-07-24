"""
Face detection module using RetinaFace (insightface backend).

Supports:
  - Mac M4 (CoreML via ONNX + insightface)
  - Linux NVIDIA GPU (CUDA)
  - CPU fallback
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


class RetinaFaceDetector:
    """
    RetinaFace face detector wrapper.

    Uses insightface's RetinaFace model (buffalo_l pack) which provides:
      - Face bounding boxes
      - 5-point landmarks (eyes, nose, mouth corners)
      - Confidence scores

    Args:
        device: "auto", "cpu", "cuda", or "mps".
        conf_thresh: Minimum confidence threshold.
    """

    def __init__(
        self,
        device: str = "auto",
        conf_thresh: float = 0.8,
    ):
        self.conf_thresh = conf_thresh

        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name="buffalo_l",
                root="~/.insightface",
                providers=self._get_providers(device),
            )
            self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_thresh=conf_thresh)
            print(f"[RetinaFaceDetector] Initialized on {device}")
        except ImportError:
            raise ImportError(
                "insightface not installed. Run: pip install insightface"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RetinaFace: {e}")

    @staticmethod
    def _get_providers(device: str) -> List[str]:
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image.

        Args:
            image: RGB or BGR image array (H, W, 3).

        Returns:
            List of dicts with keys:
              - bbox: (x1, y1, x2, y2) float
              - confidence: float
              - landmarks: (5, 2) array [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        if image.shape[2] == 3 and image.dtype == np.uint8:
            pass
        elif image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        faces = self.app.get(image)

        results = []
        for face in faces:
            bbox = face.bbox.astype(float).tolist()
            confidence = float(face.det_score)
            landmarks = face.landmark.astype(float) if face.landmark is not None else None

            results.append({
                "bbox": bbox,
                "confidence": confidence,
                "landmarks_5": landmarks,
            })

        return results

    def detect_single(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect the highest-confidence face.
        Returns None if no face found.
        """
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f["confidence"])

    def crop_face(
        self, image: np.ndarray, margin: float = 0.2
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        """
        Detect and crop the largest face.

        Args:
            image: Input image.
            margin: Extra margin ratio around face bbox.

        Returns:
            (crop, original_landmarks, face_info) or (None, None, None) if no face.
        """
        face = self.detect_single(image)
        if face is None:
            return None, None, None

        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
        fw, fh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - margin * fw))
        y1 = max(0, int(y1 - margin * fh))
        x2 = min(w, int(x2 + margin * fw))
        y2 = min(h, int(y2 + margin * fh))

        crop = image[y1:y2, x1:x2]

        # Adjust landmarks to crop coordinates
        adjusted_lms = None
        if face["landmarks_5"] is not None:
            adjusted_lms = face["landmarks_5"].copy()
            adjusted_lms[:, 0] -= x1
            adjusted_lms[:, 1] -= y1

        face["landmarks_5_cropped"] = adjusted_lms
        face["crop_offset"] = (x1, y1)

        return crop, adjusted_lms, face
