#!/usr/bin/env python3
"""Cache HRNet-predicted landmarks for all 283 axis images (pred route).

Output: 对比/results/axis_final/pred_landmarks_283.npz
  - names: list[str]
  - landmarks: (283, 29, 2) float32  (pixel coords, original image size)
  - conf: (283, 29) float32
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "对比/scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "对比/scripts"))

from train_axis_compare_pred import load_axis_pairs, load_hrnet
from pipelines.hrnet_w48.hrnet_w48_pipeline.metrics import decode_heatmaps_local_softargmax
from src.data.dataset import normalize_image_tensor


def main() -> None:
    device = torch.device("mps")
    model, input_size = load_hrnet(Path("对比/configs/hrnet_w48.yaml"),
                                   Path("checkpoints/hrnet_w48_29pts_coord010_best.pth"), device)
    pairs = load_axis_pairs([Path("../对称轴数据/1.对称轴133张"), Path("../对称轴数据/2. 对称轴157张")])
    print(f"total pairs: {len(pairs)}", flush=True)

    names = []
    landmarks = []
    confs = []
    for i, p in enumerate(pairs):
        img = cv2.imread(str(p["image_path"]))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        resized = cv2.resize(rgb, (input_size[1], input_size[0]))
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        t = normalize_image_tensor(t).unsqueeze(0).to(device)
        with torch.no_grad():
            hm = model(t)
            coords_norm, scores = decode_heatmaps_local_softargmax(hm)
        lm = coords_norm[0].cpu().numpy().copy()
        lm[:, 0] *= w
        lm[:, 1] *= h
        names.append(p["name"])
        landmarks.append(lm.astype(np.float32))
        confs.append(scores[0].cpu().numpy().astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    out_path = Path("对比/results/axis_final/pred_landmarks_283.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, names=np.array(names, dtype=object),
             landmarks=np.stack(landmarks), conf=np.stack(confs))
    print(f"Saved {len(names)} landmarks to {out_path}", flush=True)


if __name__ == "__main__":
    main()
