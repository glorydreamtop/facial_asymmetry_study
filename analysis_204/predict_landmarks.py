"""Run HRNet-W48 on a directory of images and cache 29-point landmarks.

Example
-------
python analysis_204/predict_landmarks.py \
  --image_dir 对称轴数据/1.对称轴133张 \
  --checkpoint checkpoints/hrnet_w48_29pts_coord010_best.pth \
  --config pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml \
  --output analysis_204/data/automatic_landmarks_204.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipelines.hrnet_w48.hrnet_w48_pipeline.model import build_model
from pipelines.hrnet_w48.hrnet_w48_pipeline.metrics import decode_heatmaps_local_softargmax
from src.data.dataset import normalize_image_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml")
    parser.add_argument("--output", default="analysis_204/data/automatic_landmarks_204.npz")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg["model"]["pretrained"] = False
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available()
                          else ("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device))

    model = build_model(cfg["model"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.to(device).eval()
    input_size = tuple(cfg["model"]["input_size"])

    image_dir = Path(args.image_dir)
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    names, landmarks, confs = [], [], []
    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        resized = cv2.resize(rgb, (input_size[1], input_size[0]))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = normalize_image_tensor(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            heatmaps = model(tensor)
            coords_norm, scores = decode_heatmaps_local_softargmax(heatmaps)
        lm = coords_norm[0].cpu().numpy().copy()
        lm[:, 0] *= w
        lm[:, 1] *= h
        names.append(path.stem)
        landmarks.append(lm.astype(np.float32))
        confs.append(scores[0].cpu().numpy().astype(np.float32))
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names, dtype=object),
             landmarks=np.stack(landmarks), conf=np.stack(confs))
    print(f"saved {len(names)} cases to {out}")


if __name__ == "__main__":
    main()
