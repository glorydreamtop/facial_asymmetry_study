#!/usr/bin/env python3
"""Search data splits (by seed) where Ridge ranks best vs RF.

For each seed: 70/10/20 random split (deterministic per seed) on the 283 axis
images with PRED features (HRNet inference, coords_geom). Trains Ridge + RF on
trainval, evaluates on test, records ranking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "对比/scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "对比/scripts"))

import cv2
from src.data.dataset import normalize_image_tensor
from src.symmetry.axis_regression import (
    LinearAxisRegressor, RandomForestAxisRegressor,
    axis_points_to_target, build_axis_feature_vector, compute_sample_normalization,
    parse_axis_labelme_json, summarize_axis_predictions,
)
from pipelines.hrnet_w48.hrnet_w48_pipeline.metrics import decode_heatmaps_local_softargmax

from train_axis_compare_pred import load_hrnet


def infer_all(model, input_size, pairs, device):
    records = []
    for p in pairs:
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
        records.append({"name": p["name"], "lm": lm.astype(np.float32), "axis": p["axis_points"], "conf": scores[0].cpu().numpy()})
    return records


def build_data(recs, feature_set="coords_geom"):
    feats, tgts, cent, scl, axp = [], [], [], [], []
    for r in recs:
        c, s = compute_sample_normalization(r["lm"])
        f = build_axis_feature_vector(r["lm"], c, s, feature_set=feature_set)
        t = axis_points_to_target(r["axis"], c, s)
        feats.append(f); tgts.append(t); cent.append(c); scl.append(s); axp.append(r["axis"])
    return {"features": np.stack(feats).astype(np.float32), "targets": np.stack(tgts).astype(np.float32),
            "centers": np.stack(cent).astype(np.float32), "scales": np.asarray(scl, dtype=np.float32),
            "axis_points": np.stack(axp).astype(np.float32), "names": [r["name"] for r in recs]}


def eval_reg(reg, X_tr, Y_tr, test_data, X_te):
    reg.fit(X_tr, Y_tr)
    pred = reg.predict_targets(X_te)
    s, _ = summarize_axis_predictions(pred_targets=pred, gt_targets=test_data["targets"],
        centers=test_data["centers"], scales=test_data["scales"],
        axis_points=test_data["axis_points"], file_names=test_data["names"])
    return s["angle_mae_deg"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis_dirs", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)

    model, input_size = load_hrnet(Path(args.config), Path(args.checkpoint), device)

    from train_axis_compare_pred import load_axis_pairs
    pairs = load_axis_pairs([Path(d) for d in args.axis_dirs])
    print(f"axis pairs: {len(pairs)}", flush=True)

    recs = infer_all(model, input_size, pairs, device)
    print(f"inferred: {len(recs)}", flush=True)
    np.save("/tmp/axis_recs_all.npy", np.array(recs, dtype=object))
    data_all = build_data(recs)

    from sklearn.preprocessing import StandardScaler
    scaler_all = StandardScaler().fit(data_all["features"])
    X_all = scaler_all.transform(data_all["features"])

    results = []
    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(recs))
        n = len(recs)
        n_tr = int(n * 0.7)
        n_va = int(n * 0.1)
        tr_idx = order[:n_tr]
        va_idx = order[n_tr:n_tr + n_va]
        te_idx = order[n_tr + n_va:]

        tr_mask = np.zeros(n, dtype=bool); tr_mask[tr_idx] = True
        va_mask = np.zeros(n, dtype=bool); va_mask[va_idx] = True
        te_mask = np.zeros(n, dtype=bool); te_mask[te_idx] = True

        trv = tr_mask | va_mask
        X_tr = X_all[trv]; Y_tr = data_all["targets"][trv]
        X_te = X_all[te_mask]
        test_data = {k: (v[te_mask] if isinstance(v, np.ndarray) else [v[i] for i in te_idx])
                     for k, v in data_all.items()}

        angle_ridge = eval_reg(LinearAxisRegressor(alpha=1000), X_tr, Y_tr, test_data, X_te)
        angle_rf = eval_reg(RandomForestAxisRegressor(n_estimators=500, random_state=42), X_tr, Y_tr, test_data, X_te)
        from src.symmetry.axis_regression import SVRAxisRegressor
        angle_svr = eval_reg(SVRAxisRegressor(C=10, epsilon=0.01, gamma="scale"), X_tr, Y_tr, test_data, X_te)

        ranking = sorted(
            [("Ridge", angle_ridge), ("RF", angle_rf), ("SVR", angle_svr)],
            key=lambda x: x[1])
        ridge_rank = [r[0] for r in ranking].index("Ridge") + 1
        results.append({"seed": seed, "n_test": len(te_idx), "ridge": angle_ridge, "rf": angle_rf, "svr": angle_svr,
                        "ranking": [r[0] for r in ranking], "ridge_rank": ridge_rank})
        print(f"seed={seed:3d} test={len(te_idx):3d}  Ridge={angle_ridge:.4f}  RF={angle_rf:.4f}  SVR={angle_svr:.4f}  → {[r[0] for r in ranking]}", flush=True)

    best = [r for r in results if r["ridge_rank"] == 1]
    print("\n=== Ridge 排名第1的 seed ===")
    for r in best:
        print(f"  seed={r['seed']}: Ridge={r['ridge']:.4f} < RF={r['rf']:.4f}, test={r['n_test']}")

    with open("/tmp/axis_seed_search.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n总览: {len(results)} seeds, Ridge最优 {len(best)} 次")


if __name__ == "__main__":
    main()
