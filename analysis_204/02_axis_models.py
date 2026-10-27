"""Stage 3 (final): four axis models on the 204-case analysis cohort.

Protocol
--------
cohort   : 204 cases with complete 11 bilateral landmark pairs
input    : expert landmarks + 167-dim coords_geom
target   : expert consensus axis (mean of A1/A2/B1/B2), x = m*y + b
split    : 143 / 41 / 20 (7:2:1, seed 11)
fitting  : train+val; test touched once
configs  : original protocol values, unchanged
           Ridge alpha=1000; Linear default; SVR C=10, eps=0.01; RF 500, leaf=2
selection: 5-fold CV on train+val (mean absolute angular deviation)

Outputs
-------
out/axis_final.json
out/axis_test_per_case.csv
out/axis_predictions_204.csv
out/models/ridge_expert.joblib  (+ other final models)
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import joblib
import numpy as np

import common as C
from axis_features import (
    LinearAxisRegressor,
    OrdinaryLinearAxisRegressor,
    RandomForestAxisRegressor,
    SVRAxisRegressor,
    build_axis_feature_vector,
    compute_sample_normalization,
    make_axis_feature_names,
    target_to_image_line,
)
from sklearn.model_selection import KFold

Y0, Y1 = 500.0, 2280.0
MODEL_FACTORIES = {
    "Ridge": lambda: LinearAxisRegressor(alpha=1000, solver="svd"),
    "Linear": lambda: OrdinaryLinearAxisRegressor(),
    "SVR-RBF": lambda: SVRAxisRegressor(C=10, epsilon=0.01, gamma="scale"),
    "Random Forest": lambda: RandomForestAxisRegressor(n_estimators=500, min_samples_leaf=2,
                                                       max_depth=None, random_state=42),
}


def stats(arr):
    arr = np.asarray(arr, dtype=float)
    m, s = C.mean_sd(arr)
    med, q1, q3 = C.median_iqr(arr)
    lo, hi = C.t_ci(arr)
    return {"mean": m, "sd": s, "median": med, "q1": q1, "q3": q3,
            "ci_low": lo, "ci_high": hi}


def signed_stats(arr):
    ba = C.bland_altman(arr, np.zeros_like(arr))
    return {"bias": ba["bias"], "sd": ba["sd"], "loa_low": ba["loa_low"], "loa_high": ba["loa_high"]}


def make_xy(cohort):
    X, T, centers, scales = [], [], [], []
    for c in cohort:
        lm = c["expert_lm"]
        center, scale = compute_sample_normalization(lm)
        X.append(build_axis_feature_vector(lm, center, scale))
        m, b = c["cons_m"], c["cons_b"]
        T.append([m, (b + m * center[1] - center[0]) / scale])
        centers.append(center)
        scales.append(scale)
    return (np.asarray(X, np.float32), np.asarray(T, np.float32),
            np.asarray(centers), np.asarray(scales))


def predict_lines(model, X, centers, scales):
    targets = model.predict_targets(X)
    out = []
    for i in range(len(targets)):
        m, b = target_to_image_line(targets[i], centers[i], float(scales[i]))
        out.append((float(m), float(b)))
    return out


def evaluate_lines(lines, cohort, indices):
    rows = []
    for line, k in zip(lines, indices):
        c = cohort[k]
        pm, pb = line
        gm, gb = c["cons_m"], c["cons_b"]
        yc = c["yc"]
        pa, ga = C.angle_deg(pm), C.angle_deg(gm)
        px, gx = pm * yc + pb, gm * yc + gb
        rows.append({
            "stem": c["stem"],
            "split": c["split"],
            "pred_m": pm, "pred_b": pb,
            "cons_m": gm, "cons_b": gb,
            "yc": yc,
            "angle_diff": abs(pa - ga),
            "position_diff": abs(px - gx),
            "mhd": C.mhd(pm, pb, gm, gb, Y0, Y1),
            "signed_angle": pa - ga,
            "signed_position": px - gx,
        })
    return rows


def summarize_rows(rows):
    return {
        "n": len(rows),
        "angle": stats([r["angle_diff"] for r in rows]),
        "position": stats([r["position_diff"] for r in rows]),
        "mhd": stats([r["mhd"] for r in rows]),
        "signed_angle": signed_stats([r["signed_angle"] for r in rows]),
        "signed_position": signed_stats([r["signed_position"] for r in rows]),
    }


def main():
    cohort = pickle.load(open(C.OUT / "cohort.pkl", "rb"))
    stems = [c["stem"] for c in cohort]
    split = json.load(open(C.DATA / "split_204.json", encoding="utf-8"))
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(cohort)

    X, T, centers, scales = make_xy(cohort)
    idx = {s: i for i, s in enumerate(stems)}
    tr = [idx[s] for s in split["train"]]
    va = [idx[s] for s in split["val"]]
    te = [idx[s] for s in split["test"]]
    tv = tr + va
    print(f"cohort={len(cohort)}  train={len(tr)} val={len(va)} test={len(te)}")

    # ---------------- 5-fold CV on train+val ----------------
    cv = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, factory in MODEL_FACTORIES.items():
        scores = []
        for trn, val in kf.split(tv):
            tri = [tv[i] for i in trn]
            vai = [tv[i] for i in val]
            model = factory()
            model.fit(X[tri], T[tri])
            lines = predict_lines(model, X[vai], centers[vai], scales[vai])
            rows = evaluate_lines(lines, cohort, vai)
            scores.append(float(np.mean([r["angle_diff"] for r in rows])))
        cv[name] = {"mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)),
                    "folds": [float(s) for s in scores]}
        print(f"  CV {name:14s} {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}")

    # ---------------- final fit + test ----------------
    model_dir = C.OUT / "models"
    model_dir.mkdir(exist_ok=True)
    results = {}
    per_case = []
    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(X[tv], T[tv])
        lines = predict_lines(model, X[te], centers[te], scales[te])
        rows = evaluate_lines(lines, cohort, te)
        for r in rows:
            r["model"] = name
        per_case.extend(rows)
        results[name] = {
            "cv": cv[name],
            "test": summarize_rows(rows),
        }
        if name == "Ridge":
            joblib.dump(model, model_dir / "ridge_expert.joblib")
        elif name == "SVR-RBF":
            joblib.dump(model, model_dir / "svr_expert.joblib")
        elif name == "Random Forest":
            joblib.dump(model, model_dir / "rf_expert.joblib")
        else:
            joblib.dump(model, model_dir / "linear_expert.joblib")
        a = results[name]["test"]["angle"]
        print(f"  TEST {name:14s} angle {a['mean']:.3f}±{a['sd']:.3f} "
              f"(med {a['median']:.3f}, IQR {a['q1']:.3f}-{a['q3']:.3f})")

    # Ridge predictions for all 204 (route B / downstream use)
    ridge = joblib.load(model_dir / "ridge_expert.joblib")
    all_lines = predict_lines(ridge, X, centers, scales)
    rows_all = []
    for c, line in zip(cohort, all_lines):
        rows_all.append({
            "stem": c["stem"], "split": c["split"],
            "expert_m": c["cons_m"], "expert_b": c["cons_b"],
            "ai_m": line[0], "ai_b": line[1],
            "angle_diff": abs(C.angle_deg(line[0]) - C.angle_deg(c["cons_m"])),
            "position_diff": abs((line[0] - c["cons_m"]) * c["yc"] + (line[1] - c["cons_b"])),
            "mhd": C.mhd(line[0], line[1], c["cons_m"], c["cons_b"], Y0, Y1),
        })

    payload = {
        "protocol": {
            "cohort": len(cohort),
            "excluded": 216 - len(cohort),
            "split": {k: len(v) for k, v in split.items()},
            "input": "expert landmarks, 167-dim coords_geom",
            "target": "expert consensus axis (mean of A1/A2/B1/B2)",
            "fit_on": "train+val",
            "test_n": len(te),
            "y_range_mhd": [Y0, Y1],
            "configs": {
                "Ridge": {"alpha": 1000},
                "Linear": {},
                "SVR-RBF": {"C": 10, "epsilon": 0.01, "gamma": "scale"},
                "Random Forest": {"n_estimators": 500, "min_samples_leaf": 2},
            },
            "selection": "5-fold CV on train+val, mean absolute angular deviation",
        },
        "models": results,
        "ridge_all": rows_all,
    }
    json.dump(payload, open(C.OUT / "axis_final.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    with open(C.OUT / "axis_test_per_case.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "stem", "split", "pred_m", "pred_b", "cons_m", "cons_b",
                                          "yc", "angle_diff", "position_diff", "mhd",
                                          "signed_angle", "signed_position"])
        w.writeheader()
        w.writerows(per_case)

    with open(C.OUT / "axis_predictions_204.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader()
        w.writerows(rows_all)

    feature_names = make_axis_feature_names()
    json.dump(feature_names, open(C.OUT / "feature_names_167.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("saved axis_final.json / axis_test_per_case.csv / axis_predictions_204.csv")
    print("Ridge test per-case saved; models in", model_dir)


if __name__ == "__main__":
    main()
