#!/usr/bin/env python3
"""
283-subject pipeline (PRED route — pure-AI self-validation).

Replaces the 100-case expert-landmark pipeline. All four parts run on the
283 axis images with HRNet-predicted landmarks + expert axes + 283 VAS:

  Part 1 - Expert axis vs AI axis (AI axis from HRNet-predicted landmarks)
  Part 2 - 5-region RAS + global FA (from HRNet-predicted landmarks)
  Part 3 - Regression + SHAP weight training (283 VAS)
  Part 4 - 7:2:1 evaluation
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import shap
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, t as t_dist
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.landmark.schema import NAME_TO_IDX, NUM_POINTS, SYMMETRY_PAIRS

# ============================================================
# Constants & output paths
# ============================================================
OUT_BASE = REPO_ROOT / "results"
OUT_AXIS = OUT_BASE / "part1_axis_consensus"
OUT_REGION = OUT_BASE / "part2_region_fa"
OUT_PIPELINE = OUT_BASE / "part3_weights_shap"
OUT_EVAL = OUT_BASE / "part4_evaluation"

REGION_KEYS = ["VAS-eyebrow", "VAS-eye", "VAS-nose", "VAS-mouth", "VAS-facial contour"]
REGION_PAIRS = {
    "VAS-eyebrow": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "VAS-eye": [(8, 9), (10, 11), (12, 13)],
    "VAS-nose": [(17, 18)],
    "VAS-mouth": [(19, 20), (21, 22)],
    "VAS-facial contour": [(26, 27)],
}
_NAME_TO_IDX_CI = {k.lower(): v for k, v in NAME_TO_IDX.items()}
VALID_TAGS = {"ebe","ebs","os","sci","en","p","P","ex","al","cph","ch","Go","Na","prn","Sn","ls","sto","li","Me"}
EPS = 1e-9


# ============================================================
# Geometry helpers (same as 06_pipeline)
# ============================================================
def reflect_across_line(pts: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    d = (pts[:, 0] - slope * pts[:, 1] - intercept) / (1 + slope * slope)
    reflected = np.empty_like(pts)
    reflected[:, 0] = pts[:, 0] - 2 * d
    reflected[:, 1] = pts[:, 1] + 2 * slope * d
    return reflected


def symmetry_cost(params: tuple[float, float], pts: np.ndarray) -> float:
    slope, intercept = params
    reflected = reflect_across_line(pts, slope, intercept)
    total = 0.0
    count = 0
    for li, ri in SYMMETRY_PAIRS:
        if np.any(np.isnan(pts[li])) or np.any(np.isnan(pts[ri])):
            continue
        total += float(np.linalg.norm(reflected[li] - pts[ri]) ** 2)
        count += 1
    return total / count if count else 0.0


def compute_optimized_axis(landmarks: np.ndarray) -> tuple[float, float]:
    valid = landmarks[~np.isnan(landmarks[:, 0])]
    center = valid.mean(axis=0)
    res = minimize(symmetry_cost, [0.0, center[0]], args=(landmarks,),
                   method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8})
    return float(res.x[0]), float(res.x[1])


def axis_to_angle(m: float) -> float:
    return float(np.degrees(np.arctan(m)))


def axis_position(m: float, b: float, yc: float) -> float:
    return float(m * yc + b)


def mean_horizontal_distance(m1, b1, pts1, m2, b2, pts2, n_samples: int = 100) -> float:
    y_vals = np.linspace(float(pts1[:, 1].min()), float(pts1[:, 1].max()), n_samples)
    x1 = m1 * y_vals + b1
    x2 = m2 * y_vals + b2
    return float(np.mean(np.abs(x1 - x2)))


def compute_ipd(landmarks: np.ndarray) -> float:
    return float(np.linalg.norm(landmarks[NAME_TO_IDX["p_L"]] - landmarks[NAME_TO_IDX["p_R"]]))


def compute_region_ras(landmarks, slope, intercept, region_pairs, ipd):
    sq = []
    for li, ri in region_pairs:
        if np.any(np.isnan(landmarks[li])) or np.any(np.isnan(landmarks[ri])):
            continue
        x0, y0 = landmarks[li]
        d = (x0 - slope * y0 - intercept) / (1 + slope * slope)
        ref = np.array([x0 - 2 * d, y0 + 2 * slope * d])
        sq.append(float(np.linalg.norm(landmarks[ri] - ref) ** 2))
    if not sq or ipd <= 0:
        return 0.0
    return float(np.sqrt(np.mean(sq))) / ipd


def load_axis_from_json(path: Path) -> tuple[float, float, np.ndarray] | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for shape in data.get("shapes", []):
        if shape.get("label") != "axis":
            continue
        pts = np.asarray(shape.get("points", []), dtype=np.float64)
        if pts.shape != (2, 2):
            continue
        order = np.lexsort((pts[:, 0], pts[:, 1]))
        pts = pts[order]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        if abs(y2 - y1) < 1e-8:
            return None
        m = (x2 - x1) / (y2 - y1)
        b = x1 - m * y1
        return float(m), float(b), pts
    return None


# ============================================================
# Data loading (283, pred route)
# ============================================================
def load_283_entries() -> list[dict]:
    cache = np.load(REPO_ROOT / "对比/results/axis_final/pred_landmarks_283.npz", allow_pickle=True)
    names = list(cache["names"])
    landmarks = cache["landmarks"]
    conf = cache["conf"]

    axis_dirs = [
        REPO_ROOT / "../对称轴数据/1.对称轴133张",
        REPO_ROOT / "../对称轴数据/2. 对称轴157张",
    ]
    axis_by_name = {}
    for d in axis_dirs:
        for jp in sorted(d.glob("*.json")):
            ax = load_axis_from_json(jp)
            if ax is not None:
                axis_by_name[f"{d.name}/{jp.stem.strip()}.JPG"] = ax

    # VAS 283
    vas_map = {}
    vas_path = REPO_ROOT / "对比/results/axis_final/专家VAS评分_283例.xlsx"
    wb = openpyxl.load_workbook(vas_path, data_only=True)
    ws = wb.active
    for r in range(2, ws.max_row + 1):
        name = ws.cell(r, 2).value
        avg = ws.cell(r, 3).value
        if name and avg is not None:
            vas_map[str(name).strip()] = float(avg)

    entries = []
    for i, name in enumerate(names):
        ax = axis_by_name.get(name)
        if ax is None:
            continue
        entries.append({
            "name": name,
            "landmarks": landmarks[i],
            "conf": conf[i],
            "axis": ax,
            "vas": vas_map.get(name),
        })
    print(f"Loaded {len(entries)} subjects (pred landmarks + expert axis + VAS)")
    return entries


# ============================================================
# Part 1: Expert vs AI axis
# ============================================================
def build_part1(entries: list[dict]) -> list[dict]:
    results = []
    for ent in entries:
        name = ent["name"]
        em, eb, epts = ent["axis"]
        e_angle = axis_to_angle(em)
        eyc = (epts[0, 1] + epts[1, 1]) / 2.0
        e_pos = axis_position(em, eb, eyc)

        am, ab = compute_optimized_axis(ent["landmarks"])
        a_angle = axis_to_angle(am)
        a_pos = axis_position(am, ab, eyc)
        angle_err = abs(e_angle - a_angle)
        horiz = mean_horizontal_distance(em, eb, epts, am, ab, epts)

        results.append({
            "Name": name, "Expert_m": em, "Expert_b": eb,
            "Expert_Angle": round(e_angle, 4), "Expert_Position": round(e_pos, 2),
            "AI_m": am, "AI_b": ab,
            "AI_Angle": round(a_angle, 4), "AI_Position": round(a_pos, 2),
            "Angle_Error": round(angle_err, 4),
            "Mean_Horizontal_Distance": round(horiz, 2),
        })
    return results


def save_part1(axis_data: list[dict]):
    OUT_AXIS.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUT_AXIS / "专家共识与AI对称轴数据_283例.xlsx"
    json_path = OUT_AXIS / "axis_consensus_details_283.json"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "axis_data"
    headers = ["Name", "Expert_m", "Expert_b", "Expert_Angle", "Expert_Position",
               "AI_m", "AI_b", "AI_Angle", "AI_Position", "Angle_Error", "Mean_Horizontal_Distance"]
    for c, h in enumerate(headers, 1):
        ws.cell(1, c, h)
    for r, row in enumerate(axis_data, 2):
        for c, h in enumerate(headers, 1):
            ws.cell(r, c, row[h])
    wb.save(xlsx_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(axis_data, f, indent=2, ensure_ascii=False)
    print(f"Part 1: {len(axis_data)} subjects -> {xlsx_path}")


# ============================================================
# Part 2: 5-region RAS + FA
# ============================================================
def build_part2(entries: list[dict]) -> list[dict]:
    results = []
    for ent in entries:
        lm = ent["landmarks"]
        slope, intercept = compute_optimized_axis(lm)
        ipd = compute_ipd(lm)
        if ipd <= 0:
            continue
        row = {"Name": ent["name"]}
        for rk in REGION_KEYS:
            row[rk] = round(compute_region_ras(lm, slope, intercept, REGION_PAIRS[rk], ipd), 6)
        results.append(row)
    return results


def save_part2(region_data: list[dict]):
    OUT_REGION.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUT_REGION / "AI各区域不对称评分+GlobalFA原始值_283例.xlsx"
    json_path = OUT_REGION / "part2_region_fa_details_283.json"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "region_ras"
    headers = ["Name"] + REGION_KEYS
    for c, h in enumerate(headers, 1):
        ws.cell(1, c, h)
    for r, row in enumerate(region_data, 2):
        for c, h in enumerate(headers, 1):
            ws.cell(r, c, row[h])
    wb.save(xlsx_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(region_data, f, indent=2, ensure_ascii=False)
    print(f"Part 2: {len(region_data)} subjects -> {xlsx_path}")


# ============================================================
# Part 3: Regression + SHAP weight training
# ============================================================
def calibrate_to_vas(raw_scores: np.ndarray, target_vas: np.ndarray) -> np.ndarray:
    lr = LinearRegression()
    lr.fit(raw_scores.reshape(-1, 1), target_vas)
    return lr.predict(raw_scores.reshape(-1, 1))


def run_part3(region_data: list[dict], vas_map: dict[str, float]) -> dict:
    matched = []
    for row in region_data:
        if row["Name"] in vas_map:
            matched.append(row)
    print(f"Part 3: {len(matched)} matched subjects")
    if not matched:
        raise RuntimeError("No matched subjects")

    feature_names = [rk for rk in REGION_KEYS]
    X = np.array([[row[rk] for rk in feature_names] for row in matched], dtype=np.float64)
    y = np.array([vas_map[row["Name"]] for row in matched], dtype=np.float64)

    # z-score features
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < EPS] = 1.0
    Xz = (X - mu) / sd

    # Linear regression (standardized betas -> normalized weights)
    lr = LinearRegression()
    lr.fit(Xz, y)
    beta_std = np.abs(lr.coef_)
    beta_std_pos = np.clip(lr.coef_, 0, None)
    if beta_std_pos.sum() > 0:
        weights = beta_std_pos / beta_std_pos.sum()
    else:
        weights = np.ones_like(beta_std_pos) / len(beta_std_pos)

    # Stats for weight table
    n = len(y)
    dof = n - len(feature_names) - 1
    y_pred_lr = lr.predict(Xz)
    rss = float(np.sum((y - y_pred_lr) ** 2))
    mse = rss / max(dof, 1)
    se = np.sqrt(np.diag(np.linalg.pinv(Xz.T @ Xz)) * mse) if Xz.shape[0] > Xz.shape[1] else np.ones_like(beta_std)
    t_vals = lr.coef_ / se
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), dof))

    # VIF
    vifs = []
    for i in range(Xz.shape[1]):
        others = np.delete(Xz, i, axis=1)
        reg = LinearRegression().fit(others, Xz[:, i])
        r2i = reg.score(others, Xz[:, i])
        vifs.append(float(1.0 / (1.0 - r2i)) if r2i < 1.0 else float("inf"))

    # RandomForest + SHAP
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(Xz, y)
    explainer = shap.TreeExplainer(rf)
    shap_values = np.asarray(explainer.shap_values(Xz), dtype=np.float64)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_weights = mean_abs_shap / np.sum(mean_abs_shap)

    # Global FA score
    fa_raw = Xz @ weights
    fa_calibrated = calibrate_to_vas(fa_raw, y)

    region_weights = []
    for i, rk in enumerate(feature_names):
        ci_lo, ci_hi = t_dist.interval(0.95, dof)
        beta_ci = (ci_hi - ci_lo) / 2 * se[i]
        region_weights.append({
            "region_key": rk,
            "region": rk.replace("VAS-", "").replace("facial contour", "Facial Contour").capitalize(),
            "raw_coefficient": float(lr.coef_[i]),
            "raw_coefficient_95ci": [float(lr.coef_[i] - beta_ci), float(lr.coef_[i] + beta_ci)],
            "standardized_beta": float(lr.coef_[i]),
            "standardized_beta_95ci": [float(lr.coef_[i] - beta_ci), float(lr.coef_[i] + beta_ci)],
            "positive_standardized_beta": float(beta_std_pos[i]),
            "normalized_weight": float(weights[i]),
            "vif": vifs[i],
            "p_value": float(p_vals[i]),
        })

    result = {
        "note": "283-subject pipeline (pred route): HRNet-predicted landmarks + expert axis.",
        "regression": {
            "intercept_beta_0": float(lr.intercept_),
            "r_squared": float(r2_score(y, y_pred_lr)),
            "adjusted_r_squared": float(1 - (1 - r2_score(y, y_pred_lr)) * (n - 1) / max(dof, 1)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred_lr))),
            "mae": float(mean_absolute_error(y, y_pred_lr)),
            "n_samples": n,
        },
        "region_weights": region_weights,
        "shap": {
            "mean_abs_shap": {feature_names[i]: float(mean_abs_shap[i]) for i in range(len(feature_names))},
            "normalized_weight": {feature_names[i]: float(shap_weights[i]) for i in range(len(feature_names))},
        },
        "per_sample": [
            {"Name": row["Name"], "VAS-global": float(vas_map[row["Name"]]),
             "FA-global-raw": float(fa_raw[i]), "FA-global-calibrated": float(fa_calibrated[i])}
            for i, row in enumerate(matched)
        ],
    }

    OUT_PIPELINE.mkdir(parents=True, exist_ok=True)
    with open(OUT_PIPELINE / "weight_results_283.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # SHAP figures
    plt.figure()
    shap.summary_plot(shap_values, Xz, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(OUT_PIPELINE / "shap_summary_283.png", dpi=200, bbox_inches="tight")
    plt.close()
    plt.figure()
    shap.summary_plot(shap_values, Xz, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_PIPELINE / "shap_bar_283.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Region weight figure
    reg_w = [rw["normalized_weight"] for rw in region_weights]
    shap_w_list = [shap_weights[i] for i in range(len(feature_names))]
    x = np.arange(len(feature_names))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, reg_w, w, label="Regression", color="#4C72B0")
    plt.bar(x + w/2, shap_w_list, w, label="SHAP", color="#DD8452")
    plt.xticks(x, [rk.replace("VAS-", "") for rk in feature_names], rotation=15)
    plt.ylabel("Normalized Weight")
    plt.title("Region Weight Comparison (283-subject pred route)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PIPELINE / "region_weights_283.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Part 3: -> {OUT_PIPELINE / 'weight_results_283.json'}")
    return result


# ============================================================
# Part 4: Evaluation
# ============================================================
def bland_altman_stats(machine: np.ndarray, expert: np.ndarray) -> dict:
    diff = machine - expert
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return {"bias": bias, "sd_diff": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def evaluate_subset(rows: list[dict]) -> dict:
    y_true = np.array([r["VAS-global"] for r in rows], dtype=np.float64)
    y_pred = np.array([r["FA-global-calibrated"] for r in rows], dtype=np.float64)
    pearson_r, pearson_p = pearsonr(y_pred, y_true)
    spearman_r, spearman_p = spearmanr(y_pred, y_true)
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    ba = bland_altman_stats(y_pred, y_true)
    return {
        "n": len(rows), "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "r_squared": r2, "rmse": rmse, "mae": mae, "bland_altman": ba,
    }


def make_splits(rows: list[dict]) -> dict[str, list[dict]]:
    rng = np.random.RandomState(42)
    shuffled = sorted(rows, key=lambda x: x["Name"])
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def save_part4(pipeline_result: dict):
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    rows = pipeline_result["per_sample"]
    splits = make_splits(rows)
    metrics = {k: evaluate_subset(v) for k, v in splits.items()}
    metrics["all"] = evaluate_subset(rows)

    y_true = np.array([r["VAS-global"] for r in rows], dtype=np.float64)
    y_pred = np.array([r["FA-global-calibrated"] for r in rows], dtype=np.float64)

    # scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Expert VAS (均分)")
    plt.ylabel("Machine FA Score (calibrated)")
    plt.title("Machine Score vs Expert VAS (283-subject pred route)")
    plt.tight_layout()
    plt.savefig(OUT_EVAL / "scatter_283.png", dpi=200, bbox_inches="tight")
    plt.close()

    # bland-altman
    mean_vals = (y_true + y_pred) / 2.0
    diff_vals = y_pred - y_true
    ba = bland_altman_stats(y_pred, y_true)
    plt.figure(figsize=(7, 5))
    plt.scatter(mean_vals, diff_vals, alpha=0.8)
    plt.axhline(ba["bias"], linestyle="-")
    plt.axhline(ba["loa_low"], linestyle="--")
    plt.axhline(ba["loa_high"], linestyle="--")
    plt.xlabel("Mean of Machine FA and Expert VAS")
    plt.ylabel("Difference (Machine - Expert)")
    plt.title("Bland-Altman Plot (283-subject pred route)")
    plt.tight_layout()
    plt.savefig(OUT_EVAL / "bland_altman_283.png", dpi=200, bbox_inches="tight")
    plt.close()

    # split results xlsx
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "split_results"
    headers = ["subset", "Name"] + REGION_KEYS + ["VAS-global", "FA-global-calibrated", "diff"]
    for c, h in enumerate(headers, 1):
        ws.cell(1, c, h)
    row_idx = 2
    for subset, srows in splits.items():
        for srow in srows:
            reg = next((r for r in []), None)
            ws.cell(row_idx, 1, subset)
            ws.cell(row_idx, 2, srow["Name"])
            for c, rk in enumerate(REGION_KEYS, 3):
                ws.cell(row_idx, c, srow.get(rk, ""))
            ws.cell(row_idx, len(REGION_KEYS) + 3, srow["VAS-global"])
            ws.cell(row_idx, len(REGION_KEYS) + 4, srow["FA-global-calibrated"])
            ws.cell(row_idx, len(REGION_KEYS) + 5, srow["FA-global-calibrated"] - srow["VAS-global"])
            row_idx += 1
    wb.save(OUT_EVAL / "part4_split_results_283.xlsx")

    payload = {"note": "283-subject pred-route evaluation.", "metrics": metrics,
               "splits": {k: [r["Name"] for r in v] for k, v in splits.items()}}
    with open(OUT_EVAL / "part4_metrics_283.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Part 4: 283 subjects evaluated")
    for subset in ["train", "val", "test", "all"]:
        m = metrics[subset]
        ba = m["bland_altman"]
        print(f"  {subset:5s}: n={m['n']}, r={m['pearson_r']:.4f}, R²={m['r_squared']:.4f}, "
              f"RMSE={m['rmse']:.4f}, BA bias={ba['bias']:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    entries = load_283_entries()
    print(f"Total subjects: {len(entries)}")

    # Part 1
    print("\n=== Part 1: Expert vs AI axis (283, pred route) ===")
    axis_data = build_part1(entries)
    save_part1(axis_data)

    # Part 2
    print("\n=== Part 2: 5-region RAS + FA (283) ===")
    region_data = build_part2(entries)
    save_part2(region_data)

    # Part 3
    print("\n=== Part 3: Weight training (regression + SHAP) ===")
    vas_map = {e["name"]: e["vas"] for e in entries if e["vas"] is not None}
    print(f"  VAS scores loaded: {len(vas_map)}")
    pipeline_result = run_part3(region_data, vas_map)

    # Part 4
    print("\n=== Part 4: 7:2:1 Evaluation ===")
    save_part4(pipeline_result)

    print("\nDone. All 283-subject (pred route) outputs saved.")


if __name__ == "__main__":
    main()
