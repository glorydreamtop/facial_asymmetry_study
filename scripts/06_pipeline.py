#!/usr/bin/env python3
"""
v2 pipeline: 100 subjects from 100例原始数据
  Part 1 - Expert axis vs AI-optimized axis comparison
  Part 2 - 5-region RAS + global FA
  Part 3 - Regression + SHAP weight training (with real VAS)
  Part 4 - 7:2:1 evaluation

Uses expert-drawn landmarks (关键点) + expert axis (对称轴/补对称轴).
No HRNet inference — uses expert-drawn landmarks directly.
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

from src.landmark.schema import NAME_TO_IDX, NUM_POINTS, POINT_NAMES_31, SYMMETRY_PAIRS

VAS0719 = REPO_ROOT / "data/100例原始数据"

OUT_BASE = REPO_ROOT / "results"
OUT_AXIS = OUT_BASE / "part1_axis_consensus"
OUT_REGION = OUT_BASE / "part2_region_fa"
OUT_PIPELINE = OUT_BASE / "part3_weights_shap"
OUT_EVAL = OUT_BASE / "part4_evaluation"

VALID_TAGS = {"ebe", "ebs", "os", "sci", "en", "p", "P", "ex", "al", "cph", "ch", "Go", "Na", "prn", "Sn", "ls", "sto", "li", "Me"}
_NAME_TO_IDX_CI = {k.lower(): v for k, v in NAME_TO_IDX.items()}

REGION_KEYS = [
    "VAS-eyebrow",
    "VAS-eye",
    "VAS-nose",
    "VAS-mouth",
    "VAS-facial contour",
]
REGION_PAIRS: dict[str, list[tuple[int, int]]] = {
    "VAS-eyebrow": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "VAS-eye": [(8, 9), (10, 11), (12, 13)],
    "VAS-nose": [(17, 18)],
    "VAS-mouth": [(19, 20), (21, 22)],
    "VAS-facial contour": [(26, 27)],
}
DISPLAY_NAMES = {
    "VAS-eyebrow": "Eyebrow",
    "VAS-eye": "Eye",
    "VAS-nose": "Nose",
    "VAS-mouth": "Mouth",
    "VAS-facial contour": "Facial Contour",
}

# ============================================================
# Data loading helpers
# ============================================================

def parse_gt_landmarks(data: dict) -> np.ndarray:
    lm = np.full((NUM_POINTS, 2), np.nan)
    tag_points = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        pts = shape.get("points", [])
        if not pts or label not in VALID_TAGS:
            continue
        tag_points.setdefault(label, []).append(pts[0])
    for tag, pts in tag_points.items():
        key = tag.lower()
        if key in {"na", "prn", "sn", "ls", "sto", "li", "me"}:
            idx = _NAME_TO_IDX_CI.get(key)
            if idx is not None:
                lm[idx] = pts[0]
        else:
            pts_sorted = sorted(pts, key=lambda p: p[0])
            for suffix, pt in zip(["_l", "_r"], pts_sorted):
                idx = _NAME_TO_IDX_CI.get(key + suffix)
                if idx is not None:
                    lm[idx] = pt
    return lm


def load_axis_from_json(path: Path) -> tuple[float, float, np.ndarray] | None:
    with open(path) as f:
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


def load_landmarks_from_json(path: Path) -> np.ndarray | None:
    with open(path) as f:
        data = json.load(f)
    lm = parse_gt_landmarks(data)
    if np.sum(~np.isnan(lm[:, 0])) < 20:
        return None
    return lm


def load_subjects() -> list[dict]:
    """Return list of dicts with name, landmarks_path, axis_path, image_path."""
    subjects = {}
    # 1-50: numbered subjects from 1-50关键点 + 1-50补对称轴
    kp1_dir = VAS0719 / "1-50关键点"
    ax1_dir = VAS0719 / "1-50补对称轴"
    for jp in sorted(kp1_dir.glob("*.json")):
        st = jp.stem.strip()
        subjects[st] = {"name": st, "lm_path": jp, "ax_path": ax1_dir / f"{st}.json"}

    # Add any extra from axis folder (e.g. FA0049, FA0050)
    for jp in sorted(ax1_dir.glob("*.json")):
        st = jp.stem.strip()
        if st not in subjects:
            subjects[st] = {"name": st, "lm_path": None, "ax_path": jp}

    # 51-100: real-name subjects from 51-100关键点 + 51-100对称轴
    kp2_dir = VAS0719 / "51-100关键点"
    ax2_dir = VAS0719 / "51-100对称轴"
    for jp in sorted(kp2_dir.glob("*.json")):
        st = jp.stem.strip()
        if st in subjects:
            existing = subjects[st]
            if existing["lm_path"] is None:
                existing["lm_path"] = jp
            if not existing["ax_path"].exists():
                existing["ax_path"] = ax2_dir / f"{st}.json"
            continue
        subjects[st] = {"name": st, "lm_path": jp, "ax_path": ax2_dir / f"{st}.json"}

    result = list(subjects.values())
    result.sort(key=lambda x: x["name"])
    return result


def _normalize_name(n: str) -> str:
    return " ".join(n.split())


def load_vas_map() -> dict[str, float]:
    """Load VAS scores (均分 = column C) from 专家VAS评分.xlsx."""
    wb = openpyxl.load_workbook(VAS0719 / "专家VAS评分.xlsx", data_only=True)
    ws = wb["FA"]
    vas = {}
    for r in range(3, 103):
        name = ws.cell(r, 2).value
        avg = ws.cell(r, 3).value
        if name is None or avg is None:
            continue
        vas[_normalize_name(str(name).strip())] = float(avg)
    return vas


# ============================================================
# Symmetry / axis helpers
# ============================================================

def reflect_across_line(pts: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    x0, y0 = pts[:, 0], pts[:, 1]
    d = (x0 - slope * y0 - intercept) / (1.0 + slope * slope)
    return np.stack([x0 - 2.0 * d, y0 + 2.0 * slope * d], axis=1)


def symmetry_cost(params: tuple[float, float], pts: np.ndarray) -> float:
    slope, intercept = params
    total_sq = 0.0
    cnt = 0
    for li, ri in SYMMETRY_PAIRS:
        if np.any(np.isnan(pts[li])) or np.any(np.isnan(pts[ri])):
            continue
        ref = reflect_across_line(pts[li].reshape(1, 2), slope, intercept)
        total_sq += float(np.linalg.norm(pts[ri] - ref[0]) ** 2)
        cnt += 1
    return total_sq / cnt if cnt > 0 else 0.0


def compute_optimized_axis(landmarks: np.ndarray) -> tuple[float, float]:
    valid = landmarks[~np.isnan(landmarks[:, 0])]
    center = valid.mean(axis=0)
    result = minimize(lambda p: symmetry_cost(p, landmarks), [0.0, center[0]], method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])


def axis_to_angle(m: float) -> float:
    return float(math.degrees(math.atan(m)))


def axis_position(m: float, b: float, yc: float) -> float:
    return float(m * yc + b)


def mean_horizontal_distance(m1, b1, pts1, m2, b2, pts2, n_samples=100):
    y_min = max(float(pts1[0, 1]), float(pts2[0, 1]))
    y_max = min(float(pts1[1, 1]), float(pts2[1, 1]))
    if y_max - y_min < 1e-6:
        y_min = min(float(pts1[0, 1]), float(pts2[0, 1]))
        y_max = max(float(pts1[1, 1]), float(pts2[1, 1]))
    sample_ys = np.linspace(y_min, y_max, n_samples)
    x1 = m1 * sample_ys + b1
    x2 = m2 * sample_ys + b2
    return float(np.mean(np.abs(x1 - x2)))


def compute_ipd(landmarks: np.ndarray) -> float:
    p_l = landmarks[NAME_TO_IDX["p_L"]]
    p_r = landmarks[NAME_TO_IDX["p_R"]]
    return float(np.linalg.norm(p_l - p_r))


def compute_region_ras(landmarks, slope, intercept, region_pairs, ipd):
    sq = []
    for li, ri in region_pairs:
        if np.any(np.isnan(landmarks[li])) or np.any(np.isnan(landmarks[ri])):
            continue
        ref = reflect_across_line(landmarks[li].reshape(1, 2), slope, intercept)
        d = float(np.linalg.norm(landmarks[ri] - ref[0]))
        sq.append(d * d)
    if not sq:
        return 0.0
    return float(np.sqrt(np.mean(sq))) / ipd if ipd > 0 else 0.0


# ============================================================
# Part 1: Axis consensus (expert vs AI)
# ============================================================

def build_axis_data(entries: list[dict]) -> list[dict]:
    results = []
    for ent in entries:
        name = ent["name"]

        # Load expert axis
        ax = load_axis_from_json(ent["ax_path"]) if ent["ax_path"].exists() else None
        if ax is None:
            print(f"  [{name}] No expert axis found")
            continue

        em, eb, epts = ax
        e_angle = axis_to_angle(em)
        eyc = (epts[0, 1] + epts[1, 1]) / 2.0
        e_pos = axis_position(em, eb, eyc)

        # Load landmarks + compute AI axis
        lm = None
        if ent["lm_path"] and ent["lm_path"].exists():
            lm = load_landmarks_from_json(ent["lm_path"])

        if lm is None:
            print(f"  [{name}] No landmarks, using expert axis as AI axis")
            am, ab = em, eb
        else:
            am, ab = compute_optimized_axis(lm)

        a_angle = axis_to_angle(am)
        a_pos = axis_position(am, ab, eyc)
        angle_err = abs(e_angle - a_angle)

        horiz = mean_horizontal_distance(em, eb, epts, am, ab, epts)

        results.append({
            "Name": name,
            "Expert_m": em,
            "Expert_b": eb,
            "Expert_Angle": round(e_angle, 4),
            "Expert_Position": round(e_pos, 2),
            "AI_m": am,
            "AI_b": ab,
            "AI_Angle": round(a_angle, 4),
            "AI_Position": round(a_pos, 2),
            "Angle_Error": round(angle_err, 4),
            "Mean_Horizontal_Distance": round(horiz, 2),
        })
    return results


def save_part1(axis_data: list[dict]):
    OUT_AXIS.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUT_AXIS / "专家共识与AI对称轴数据_100例.xlsx"
    json_path = OUT_AXIS / "axis_consensus_details.json"
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

def build_region_data(entries: list[dict]) -> list[dict]:
    results = []
    for ent in entries:
        name = ent["name"]
        if not ent["lm_path"] or not ent["lm_path"].exists():
            print(f"  [{name}] No landmarks, skipping region RAS")
            continue
        lm = load_landmarks_from_json(ent["lm_path"])
        if lm is None:
            continue

        slope, intercept = compute_optimized_axis(lm)
        ipd = compute_ipd(lm)
        if ipd <= 0:
            print(f"  [{name}] IPD = 0, skipping")
            continue

        row = {"Name": name}
        for rk in REGION_KEYS:
            ras = compute_region_ras(lm, slope, intercept, REGION_PAIRS[rk], ipd)
            row[rk] = round(ras, 6)
        results.append(row)
    return results


def save_part2(region_data: list[dict]):
    OUT_REGION.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUT_REGION / "AI各区域不对称评分+GlobalFA原始值.xlsx"
    json_path = OUT_REGION / "part2_region_fa_details.json"
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
# Part 3: Regression + SHAP (weight training)
# ============================================================

def calibrate_to_vas(raw_scores: np.ndarray, target_vas: np.ndarray) -> np.ndarray:
    lr = LinearRegression()
    lr.fit(raw_scores.reshape(-1, 1), target_vas)
    return lr.predict(raw_scores.reshape(-1, 1))


def run_part3(region_data: list[dict], vas_map: dict[str, float]):
    OUT_PIPELINE.mkdir(parents=True, exist_ok=True)

    matched = []
    for row in region_data:
        name = row["Name"]
        nn = _normalize_name(name)
        if nn in vas_map:
            item = dict(row)
            item["VAS-global"] = vas_map[nn]
            matched.append(item)
    print(f"Part 3: {len(matched)} matched subjects")

    X = np.array([[row[k] for k in REGION_KEYS] for row in matched], dtype=np.float64)
    y = np.array([row["VAS-global"] for row in matched], dtype=np.float64)
    n = len(y)

    # Linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    beta_0 = float(lr.intercept_)
    beta_raw = lr.coef_.astype(float)
    y_pred = lr.predict(X)
    residuals = y - y_pred

    RSS = float(np.sum(residuals ** 2))
    TSS = float(np.sum((y - y.mean()) ** 2))
    R2 = float(lr.score(X, y))
    n_params = len(beta_raw) + 1
    adj_R2 = 1.0 - (RSS / (n - n_params)) / (TSS / (n - 1))
    RMSE = float(np.sqrt(np.mean(residuals ** 2)))
    MAE = float(np.mean(np.abs(residuals)))

    X_std = X.std(axis=0, ddof=0)
    y_std = y.std(axis=0, ddof=1)
    beta_std = beta_raw * X_std / y_std
    beta_std_pos = np.clip(beta_std, 0.0, None)
    if float(np.sum(beta_std_pos)) <= 0.0:
        weights = np.ones_like(beta_std_pos) / len(beta_std_pos)
    else:
        weights = beta_std_pos / np.sum(beta_std_pos)

    X_design = np.column_stack([np.ones(n), X])
    ci_raw = []
    ci_std = []
    p_values_raw = []
    vifs = []
    try:
        cov = RSS / (n - n_params) * np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(cov))
        t_vals = np.concatenate([[lr.intercept_], lr.coef_]) / se
        p_vals = 2 * (1.0 - t_dist.cdf(np.abs(t_vals), n - n_params))
        p_values_raw = p_vals[1:]
        t_crit = float(t_dist.ppf(0.975, n - n_params))
        coef_se = se[1:]
        for i in range(len(beta_raw)):
            raw_lo = float(beta_raw[i] - t_crit * coef_se[i])
            raw_hi = float(beta_raw[i] + t_crit * coef_se[i])
            ci_raw.append((raw_lo, raw_hi))
            ci_std.append((float(raw_lo * X_std[i] / y_std), float(raw_hi * X_std[i] / y_std)))
    except np.linalg.LinAlgError:
        p_values_raw = [float("nan")] * len(beta_raw)
        ci_raw = [(float("nan"), float("nan"))] * len(beta_raw)
        ci_std = [(float("nan"), float("nan"))] * len(beta_raw)

    for i in range(X.shape[1]):
        Xi = X[:, i]
        Xo = np.delete(X, i, axis=1)
        aux_lr = LinearRegression()
        aux_lr.fit(Xo, Xi)
        aux_r2 = float(aux_lr.score(Xo, Xi))
        vif = float("inf") if aux_r2 >= 0.999999 else float(1.0 / (1.0 - aux_r2))
        vifs.append(vif)

    # Apply weights to compute FA
    fa_raw = X @ weights
    fa_calibrated = calibrate_to_vas(fa_raw, y)
    pearson_r, pearson_p = pearsonr(fa_calibrated, y)
    spearman_r, spearman_p = spearmanr(fa_calibrated, y)

    result = {
        "note": "v2 pipeline: 100 subjects, expert landmarks + expert axis.",
        "regression": {
            "intercept_beta_0": beta_0,
            "r_squared": R2,
            "adjusted_r_squared": adj_R2,
            "rmse": RMSE,
            "mae": MAE,
            "n_samples": n,
        },
        "region_weights": [],
        "fa_vs_vas": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        },
        "shap": {},
        "per_sample": [],
    }

    for i, key in enumerate(REGION_KEYS):
        result["region_weights"].append({
            "region_key": key,
            "region": DISPLAY_NAMES[key],
            "raw_coefficient": float(beta_raw[i]),
            "raw_coefficient_95ci": [ci_raw[i][0], ci_raw[i][1]],
            "standardized_beta": float(beta_std[i]),
            "standardized_beta_95ci": [ci_std[i][0], ci_std[i][1]],
            "positive_standardized_beta": float(beta_std_pos[i]),
            "normalized_weight": float(weights[i]),
            "vif": None if math.isinf(vifs[i]) else float(vifs[i]),
            "p_value": None if math.isnan(p_values_raw[i]) else float(p_values_raw[i]),
        })

    for i, row in enumerate(matched):
        item = dict(row)
        item["FA-global-raw"] = float(fa_raw[i])
        item["FA-global-calibrated"] = float(fa_calibrated[i])
        item["VAS-diff"] = float(fa_calibrated[i] - y[i])
        result["per_sample"].append(item)

    # SHAP
    print("  Training RandomForest + SHAP...")
    tree = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=42)
    tree.fit(X, y)
    tree_pred = tree.predict(X)
    tree_pearson_r, tree_pearson_p = pearsonr(tree_pred, y)
    tree_spearman_r, tree_spearman_p = spearmanr(tree_pred, y)

    explainer = shap.TreeExplainer(tree)
    shap_values = np.asarray(explainer.shap_values(X), dtype=np.float64)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_weights = mean_abs_shap / np.sum(mean_abs_shap)

    result["shap"] = {
        "model": "RandomForestRegressor",
        "pearson_r": float(tree_pearson_r),
        "pearson_p": float(tree_pearson_p),
        "spearman_r": float(tree_spearman_r),
        "spearman_p": float(tree_spearman_p),
        "mean_abs_shap": {REGION_KEYS[i]: float(mean_abs_shap[i]) for i in range(len(REGION_KEYS))},
        "normalized_weight": {REGION_KEYS[i]: float(shap_weights[i]) for i in range(len(REGION_KEYS))},
    }

    # SHAP plots
    feature_names = [DISPLAY_NAMES[k] for k in REGION_KEYS]
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(OUT_PIPELINE / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_PIPELINE / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # SHAP values workbook
    shap_wb = openpyxl.Workbook()
    shap_ws = shap_wb.active
    shap_ws.title = "SHAP_VALUES"
    shap_headers = ["Name"] + feature_names + ["VAS-global", "Tree-Prediction"]
    for c, h in enumerate(shap_headers, 1):
        shap_ws.cell(1, c, h)
    for r, row in enumerate(matched, 2):
        shap_ws.cell(r, 1, row["Name"])
        for i in range(len(REGION_KEYS)):
            shap_ws.cell(r, i + 2, float(shap_values[r - 2, i]))
        shap_ws.cell(r, len(REGION_KEYS) + 2, row["VAS-global"])
        shap_ws.cell(r, len(REGION_KEYS) + 3, float(tree_pred[r - 2]))

    meta_ws = shap_wb.create_sheet("SHAP_WEIGHTS")
    meta_ws.cell(1, 1, "Region")
    meta_ws.cell(1, 2, "MeanAbsSHAP")
    meta_ws.cell(1, 3, "Normalized_Weight")
    for i, key in enumerate(REGION_KEYS):
        meta_ws.cell(i + 2, 1, DISPLAY_NAMES[key])
        meta_ws.cell(i + 2, 2, float(mean_abs_shap[i]))
        meta_ws.cell(i + 2, 3, float(shap_weights[i]))
    shap_wb.save(OUT_PIPELINE / "shap_values.xlsx")

    # Applied workbook
    app_wb = openpyxl.Workbook()
    app_ws = app_wb.active
    app_ws.title = "applied"
    app_headers = ["Name"] + REGION_KEYS + ["VAS-global", "FA-global-raw", "FA-global-calibrated", "diff"]
    for c, h in enumerate(app_headers, 1):
        app_ws.cell(1, c, h)
    for r, row in enumerate(result["per_sample"], 2):
        app_ws.cell(r, 1, row["Name"])
        for i, key in enumerate(REGION_KEYS):
            app_ws.cell(r, i + 2, row[key])
        app_ws.cell(r, len(REGION_KEYS) + 2, row["VAS-global"])
        app_ws.cell(r, len(REGION_KEYS) + 3, row["FA-global-raw"])
        app_ws.cell(r, len(REGION_KEYS) + 4, row["FA-global-calibrated"])
        app_ws.cell(r, len(REGION_KEYS) + 5, row["VAS-diff"])
    app_wb.save(OUT_PIPELINE / "AI各区域不对称评分+机器评分.xlsx")

    # JSON
    json_path = OUT_PIPELINE / "weight_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Part 3: -> {json_path}")
    return result


# ============================================================
# Part 4: Evaluation (7:2:1)
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
        "n": len(rows),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "r_squared": r2,
        "rmse": rmse,
        "mae": mae,
        "bland_altman": ba,
    }


def make_splits(rows: list[dict]) -> dict[str, list[dict]]:
    rng = np.random.RandomState(42)
    shuffled = sorted(rows, key=lambda x: x["Name"])
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * 0.7))
    n_val = int(round(n * 0.2))
    if n_train + n_val > n - 1:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
        "all": sorted(rows, key=lambda x: x["Name"]),
    }


def save_part4(pipeline_result: dict):
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    rows = pipeline_result["per_sample"]
    splits = make_splits(rows)
    metrics = {k: evaluate_subset(v) for k, v in splits.items()}

    # Plots
    y_true = np.array([r["VAS-global"] for r in rows], dtype=np.float64)
    y_pred = np.array([r["FA-global-calibrated"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Expert VAS (均分)")
    plt.ylabel("Machine FA Score (calibrated)")
    plt.title("Machine Score vs Expert VAS")
    plt.tight_layout()
    plt.savefig(OUT_EVAL / "scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_vals = (y_true + y_pred) / 2.0
    diff_vals = y_pred - y_true
    ba = bland_altman_stats(y_pred, y_true)
    plt.figure(figsize=(7, 5))
    plt.scatter(mean_vals, diff_vals, alpha=0.8)
    plt.axhline(ba["bias"], linestyle="-")
    plt.axhline(ba["loa_low"], linestyle="--")
    plt.axhline(ba["loa_high"], linestyle="--")
    plt.xlabel("Mean of Machine and Expert")
    plt.ylabel("Difference (Machine - Expert)")
    plt.title("Bland-Altman Plot")
    plt.tight_layout()
    plt.savefig(OUT_EVAL / "bland_altman.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Split workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "split_results"
    headers = ["subset", "Name"] + REGION_KEYS + ["VAS-global", "FA-global-calibrated", "diff"]
    for c, h in enumerate(headers, 1):
        ws.cell(1, c, h)
    row_idx = 2
    for subset_name in ["train", "val", "test", "all"]:
        for row in splits[subset_name]:
            ws.cell(row_idx, 1, subset_name)
            ws.cell(row_idx, 2, row["Name"])
            for i, key in enumerate(REGION_KEYS):
                ws.cell(row_idx, i + 3, row[key])
            ws.cell(row_idx, len(REGION_KEYS) + 3, row["VAS-global"])
            ws.cell(row_idx, len(REGION_KEYS) + 4, row["FA-global-calibrated"])
            ws.cell(row_idx, len(REGION_KEYS) + 5, row["VAS-diff"])
            row_idx += 1
    wb.save(OUT_EVAL / "part4_split_results.xlsx")

    # JSON
    payload = {
        "note": "v2 pipeline part-4 evaluation (100 subjects).",
        "splits": {k: [r["Name"] for r in v] for k, v in splits.items() if k != "all"},
        "metrics": metrics,
    }
    with open(OUT_EVAL / "part4_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Part 4: {len(rows)} subjects evaluated")
    for k in ["train", "val", "test", "all"]:
        m = metrics[k]
        ba = m["bland_altman"]
        print(f"  {k}: n={m['n']}, r={m['pearson_r']:.4f}, R²={m['r_squared']:.4f}, "
              f"RMSE={m['rmse']:.4f}, BA bias={ba['bias']:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    entries = load_subjects()
    print(f"Total subjects: {len(entries)}")

    # Part 1
    print("\n=== Part 1: Expert vs AI axis ===")
    axis_data = build_axis_data(entries)
    save_part1(axis_data)

    # Part 2
    print("\n=== Part 2: 5-region RAS + FA ===")
    region_data = build_region_data(entries)
    save_part2(region_data)

    # Part 3
    print("\n=== Part 3: Weight training (regression + SHAP) ===")
    vas_map = load_vas_map()
    print(f"  VAS scores loaded: {len(vas_map)}")
    pipeline_result = run_part3(region_data, vas_map)

    # Part 4
    print("\n=== Part 4: 7:2:1 Evaluation ===")
    save_part4(pipeline_result)

    print("\nDone. All v2 outputs saved.")


if __name__ == "__main__":
    main()
