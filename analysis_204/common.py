"""Shared geometry and statistics helpers for the 204-case analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "out"
DELIV = BASE / "outputs"

REGION_PAIRS = {
    "eyebrow": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "eye": [(8, 9), (10, 11), (12, 13)],
    "nose": [(17, 18)],
    "mouth": [(19, 20), (21, 22)],
    "contour": [(26, 27)],
}
REGION_KEYS = list(REGION_PAIRS)
REGION_LABELS = {
    "eyebrow": "Eyebrow",
    "eye": "Eye",
    "nose": "Nose",
    "mouth": "Mouth",
    "contour": "Facial contour",
}
PAIR_INDEX = {
    "ebe": (0, 1), "ebs": (2, 3), "os": (4, 5), "sci": (6, 7),
    "en": (8, 9), "p": (10, 11), "ex": (12, 13), "al": (17, 18),
    "cph": (19, 20), "ch": (21, 22), "Go": (26, 27),
}


def angle_deg(m: float) -> float:
    return float(np.degrees(np.arctan(m)))


def mhd(m1: float, b1: float, m2: float, b2: float,
        y0: float = 500.0, y1: float = 2280.0, n: int = 100) -> float:
    ys = np.linspace(float(y0), float(y1), n)
    return float(np.mean(np.abs((m1 * ys + b1) - (m2 * ys + b2))))


def reflect_across_axis(pts: np.ndarray, m: float, b: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    d = (pts[:, 0] - m * pts[:, 1] - b) / (1.0 + m * m)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] - 2.0 * d
    out[:, 1] = pts[:, 1] + 2.0 * m * d
    return out


def compute_ipd(lm: np.ndarray) -> float:
    li, ri = PAIR_INDEX["p"]
    if np.any(np.isnan(lm[li])) or np.any(np.isnan(lm[ri])):
        return float("nan")
    return float(np.linalg.norm(lm[li] - lm[ri]))


def compute_yc(lm: np.ndarray) -> float:
    li, ri = PAIR_INDEX["p"]
    if np.any(np.isnan(lm[li])) or np.any(np.isnan(lm[ri])):
        return float("nan")
    return float((lm[li, 1] + lm[ri, 1]) / 2.0)


def region_ras(lm: np.ndarray, m: float, b: float, ipd: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for region, pairs in REGION_PAIRS.items():
        sq = []
        for li, ri in pairs:
            if np.any(np.isnan(lm[li])) or np.any(np.isnan(lm[ri])):
                continue
            ref = reflect_across_axis(lm[li].reshape(1, 2), m, b)[0]
            sq.append(float(np.linalg.norm(lm[ri] - ref) ** 2))
        out[region] = float(np.sqrt(np.mean(sq)) / ipd) if sq and ipd > 0 else float("nan")
    return out


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def mean_sd(x: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def median_iqr(x: Sequence[float]) -> Tuple[float, float, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def t_ci(x: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    from scipy import stats
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float("nan"), float("nan")
    se = a.std(ddof=1) / np.sqrt(len(a))
    h = stats.t.ppf(1 - alpha / 2, len(a) - 1) * se
    return float(a.mean() - h), float(a.mean() + h)


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            stats.append(stat_fn(values[idx]))
        except Exception:
            continue
    stats = np.asarray(stats, dtype=np.float64)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def pearson_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    from scipy import stats
    if n <= 3:
        return float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def bland_altman(a: Sequence[float], b: Sequence[float]) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def icc_a1(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    k = 2
    data = np.stack([x, y], axis=1)
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_total = ((data - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom != 0 else float("nan")


def regression_stats(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> dict:
    from scipy import stats
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = n - p - 1
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    tcrit = stats.t.ppf(0.975, dof)
    ci = np.column_stack([beta - tcrit * se, beta + tcrit * se])

    xs = X.std(axis=0, ddof=1)
    xs[xs == 0] = 1.0
    ys = y.std(ddof=1) if y.std(ddof=1) > 0 else 1.0
    beta_std = beta[1:] * xs / ys
    se_std = se[1:] * xs / ys
    ci_std = np.column_stack([beta_std - tcrit * se_std, beta_std + tcrit * se_std])

    y_pred = design @ beta
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y - y_pred)))

    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            "region": name,
            "B": float(beta[i + 1]),
            "B_ci_low": float(ci[i + 1, 0]),
            "B_ci_high": float(ci[i + 1, 1]),
            "beta_std": float(beta_std[i]),
            "beta_std_ci_low": float(ci_std[i, 0]),
            "beta_std_ci_high": float(ci_std[i, 1]),
            "p": float(pvals[i + 1]),
        })
    return {
        "intercept": float(beta[0]),
        "intercept_ci": [float(ci[0, 0]), float(ci[0, 1])],
        "rows": rows, "n": int(n),
        "r2": float(r2), "adj_r2": float(adj),
        "rmse": rmse, "mae": mae, "dof": int(dof),
        "beta": beta, "y_pred": y_pred, "resid": resid,
    }


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DELIV.mkdir(parents=True, exist_ok=True)
