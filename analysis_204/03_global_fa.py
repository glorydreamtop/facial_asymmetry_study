"""Final analysis: anchored scoring, Global FA formula, routes A/B/C, supplements.

Route definitions
-----------------
C : expert landmarks + expert consensus axis (reference)
B : expert landmarks + Ridge axis predicted from expert landmarks
A : automatic landmarks + Ridge axis predicted from automatic landmarks

The symmetry axis is anchored at the interpupillary midpoint before RAS
computation so that the score is invariant to a lateral translation of the
axis; axis-accuracy tables still use the raw (unanchored) axes.

Global FA formula is fitted on train+val (expert points + consensus axis) and
frozen; the same coefficients are applied to A and B.
"""
from __future__ import annotations

import csv
import json
import pickle
import joblib
import numpy as np

import common as C
from axis_features import (
    build_axis_feature_vector,
    compute_sample_normalization,
    target_to_image_line,
)

Y0, Y1 = 500.0, 2280.0


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def anchor_b(m, b, yc, anchor_x):
    return float(anchor_x - m * yc)


def ras_anchored(lm, m, b, yc, anchor_x, ipd):
    return C.region_ras(lm, m, anchor_b(m, b, yc, anchor_x), ipd)


def predict_axes(model, lms):
    X, centers, scales = [], [], []
    for lm in lms:
        c0, s0 = compute_sample_normalization(lm)
        X.append(build_axis_feature_vector(lm, c0, s0))
        centers.append(c0)
        scales.append(s0)
    targets = model.predict_targets(np.asarray(X, np.float32))
    return [target_to_image_line(t, centers[i], float(scales[i])) for i, t in enumerate(targets)]


def agreement(machine, expert, n_boot=2000):
    machine = np.asarray(machine, float)
    expert = np.asarray(expert, float)
    r = float(np.corrcoef(machine, expert)[0, 1])
    r = float(np.clip(r, -0.999999, 0.999999))
    r_ci = C.pearson_ci(r, len(machine))
    ss_res = float(((expert - machine) ** 2).sum())
    ss_tot = float(((expert - expert.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(machine - expert)))
    rmse = float(np.sqrt(np.mean((machine - expert) ** 2)))
    mae_ci = C.bootstrap_ci(np.arange(len(machine)),
                            lambda idx: float(np.mean(np.abs(machine[idx] - expert[idx]))), n_boot=n_boot)
    rmse_ci = C.bootstrap_ci(np.arange(len(machine)),
                             lambda idx: float(np.sqrt(np.mean((machine[idx] - expert[idx]) ** 2))), n_boot=n_boot)
    icc = C.icc_a1(machine, expert)
    icc_ci = C.bootstrap_ci(np.arange(len(machine)),
                            lambda idx: C.icc_a1(machine[idx], expert[idx]), n_boot=n_boot)
    ba = C.bland_altman(machine, expert)
    return {
        "n": int(len(machine)),
        "pearson_r": r, "pearson_r_ci": [float(r_ci[0]), float(r_ci[1])],
        "r2": float(r2), "mae": mae, "mae_ci": [float(mae_ci[0]), float(mae_ci[1])],
        "rmse": rmse, "rmse_ci": [float(rmse_ci[0]), float(rmse_ci[1])],
        "icc_a1": float(icc), "icc_a1_ci": [float(icc_ci[0]), float(icc_ci[1])],
        "bias": float(ba["bias"]), "sd_diff": float(ba["sd"]),
        "loa_low": float(ba["loa_low"]), "loa_high": float(ba["loa_high"]),
    }


def axis_rows(route_line, cohort, indices):
    rows = []
    for k in indices:
        line = route_line[k]
        c = cohort[k]
        pm, pb = line
        gm, gb = c["cons_m"], c["cons_b"]
        yc = c["yc"]
        rows.append({
            "stem": c["stem"],
            "angle": abs(C.angle_deg(pm) - C.angle_deg(gm)),
            "position": abs((pm - gm) * yc + (pb - gb)),
            "mhd": C.mhd(pm, pb, gm, gb, Y0, Y1),
            "signed_angle": C.angle_deg(pm) - C.angle_deg(gm),
            "signed_position": (pm - gm) * yc + (pb - gb),
        })
    return rows


def axis_summary(rows):
    out = {}
    for key in ("angle", "position", "mhd"):
        arr = np.array([r[key] for r in rows])
        m, s = C.mean_sd(arr)
        med, q1, q3 = C.median_iqr(arr)
        lo, hi = C.t_ci(arr)
        out[key] = {"mean": m, "sd": s, "median": med, "q1": q1, "q3": q3,
                    "ci_low": lo, "ci_high": hi}
    return out


def diff_summary(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, s = C.mean_sd(d)
    med, q1, q3 = C.median_iqr(d)
    return {"mean": m, "sd": s, "median": med, "q1": q1, "q3": q3}


# ----------------------------------------------------------------------
def main():
    cohort = pickle.load(open(C.OUT / "cohort.pkl", "rb"))
    split = json.load(open(C.DATA / "split_204.json", encoding="utf-8"))
    ridge = joblib.load(C.OUT / "models" / "ridge_expert.joblib")

    for c in cohort:
        c["ipd_machine"] = C.compute_ipd(c["machine_lm"])
        c["anchor_e"] = float((c["expert_lm"][10, 0] + c["expert_lm"][11, 0]) / 2.0)
        c["anchor_m"] = float((c["machine_lm"][10, 0] + c["machine_lm"][11, 0]) / 2.0)

    idx = {c["stem"]: i for i, c in enumerate(cohort)}
    tri = [idx[s] for s in split["train"]]
    vai = [idx[s] for s in split["val"]]
    tei = [idx[s] for s in split["test"]]
    tvi = tri + vai

    # ---------------- axes ----------------
    lines_b = predict_axes(ridge, [c["expert_lm"] for c in cohort])
    lines_a = predict_axes(ridge, [c["machine_lm"] for c in cohort])

    # ---------------- RAS ----------------
    K = C.REGION_KEYS
    ras_c, ras_b, ras_a = [], [], []
    for i, c in enumerate(cohort):
        ras_c.append([ras_anchored(c["expert_lm"], c["cons_m"], c["cons_b"], c["yc"], c["anchor_e"], c["ipd"])[k] for k in K])
        ras_b.append([ras_anchored(c["expert_lm"], lines_b[i][0], lines_b[i][1], c["yc"], c["anchor_e"], c["ipd"])[k] for k in K])
        ras_a.append([ras_anchored(c["machine_lm"], lines_a[i][0], lines_a[i][1], c["yc"], c["anchor_m"], c["ipd_machine"])[k] for k in K])
    Xc, Xb, Xa = np.array(ras_c), np.array(ras_b), np.array(ras_a)

    # ---------------- expert VAS ----------------
    vas = np.array([c["vas"] for c in cohort], dtype=np.float64)
    vas_a = np.array([c["vas_a"] for c in cohort], dtype=np.float64)
    vas_b = np.array([c["vas_b"] for c in cohort], dtype=np.float64)

    # ---------------- formula ----------------
    fit = C.regression_stats(Xc[tvi], vas[tvi], [C.REGION_LABELS[k] for k in K])
    B = fit["beta"]
    zc = B[0] + Xc @ B[1:]
    zb = B[0] + Xb @ B[1:]
    za = B[0] + Xa @ B[1:]
    fc, fb, fa = np.clip(zc, 0, 100), np.clip(zb, 0, 100), np.clip(za, 0, 100)
    rc_all = {
        "raw_min": float(zc.min()), "raw_max": float(zc.max()),
        "fa_min": float(fc.min()), "fa_max": float(fc.max()),
        "n_below_0": int((zc < 0).sum()), "n_above_100": int((zc > 100).sum()),
        "out_of_range_pct": float((((zc < 0) | (zc > 100)).sum()) / len(zc) * 100.0),
    }
    rc_test = {
        "raw_min": float(zc[tei].min()), "raw_max": float(zc[tei].max()),
        "fa_min": float(fc[tei].min()), "fa_max": float(fc[tei].max()),
        "n_below_0": int((zc[tei] < 0).sum()), "n_above_100": int((zc[tei] > 100).sum()),
        "out_of_range_pct": float((((zc[tei] < 0) | (zc[tei] > 100)).sum()) / len(tei) * 100.0),
    }

    # ---------------- agreement ----------------
    routes = {
        "A": agreement(fa[tei], vas[tei]),
        "B": agreement(fb[tei], vas[tei]),
        "C": agreement(fc[tei], vas[tei]),
    }

    # ---------------- axis errors (raw axes) ----------------
    ax_a = axis_summary(axis_rows(lines_a, cohort, tei))
    ax_b = axis_summary(axis_rows(lines_b, cohort, tei))
    ab_rows = []
    for i in tei:
        pm, pb = lines_a[i]
        qm, qb = lines_b[i]
        yc = cohort[i]["yc"]
        ab_rows.append({
            "angle": abs(C.angle_deg(pm) - C.angle_deg(qm)),
            "position": abs((pm - qm) * yc + (pb - qb)),
            "mhd": C.mhd(pm, pb, qm, qb, Y0, Y1),
        })
    ax_ab = axis_summary(ab_rows)

    # ---------------- supplement 2 ----------------
    ind_rows = {}
    for ann in ("A1", "A2", "B1", "B2"):
        rows = []
        for i in tei:
            pm, pb = lines_a[i]
            em, eb = cohort[i]["axes"][ann]
            yc = cohort[i]["yc"]
            rows.append({
                "angle": abs(C.angle_deg(pm) - C.angle_deg(em)),
                "position": abs((pm - em) * yc + (pb - eb)),
                "mhd": C.mhd(pm, pb, em, eb, Y0, Y1),
            })
        ind_rows["AI vs " + ann] = axis_summary(rows)
    for pair in (("A1", "A2"), ("B1", "B2"), ("A1", "B1")):
        rows = []
        for i in tei:
            m1, b1 = cohort[i]["axes"][pair[0]]
            m2, b2 = cohort[i]["axes"][pair[1]]
            yc = cohort[i]["yc"]
            rows.append({
                "angle": abs(C.angle_deg(m1) - C.angle_deg(m2)),
                "position": abs((m1 - m2) * yc + (b1 - b2)),
                "mhd": C.mhd(m1, b1, m2, b2, Y0, Y1),
            })
        ind_rows[f"{pair[0]} vs {pair[1]}"] = axis_summary(rows)

    # ---------------- per-case table ----------------
    rows = []
    for i, c in enumerate(cohort):
        rows.append({
            "stem": c["stem"], "split": c["split"],
            "width": c["width"], "height": c["height"],
            "ipd_expert": c["ipd"], "ipd_machine": c["ipd_machine"], "yc": c["yc"],
            "expert_m": c["cons_m"], "expert_b": c["cons_b"],
            "ai_m_B": lines_b[i][0], "ai_b_B": lines_b[i][1],
            "ai_m_A": lines_a[i][0], "ai_b_A": lines_a[i][1],
            **{f"RAS_C_{k}": Xc[i, j] for j, k in enumerate(K)},
            **{f"RAS_B_{k}": Xb[i, j] for j, k in enumerate(K)},
            **{f"RAS_A_{k}": Xa[i, j] for j, k in enumerate(K)},
            "VAS": float(vas[i]), "VAS_A": float(vas_a[i]), "VAS_B": float(vas_b[i]),
            "z_C": float(zc[i]), "FA_C": float(fc[i]),
            "z_B": float(zb[i]), "FA_B": float(fb[i]),
            "z_A": float(za[i]), "FA_A": float(fa[i]),
        })
    with open(C.OUT / "final_per_case.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- descriptives ----------------
    ipds = np.array([c["ipd"] for c in cohort])
    res = {}
    for c in cohort:
        key = f"{c['width']}x{c['height']}"
        res[key] = res.get(key, 0) + 1
    desc = {
        "n": len(cohort),
        "resolution": res,
        "ipd": {"mean": float(ipds.mean()), "sd": float(ipds.std(ddof=1)),
                "median": float(np.median(ipds)), "min": float(ipds.min()), "max": float(ipds.max())},
        "vas": {"mean": float(vas.mean()), "sd": float(vas.std(ddof=1)),
                "median": float(np.median(vas)), "min": float(vas.min()), "max": float(vas.max())},
        "split": {k: len(v) for k, v in split.items()},
    }

    # ---------------- save ----------------
    payload = {
        "cohort": len(cohort),
        "split": {k: len(v) for k, v in split.items()},
        "formula": {
            "intercept": float(B[0]), "intercept_ci": fit["intercept_ci"],
            "terms": fit["rows"], "n_fit": fit["n"],
            "r2": fit["r2"], "adj_r2": fit["adj_r2"], "rmse": fit["rmse"], "mae": fit["mae"],
        },
        "range_check_all": rc_all, "range_check_test": rc_test,
        "routes": routes,
        "axis_errors_test": {"A": ax_a, "B": ax_b, "A_vs_B": ax_ab},
        "supplement2": ind_rows,
        "score_diffs_test": {
            "A_minus_B": diff_summary(fa[tei], fb[tei]),
            "B_minus_C": diff_summary(fb[tei], fc[tei]),
        },
        "descriptives": desc,
    }
    json.dump(payload, open(C.OUT / "final_results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("formula B", np.round(B, 3), "R2", round(fit["r2"], 4))
    print("range all", rc_all)
    for key in ("A", "B", "C"):
        r = routes[key]
        print(f"{key}: r={r['pearson_r']:.3f} R2={r['r2']:.3f} MAE={r['mae']:.2f} "
              f"RMSE={r['rmse']:.2f} ICC={r['icc_a1']:.3f} bias={r['bias']:.2f}")
    for key in ("A", "B", "A_vs_B"):
        s = payload["axis_errors_test"][key]
        print(f"axis {key}: angle {s['angle']['median']:.3f} ({s['angle']['q1']:.3f}-{s['angle']['q3']:.3f}) "
              f"pos {s['position']['median']:.2f} ({s['position']['q1']:.2f}-{s['position']['q3']:.2f})")
    print("saved final_results.json / final_per_case.csv")


if __name__ == "__main__":
    main()
