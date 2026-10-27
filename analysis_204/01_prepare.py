"""Stage 1: load the landmark, axis, VAS and split tables and assemble the cohort.

Inputs (data/):
  expert_landmarks_204.npz   names, landmarks (204,29,2)
  automatic_landmarks_204.npz names, landmarks (204,29,2)
  axes_204.json              A1/A2/B1/B2 and consensus axis parameters per case
  vas_204.csv                stem, split, vas, vas_a, vas_b
  split_204.json             train/val/test lists

Output: out/cohort.pkl
"""
from __future__ import annotations

import csv
import json
import pickle

import numpy as np

import common as C


def main() -> None:
    C.ensure_out()
    expert = np.load(C.DATA / "expert_landmarks_204.npz", allow_pickle=True)
    automatic = np.load(C.DATA / "automatic_landmarks_204.npz", allow_pickle=True)
    auto = {str(n): lm.astype(np.float64) for n, lm in zip(automatic["names"], automatic["landmarks"])}
    axes = json.load(open(C.DATA / "axes_204.json", encoding="utf-8"))
    split = json.load(open(C.DATA / "split_204.json", encoding="utf-8"))
    split_of = {s: k for k, v in split.items() for s in v}

    vas = {}
    for row in csv.DictReader(open(C.DATA / "vas_204.csv", encoding="utf-8-sig")):
        vas[row["stem"]] = {
            "vas": float(row["vas"]),
            "vas_a": float(row["vas_a"]),
            "vas_b": float(row["vas_b"]),
        }

    cohort = []
    for name, lm in zip(expert["names"], expert["landmarks"]):
        stem = str(name)
        lm = lm.astype(np.float64)
        ax = {k: (float(v[0]), float(v[1])) for k, v in axes[stem].items() if k != "consensus"}
        cons = axes[stem]["consensus"]
        c = {
            "stem": stem,
            "expert_lm": lm,
            "machine_lm": auto[stem],
            "axes": ax,
            "cons_m": float(cons[0]),
            "cons_b": float(cons[1]),
            "width": 1856,
            "height": 2784,
            "yc": C.compute_yc(lm),
            "ipd": C.compute_ipd(lm),
            "split": split_of[stem],
            **vas[stem],
        }
        cohort.append(c)

    with open(C.OUT / "cohort.pkl", "wb") as f:
        pickle.dump(cohort, f)
    print(f"cohort={len(cohort)}  split={ {k: len(v) for k, v in split.items()} }")
    print("saved", C.OUT / "cohort.pkl")


if __name__ == "__main__":
    main()
