"""Axis feature construction and the four regression models used in the paper.

The feature vector follows the project's 167-dimensional coords_geom design:
  58 normalized coordinates + 88 bilateral-pair geometry + 21 midline geometry.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

EPS = 1e-6

POINT_NAMES_31 = [
    "ebe_L", "ebe_R", "ebs_L", "ebs_R", "os_L", "os_R", "sci_L", "sci_R",
    "en_L", "en_R", "p_L", "p_R", "ex_L", "ex_R", "Na", "prn", "Sn",
    "al_L", "al_R", "cph_L", "cph_R", "ch_L", "ch_R", "ls", "sto", "li",
    "Go_L", "Go_R", "Me",
]
SYMMETRY_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11),
                  (12, 13), (17, 18), (19, 20), (21, 22), (26, 27)]
MIDLINE_INDICES = [14, 15, 16, 23, 24, 25, 28]


def compute_sample_normalization(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    center = landmarks.mean(axis=0).astype(np.float32)
    spans = landmarks.max(axis=0) - landmarks.min(axis=0)
    scale = float(max(float(np.hypot(spans[0], spans[1])), 1.0))
    return center, scale


def normalize_landmarks(landmarks: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return ((landmarks - center) / scale).astype(np.float32)


def build_axis_feature_vector(landmarks: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    norm = normalize_landmarks(landmarks, center, scale)
    coord = norm.reshape(-1)

    geom = []
    for l_idx, r_idx in SYMMETRY_PAIRS:
        left = norm[l_idx]
        right = norm[r_idx]
        midpoint = (left + right) / 2.0
        diff = right - left
        width = float(abs(diff[0]))
        height = float(abs(diff[1]))
        pair_distance = float(np.linalg.norm(diff))
        slope = float(diff[1] / (diff[0] + EPS))
        geom.extend([
            float(midpoint[0]), float(midpoint[1]),
            float(diff[0]), float(diff[1]),
            width, height, pair_distance, slope,
        ])
    for idx in MIDLINE_INDICES:
        point = norm[idx]
        geom.extend([float(point[0]), float(point[1]), float(np.linalg.norm(point))])
    return np.concatenate([coord, np.asarray(geom, dtype=np.float32)]).astype(np.float32)


def make_feature_names() -> List[str]:
    names = []
    for point_name in POINT_NAMES_31:
        names.append(f"{point_name}_x")
        names.append(f"{point_name}_y")
    return names


def make_axis_feature_names() -> List[str]:
    names = make_feature_names()
    for l_idx, r_idx in SYMMETRY_PAIRS:
        pair = f"{POINT_NAMES_31[l_idx]}_{POINT_NAMES_31[r_idx]}"
        names.extend([f"{pair}_mid_x", f"{pair}_mid_y", f"{pair}_dx", f"{pair}_dy",
                      f"{pair}_width", f"{pair}_height", f"{pair}_dist", f"{pair}_slope"])
    for idx in MIDLINE_INDICES:
        point = POINT_NAMES_31[idx]
        names.extend([f"{point}_x_norm", f"{point}_y_norm", f"{point}_radius_norm"])
    return names


def target_to_image_line(target: np.ndarray, center: np.ndarray, scale: float) -> Tuple[float, float]:
    slope = float(target[0])
    intercept = float(center[0] - slope * center[1] + scale * float(target[1]))
    return slope, intercept


class LinearAxisRegressor:
    """Ridge regression on standardized features."""

    def __init__(self, alpha: float = 1000.0, solver: str = "svd") -> None:
        self.alpha = float(alpha)
        self.solver = str(solver)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, solver=self.solver)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = self.scaler.fit_transform(np.asarray(features, np.float64))
        self.model.fit(x, np.asarray(targets, np.float64))

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(np.asarray(features, np.float64))
        return self.model.predict(x).astype(np.float32)


class OrdinaryLinearAxisRegressor:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = self.scaler.fit_transform(np.asarray(features, np.float64))
        self.model = LinearRegression()
        self.model.fit(x, np.asarray(targets, np.float64))

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(np.asarray(features, np.float64))
        return self.model.predict(x).astype(np.float32)


class SVRAxisRegressor:
    def __init__(self, C: float = 10.0, epsilon: float = 0.01, gamma: str = "scale") -> None:
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = self.scaler.fit_transform(np.asarray(features, np.float64))
        y = np.asarray(targets, np.float64)
        self.model = [SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma).fit(x, y[:, i])
                      for i in range(y.shape[1])]

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(np.asarray(features, np.float64))
        return np.column_stack([m.predict(x) for m in self.model]).astype(np.float32)


class RandomForestAxisRegressor:
    def __init__(self, n_estimators: int = 500, min_samples_leaf: int = 2,
                 max_depth=None, random_state: int = 42) -> None:
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = self.scaler.fit_transform(np.asarray(features, np.float64))
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth, random_state=self.random_state, n_jobs=-1,
        )
        self.model.fit(x, np.asarray(targets, np.float64))

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(np.asarray(features, np.float64))
        return self.model.predict(x).astype(np.float32)
