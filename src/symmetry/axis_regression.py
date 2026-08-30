from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from src.landmark.schema import MIDLINE_INDICES, POINT_NAMES_31, SYMMETRY_PAIRS


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EPS = 1e-6


@dataclass
class AxisAnnotation:
    image_path: str
    image_width: int
    image_height: int
    axis_points: np.ndarray
    json_path: str


def _match_existing_path(candidate: Path) -> Path | None:
    parent = candidate.parent
    if not parent.exists():
        return None

    target_name = candidate.name.casefold()
    for child in parent.iterdir():
        if child.name.casefold() == target_name:
            return child
    return None


def _resolve_image_path(json_path: Path, image_path: str) -> Path:
    candidates: list[Path] = []
    if image_path:
        image_path_obj = Path(image_path)
        if image_path_obj.is_absolute():
            candidates.append(image_path_obj)
        candidates.append(json_path.parent / image_path_obj)

    for ext in IMAGE_EXTENSIONS:
        candidates.append(json_path.with_suffix(ext))
        candidates.append(json_path.with_suffix(ext.upper()))

    for candidate in candidates:
        matched = _match_existing_path(candidate)
        if matched is not None:
            return matched

    if image_path:
        fallback = json_path.parent / image_path
        return fallback
    return json_path.with_suffix(".jpg")


def parse_axis_labelme_json(json_path: str) -> AxisAnnotation | None:
    json_path_obj = Path(json_path)
    with open(json_path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    axis_points = None
    for shape in data.get("shapes", []):
        if shape.get("label") != "axis" or shape.get("shape_type") != "line":
            continue
        points = np.array(shape.get("points", []), dtype=np.float32)
        if points.shape == (2, 2):
            axis_points = points
            break

    if axis_points is None:
        return None

    image_path = _resolve_image_path(json_path_obj, data.get("imagePath", ""))
    return AxisAnnotation(
        image_path=str(image_path),
        image_width=int(data.get("imageWidth", 0)),
        image_height=int(data.get("imageHeight", 0)),
        axis_points=axis_points,
        json_path=str(json_path_obj),
    )


def make_feature_names() -> list[str]:
    names = []
    for point_name in POINT_NAMES_31:
        names.append(f"{point_name}_x")
        names.append(f"{point_name}_y")
    return names


def build_axis_feature_vector(landmarks: np.ndarray, center: np.ndarray, scale: float, feature_set: str = "coords") -> np.ndarray:
    norm_landmarks = normalize_landmarks(landmarks, center, scale)
    coord_features = norm_landmarks.reshape(-1)
    if feature_set == "coords":
        return coord_features.astype(np.float32)

    geometry_features = []
    for l_idx, r_idx in SYMMETRY_PAIRS:
        left = norm_landmarks[l_idx]
        right = norm_landmarks[r_idx]
        midpoint = (left + right) / 2.0
        diff = right - left
        width = float(abs(diff[0]))
        height = float(abs(diff[1]))
        pair_distance = float(np.linalg.norm(diff))
        slope = float(diff[1] / (diff[0] + EPS))
        geometry_features.extend([
            float(midpoint[0]),
            float(midpoint[1]),
            float(diff[0]),
            float(diff[1]),
            width,
            height,
            pair_distance,
            slope,
        ])

    for idx in MIDLINE_INDICES:
        point = norm_landmarks[idx]
        geometry_features.extend([
            float(point[0]),
            float(point[1]),
            float(np.linalg.norm(point)),
        ])

    return np.concatenate([coord_features, np.asarray(geometry_features, dtype=np.float32)], axis=0).astype(np.float32)


def make_axis_feature_names(feature_set: str = "coords") -> list[str]:
    coord_names = make_feature_names()
    if feature_set == "coords":
        return coord_names

    geometry_names = []
    for l_idx, r_idx in SYMMETRY_PAIRS:
        pair_name = f"{POINT_NAMES_31[l_idx]}_{POINT_NAMES_31[r_idx]}"
        geometry_names.extend(
            [
                f"{pair_name}_mid_x",
                f"{pair_name}_mid_y",
                f"{pair_name}_dx",
                f"{pair_name}_dy",
                f"{pair_name}_width",
                f"{pair_name}_height",
                f"{pair_name}_dist",
                f"{pair_name}_slope",
            ]
        )

    for idx in MIDLINE_INDICES:
        point_name = POINT_NAMES_31[idx]
        geometry_names.extend(
            [
                f"{point_name}_x_norm",
                f"{point_name}_y_norm",
                f"{point_name}_radius_norm",
            ]
        )

    return coord_names + geometry_names


def sort_axis_points_by_y(axis_points: np.ndarray) -> np.ndarray:
    axis_points = np.asarray(axis_points, dtype=np.float32)
    order = np.lexsort((axis_points[:, 0], axis_points[:, 1]))
    return axis_points[order].astype(np.float32)


def compute_sample_normalization(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    center = landmarks.mean(axis=0).astype(np.float32)
    spans = landmarks.max(axis=0) - landmarks.min(axis=0)
    scale = float(max(float(np.hypot(spans[0], spans[1])), 1.0))
    return center, scale


def normalize_landmarks(landmarks: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return ((landmarks - center) / scale).astype(np.float32)


def axis_points_to_target(axis_points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    pts = normalize_landmarks(sort_axis_points_by_y(axis_points), center, scale)
    dy = float(pts[1, 1] - pts[0, 1])
    if abs(dy) < EPS:
        dy = EPS if dy >= 0 else -EPS
    slope = float((pts[1, 0] - pts[0, 0]) / dy)
    intercept = float(pts[0, 0] - slope * pts[0, 1])
    return np.array([slope, intercept], dtype=np.float32)


def target_to_image_line(target: np.ndarray, center: np.ndarray, scale: float) -> tuple[float, float]:
    slope = float(target[0])
    intercept = float(center[0] - slope * center[1] + scale * float(target[1]))
    return slope, intercept


def axis_points_to_image_line(axis_points: np.ndarray) -> tuple[float, float]:
    axis_points = sort_axis_points_by_y(axis_points)
    dy = float(axis_points[1, 1] - axis_points[0, 1])
    if abs(dy) < EPS:
        dy = EPS if dy >= 0 else -EPS
    slope = float((axis_points[1, 0] - axis_points[0, 0]) / dy)
    intercept = float(axis_points[0, 0] - slope * axis_points[0, 1])
    return slope, intercept


def point_to_axis_distance(points: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    denom = np.sqrt(1.0 + slope ** 2)
    return np.abs(points[:, 0] - slope * points[:, 1] - intercept) / denom


def endpoint_x_mae(axis_points: np.ndarray, slope: float, intercept: float) -> float:
    sorted_points = sort_axis_points_by_y(axis_points)
    pred_x = slope * sorted_points[:, 1] + intercept
    return float(np.mean(np.abs(pred_x - sorted_points[:, 0])))


def endpoint_x_mae_normalized(axis_points: np.ndarray, target: np.ndarray, center: np.ndarray, scale: float) -> float:
    sorted_points = normalize_landmarks(sort_axis_points_by_y(axis_points), center, scale)
    pred_x = float(target[0]) * sorted_points[:, 1] + float(target[1])
    return float(np.mean(np.abs(pred_x - sorted_points[:, 0])))


def midpoint_x_error(axis_points: np.ndarray, slope: float, intercept: float) -> float:
    y_mid = float(axis_points[:, 1].mean())
    gt_x = float(axis_points[:, 0].mean())
    pred_x = float(slope * y_mid + intercept)
    return abs(pred_x - gt_x)


def midpoint_x_error_normalized(axis_points: np.ndarray, target: np.ndarray, center: np.ndarray, scale: float) -> float:
    sorted_points = normalize_landmarks(sort_axis_points_by_y(axis_points), center, scale)
    y_mid = float(sorted_points[:, 1].mean())
    gt_x = float(sorted_points[:, 0].mean())
    pred_x = float(float(target[0]) * y_mid + float(target[1]))
    return abs(pred_x - gt_x)


def angle_error_deg(pred_slope: float, gt_slope: float) -> float:
    pred_angle = np.degrees(np.arctan(pred_slope))
    gt_angle = np.degrees(np.arctan(gt_slope))
    return float(abs(pred_angle - gt_angle))


def summarize_axis_predictions(
    pred_targets: np.ndarray,
    gt_targets: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    axis_points: np.ndarray,
    file_names: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    endpoint_errors = []
    endpoint_errors_norm = []
    midpoint_errors = []
    midpoint_errors_norm = []
    angle_errors = []
    records = []

    for idx in range(len(pred_targets)):
        pred_slope, pred_intercept = target_to_image_line(pred_targets[idx], centers[idx], float(scales[idx]))
        gt_slope, _ = target_to_image_line(gt_targets[idx], centers[idx], float(scales[idx]))
        gt_line_slope, gt_line_intercept = axis_points_to_image_line(axis_points[idx])
        endpoint_error = endpoint_x_mae(axis_points[idx], pred_slope, pred_intercept)
        endpoint_error_norm = endpoint_x_mae_normalized(axis_points[idx], pred_targets[idx], centers[idx], float(scales[idx]))
        midpoint_error = midpoint_x_error(axis_points[idx], pred_slope, pred_intercept)
        midpoint_error_norm = midpoint_x_error_normalized(axis_points[idx], pred_targets[idx], centers[idx], float(scales[idx]))
        angle_error = angle_error_deg(pred_slope, gt_line_slope)

        endpoint_errors.append(endpoint_error)
        endpoint_errors_norm.append(endpoint_error_norm)
        midpoint_errors.append(midpoint_error)
        midpoint_errors_norm.append(midpoint_error_norm)
        angle_errors.append(angle_error)
        records.append(
            {
                "file_name": file_names[idx] if file_names is not None else str(idx),
                "endpoint_mae_px": endpoint_error,
                "endpoint_mae_norm": endpoint_error_norm,
                "midpoint_x_error_px": midpoint_error,
                "midpoint_x_error_norm": midpoint_error_norm,
                "angle_error_deg": angle_error,
                "pred_slope_xy": pred_slope,
                "pred_intercept_x": pred_intercept,
                "gt_slope_xy": gt_line_slope,
                "gt_intercept_x": gt_line_intercept,
            }
        )

    target_r2 = None
    if len(pred_targets) >= 2:
        try:
            target_r2 = float(r2_score(gt_targets, pred_targets, multioutput="uniform_average"))
        except ValueError:
            target_r2 = None

    summary = {
        "n_samples": len(pred_targets),
        "endpoint_mae_px": float(np.mean(endpoint_errors)) if endpoint_errors else None,
        "endpoint_median_px": float(np.median(endpoint_errors)) if endpoint_errors else None,
        "endpoint_mae_norm": float(np.mean(endpoint_errors_norm)) if endpoint_errors_norm else None,
        "endpoint_median_norm": float(np.median(endpoint_errors_norm)) if endpoint_errors_norm else None,
        "midpoint_x_mae_px": float(np.mean(midpoint_errors)) if midpoint_errors else None,
        "midpoint_x_mae_norm": float(np.mean(midpoint_errors_norm)) if midpoint_errors_norm else None,
        "angle_mae_deg": float(np.mean(angle_errors)) if angle_errors else None,
        "angle_median_deg": float(np.median(angle_errors)) if angle_errors else None,
        "target_r2": target_r2,
    }
    return summary, records


class LinearAxisRegressor:
    def __init__(self, alpha: float = 1.0, solver: str = "svd") -> None:
        self.alpha = float(alpha)
        self.solver = str(solver)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, solver=self.solver)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features64 = np.asarray(features, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)
        features_scaled = self.scaler.fit_transform(features64)
        self.model.fit(features_scaled, targets64)

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        features64 = np.asarray(features, dtype=np.float64)
        features_scaled = self.scaler.transform(features64)
        return self.model.predict(features_scaled).astype(np.float32)

    def export_coefficients(self, feature_names: list[str]) -> dict:
        coef = np.asarray(self.model.coef_, dtype=np.float64)
        intercept = np.asarray(self.model.intercept_, dtype=np.float64)
        outputs = []
        output_names = ["slope_xy", "intercept_norm"]

        for output_idx, output_name in enumerate(output_names):
            outputs.append(
                {
                    "name": output_name,
                    "intercept": float(intercept[output_idx]),
                    "coefficients": {
                        feature_names[i]: float(coef[output_idx, i])
                        for i in range(len(feature_names))
                    },
                }
            )

        return {
            "alpha": self.alpha,
            "solver": self.solver,
            "feature_mean": self.scaler.mean_.tolist(),
            "feature_scale": self.scaler.scale_.tolist(),
            "outputs": outputs,
        }

    def save(self, path: str | Path) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "LinearAxisRegressor":
        return load(path)


class PolynomialAxisRegressor:
    def __init__(self, alpha: float = 1.0, degree: int = 2, solver: str = "svd") -> None:
        self.alpha = float(alpha)
        self.degree = int(degree)
        self.solver = str(solver)
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.model = Ridge(alpha=self.alpha, solver=self.solver)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features64 = np.asarray(features, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)
        features_scaled = self.scaler.fit_transform(features64)
        features_poly = self.poly.fit_transform(features_scaled)
        self.model.fit(features_poly, targets64)

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        features64 = np.asarray(features, dtype=np.float64)
        features_scaled = self.scaler.transform(features64)
        features_poly = self.poly.transform(features_scaled)
        return self.model.predict(features_poly).astype(np.float32)

    def export_coefficients(self, feature_names: list[str]) -> dict:
        transformed_names = self.poly.get_feature_names_out(feature_names).tolist()
        coef = np.asarray(self.model.coef_, dtype=np.float64)
        intercept = np.asarray(self.model.intercept_, dtype=np.float64)
        outputs = []
        output_names = ["slope_xy", "intercept_norm"]

        for output_idx, output_name in enumerate(output_names):
            outputs.append(
                {
                    "name": output_name,
                    "intercept": float(intercept[output_idx]),
                    "coefficients": {
                        transformed_names[i]: float(coef[output_idx, i])
                        for i in range(len(transformed_names))
                    },
                }
            )

        return {
            "model_type": "poly_ridge",
            "alpha": self.alpha,
            "degree": self.degree,
            "solver": self.solver,
            "feature_mean": self.scaler.mean_.tolist(),
            "feature_scale": self.scaler.scale_.tolist(),
            "transformed_feature_names": transformed_names,
            "outputs": outputs,
        }

    def save(self, path: str | Path) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "PolynomialAxisRegressor":
        return load(path)


class PLSAxisRegressor:
    def __init__(self, n_components: int = 8) -> None:
        self.n_components = int(n_components)
        self.scaler = StandardScaler()
        self.model = PLSRegression(n_components=self.n_components, scale=False, max_iter=1000)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features64 = np.asarray(features, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)
        features_scaled = self.scaler.fit_transform(features64)
        self.model.fit(features_scaled, targets64)

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        features64 = np.asarray(features, dtype=np.float64)
        features_scaled = self.scaler.transform(features64)
        return np.asarray(self.model.predict(features_scaled), dtype=np.float32)

    def export_coefficients(self, feature_names: list[str]) -> dict:
        metadata = {
            "model_type": "pls",
            "n_components": self.n_components,
            "feature_mean": self.scaler.mean_.tolist(),
            "feature_scale": self.scaler.scale_.tolist(),
            "feature_names": feature_names,
        }

        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_, dtype=np.float64)
            if coef.shape[0] != len(feature_names):
                coef = coef.T
            outputs = []
            output_names = ["slope_xy", "intercept_norm"]
            intercept = np.asarray(getattr(self.model, "intercept_", np.zeros(2, dtype=np.float64)), dtype=np.float64)
            for output_idx, output_name in enumerate(output_names):
                outputs.append(
                    {
                        "name": output_name,
                        "intercept": float(intercept[output_idx]) if intercept.ndim > 0 else float(intercept),
                        "coefficients": {
                            feature_names[i]: float(coef[i, output_idx])
                            for i in range(len(feature_names))
                        },
                    }
                )
            metadata["outputs"] = outputs

        return metadata

    def save(self, path: str | Path) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "PLSAxisRegressor":
        return load(path)


class MLPAxisRegressor:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        alpha: float = 1.0e-3,
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = tuple(int(v) for v in hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="lbfgs",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features64 = np.asarray(features, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)
        features_scaled = self.scaler.fit_transform(features64)
        targets_scaled = self.target_scaler.fit_transform(targets64)
        self.model.fit(features_scaled, targets_scaled)

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        features64 = np.asarray(features, dtype=np.float64)
        features_scaled = self.scaler.transform(features64)
        pred_scaled = np.asarray(self.model.predict(features_scaled), dtype=np.float64)
        return self.target_scaler.inverse_transform(pred_scaled).astype(np.float32)

    def export_coefficients(self, feature_names: list[str]) -> dict:
        return {
            "model_type": "mlp",
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "feature_names": feature_names,
            "feature_mean": self.scaler.mean_.tolist(),
            "feature_scale": self.scaler.scale_.tolist(),
            "target_mean": self.target_scaler.mean_.tolist(),
            "target_scale": self.target_scaler.scale_.tolist(),
            "n_iter": int(getattr(self.model, "n_iter_", 0)),
            "loss": float(getattr(self.model, "loss_", 0.0)),
        }

    def save(self, path: str | Path) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MLPAxisRegressor":
        return load(path)


class OrdinaryLinearAxisRegressor:
    """Unregularized linear regression baseline for axis prediction."""

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        from sklearn.linear_model import LinearRegression
        features64 = np.asarray(features, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)
        features_scaled = self.scaler.fit_transform(features64)
        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(features_scaled, targets64)

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        features64 = np.asarray(features, dtype=np.float64)
        features_scaled = self.scaler.transform(features64)
        return self.model.predict(features_scaled).astype(np.float32)

    def export_coefficients(self, feature_names: list[str]) -> dict:
        coef = np.asarray(self.model.coef_, dtype=np.float64)
        intercept = np.asarray(self.model.intercept_, dtype=np.float64)
        outputs = []
        for output_idx, output_name in enumerate(["slope_xy", "intercept_norm"]):
            outputs.append(
                {
                    "name": output_name,
                    "intercept": float(intercept[output_idx]),
                    "coefficients": {
                        feature_names[i]: float(coef[output_idx, i])
                        for i in range(len(feature_names))
                    },
                }
            )
        return {
            "model_type": "linear",
            "feature_mean": self.scaler.mean_.tolist(),
            "feature_scale": self.scaler.scale_.tolist(),
            "outputs": outputs,
        }

    def save(self, path: str | Path) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "OrdinaryLinearAxisRegressor":
        return load(path)


class SVRAxisRegressor:
    """SVR with RBF kernel for axis prediction (one regressor per output)."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.01, gamma: str = "scale") -> None:
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features64 = np.asarray(features, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)
        features_scaled = self.scaler.fit_transform(features64)
        self.model = [
            SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma).fit(features_scaled, targets64[:, i])
            for i in range(targets64.shape[1])
        ]

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        features64 = np.asarray(features, dtype=np.float64)
        features_scaled = self.scaler.transform(features64)
        pred = np.column_stack([svr.predict(features_scaled) for svr in self.model])
        return pred.astype(np.float32)

    def export_coefficients(self, feature_names: list[str]) -> dict:
        return {
            "model_type": "svr",
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "n_support_vectors": [int(svr.n_support_) for svr in self.model],
            "feature_names": feature_names,
        }

    def save(self, path: str | Path) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "SVRAxisRegressor":
        return load(path)


class RandomForestAxisRegressor:
    """Random Forest regression baseline for axis prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features64 = np.asarray(features, dtype=np.float64)
        targets64 = np.asarray(targets, dtype=np.float64)
        features_scaled = self.scaler.fit_transform(features64)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(features_scaled, targets64)

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        features64 = np.asarray(features, dtype=np.float64)
        features_scaled = self.scaler.transform(features64)
        return self.model.predict(features_scaled).astype(np.float32)

    def export_coefficients(self, feature_names: list[str]) -> dict:
        importances = np.asarray(self.model.feature_importances_, dtype=np.float64)
        return {
            "model_type": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_importances": {
                feature_names[i]: float(importances[i])
                for i in range(len(feature_names))
            },
            "feature_names": feature_names,
        }

    def save(self, path: str | Path) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RandomForestAxisRegressor":
        return load(path)
