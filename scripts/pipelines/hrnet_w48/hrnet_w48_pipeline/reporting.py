from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from src.data.dataset import normalize_image_tensor
from src.landmark.schema import POINT_NAMES_31, REGIONS

from .metrics import compute_pixel_metrics, decode_heatmaps_local_softargmax


REGION_DISPLAY_NAMES = {
    "jaw": "facial contour",
}


def save_ced_curve(all_nme, save_path: Path) -> None:
    if not all_nme:
        return
    nme = np.sort(np.asarray(all_nme, dtype=np.float32))
    y = np.arange(1, len(nme) + 1) / len(nme)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nme, y, color="#1f77b4", linewidth=2)
    ax.set_xlabel("NME")
    ax.set_ylabel("Cumulative sample ratio")
    ax.set_title("CED Curve")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(0.1, float(nme.max()) * 1.05))
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_nme_histogram(all_nme, save_path: Path) -> None:
    if not all_nme:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.asarray(all_nme, dtype=np.float32), bins=15, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("NME")
    ax.set_ylabel("Sample count")
    ax.set_title("Test-set NME Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_point_error_plot(per_point_errors, save_path: Path) -> None:
    means = []
    stds = []
    labels = []
    for name, errors in zip(POINT_NAMES_31, per_point_errors):
        means.append(float(np.mean(errors)) if errors else 0.0)
        stds.append(float(np.std(errors)) if errors else 0.0)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color="#72B7B2", ecolor="#1B4332", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Mean error (pixels)")
    ax.set_title("Per-Landmark Error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_region_error_plot(per_point_errors, save_path: Path) -> None:
    region_names = []
    region_means = []
    region_stds = []
    for region, indices in REGIONS.items():
        region_values = []
        for idx in indices:
            region_values.extend(per_point_errors[idx])
        if not region_values:
            continue
        region_names.append(REGION_DISPLAY_NAMES.get(region, region))
        region_means.append(float(np.mean(region_values)))
        region_stds.append(float(np.std(region_values)))

    if not region_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(region_names))
    ax.bar(x, region_means, yerr=region_stds, color="#F58518", ecolor="#7F2704", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(region_names)
    ax.set_ylabel("Mean error (pixels)")
    ax.set_title("Regional Landmark Error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_summary(all_nme, per_point_errors, sample_records) -> dict:
    pixel_metrics = compute_pixel_metrics(per_point_errors)
    region_stats = {}
    for region, indices in REGIONS.items():
        region_values = []
        for idx in indices:
            region_values.extend(per_point_errors[idx])
        if region_values:
            region_stats[region] = {
                "mean_error_px": float(np.mean(region_values)),
                "std_error_px": float(np.std(region_values)),
                "count": len(region_values),
            }

    per_point_stats = {}
    for name, errors in zip(POINT_NAMES_31, per_point_errors):
        per_point_stats[name] = {
            "mean_error_px": float(np.mean(errors)) if errors else None,
            "std_error_px": float(np.std(errors)) if errors else None,
            "count": len(errors),
        }

    valid_nme = np.asarray(all_nme, dtype=np.float32)
    sorted_samples = sorted(sample_records, key=lambda item: item["nme"])
    return {
        "mean_nme": float(np.mean(valid_nme)) if len(valid_nme) else None,
        "median_nme": float(np.median(valid_nme)) if len(valid_nme) else None,
        "std_nme": float(np.std(valid_nme)) if len(valid_nme) else None,
        "dr_0.05": float(np.mean(valid_nme < 0.05)) if len(valid_nme) else None,
        "dr_0.10": float(np.mean(valid_nme < 0.10)) if len(valid_nme) else None,
        "mre_px": pixel_metrics["mre_px"] if pixel_metrics else None,
        "std_mre_px": pixel_metrics["std_px"] if pixel_metrics else None,
        "sdr_px": pixel_metrics["sdr_px"] if pixel_metrics else None,
        "n_points": pixel_metrics["n_points"] if pixel_metrics else None,
        "n_samples": len(valid_nme),
        "per_point_error_px": per_point_stats,
        "per_region_error_px": region_stats,
        "best_samples": sorted_samples[:10],
        "worst_samples": sorted_samples[-10:],
    }


def save_error_details(sample_records, save_path: Path) -> None:
    payload = {
        "samples": sample_records,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def find_event_file(log_dir: Path) -> Path:
    event_files = sorted(log_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found under {log_dir}")
    return event_files[-1]


def load_scalar_history(event_file: Path) -> dict[str, list[dict[str, float]]]:
    acc = event_accumulator.EventAccumulator(str(event_file))
    acc.Reload()
    history = {}
    for tag in ["loss/train", "loss/val", "nme/train", "nme/val", "lr", "loss/train_heatmap", "loss/train_coord", "loss/val_heatmap", "loss/val_coord"]:
        if tag not in acc.Tags().get("scalars", []):
            continue
        history[tag] = [{"step": int(item.step), "value": float(item.value)} for item in acc.Scalars(tag)]
    return history


def _robust_display_series(
    series: list[dict[str, float]],
    quantile: float = 0.85,
    median_window: int = 5,
    ema_alpha: float = 0.35,
) -> list[dict[str, float]]:
    if not series:
        return []

    values = np.asarray([item["value"] for item in series], dtype=np.float64)
    clip_max = float(np.quantile(values, quantile))
    clipped = np.minimum(values, clip_max)

    radius = max(1, median_window // 2)
    median_smoothed = clipped.copy()
    for i in range(len(clipped)):
        lo = max(0, i - radius)
        hi = min(len(clipped), i + radius + 1)
        median_smoothed[i] = float(np.median(clipped[lo:hi]))

    ema_smoothed = median_smoothed.copy()
    for i in range(1, len(ema_smoothed)):
        ema_smoothed[i] = ema_alpha * median_smoothed[i] + (1.0 - ema_alpha) * ema_smoothed[i - 1]

    return [
        {"step": int(item["step"]), "value": float(value)}
        for item, value in zip(series, ema_smoothed)
    ]


def save_training_curves(history: dict[str, list[dict[str, float]]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    train_loss_display = history.get("loss/train_display")
    if train_loss_display is None and "loss/train" in history:
        train_loss_display = _robust_display_series(history["loss/train"], quantile=1.0, median_window=5, ema_alpha=0.30)

    val_loss_display = history.get("loss/val_display")
    if val_loss_display is None and "loss/val" in history:
        val_loss_display = _robust_display_series(history["loss/val"], quantile=0.85, median_window=5, ema_alpha=0.35)

    if train_loss_display:
        x = [item["step"] for item in train_loss_display]
        y = [item["value"] for item in train_loss_display]
        axes[0].plot(x, y, label="Train Loss", color="#1f77b4", linewidth=2)
    else:
        for tag, label, color in [("loss/train", "Train Loss", "#1f77b4")]:
            if tag in history:
                x = [item["step"] for item in history[tag]]
                y = [item["value"] for item in history[tag]]
                axes[0].plot(x, y, label=label, color=color, linewidth=2)
    if val_loss_display:
        x = [item["step"] for item in val_loss_display]
        y = [item["value"] for item in val_loss_display]
        axes[0].plot(x, y, label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for tag, label, color in [("nme/train", "Train NME", "#2ca02c"), ("nme/val", "Val NME", "#d62728")]:
        if tag in history:
            x = [item["step"] for item in history[tag]]
            y = [item["value"] for item in history[tag]]
            axes[1].plot(x, y, label=label, color=color, linewidth=2)
    axes[1].set_title("Training NME Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NME")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def predict_landmarks(model, image_rgb: np.ndarray, device: torch.device, input_size: tuple[int, int]) -> np.ndarray:
    orig_h, orig_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (input_size[1], input_size[0]))
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    image_tensor = normalize_image_tensor(image_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmaps = model(image_tensor)
        coords_norm, _ = decode_heatmaps_local_softargmax(pred_heatmaps)

    pred = coords_norm[0].cpu().numpy().copy()
    pred[:, 0] *= orig_w
    pred[:, 1] *= orig_h
    return pred


def _safe_stem(file_name: str) -> str:
    return Path(file_name).as_posix().replace("/", "__")


def draw_overlay(image_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.imshow(image_rgb)
    ax.set_title(title)
    ax.axis("off")
    ax.scatter(gt[valid, 0], gt[valid, 1], s=30, facecolors="none", edgecolors="#00FF66", linewidths=1.2, label="GT")
    ax.scatter(pred[valid, 0], pred[valid, 1], c="#FF3B30", s=18, marker="x", linewidths=1.0, label="Prediction")
    ax.legend(loc="lower right")
    return fig


def render_ranked_samples(ranked_samples, annotations_by_image_id, images_by_id, img_root: Path, model, device: torch.device, output_dir: Path, input_size: tuple[int, int]):
    rendered = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, item in enumerate(ranked_samples, start=1):
        image_id = int(item["image_id"])
        ann = annotations_by_image_id[image_id]
        img_info = images_by_id[image_id]
        img_path = img_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        gt = kp[:, :2]
        valid = kp[:, 2] > 0
        pred = predict_landmarks(model, image_rgb, device, input_size=input_size)

        fig = draw_overlay(image_rgb, gt, pred, valid, title=f"#{rank} | NME={item['nme']:.4f}")
        save_path = output_dir / f"{rank:02d}_{_safe_stem(img_info['file_name'])}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rendered.append({"rank": rank, "nme": float(item["nme"]), "file_name": img_info["file_name"], "path": str(save_path)})
    return rendered


def save_sample_grid(rendered_samples, save_path: Path, title: str, max_items: int = 5):
    picked = rendered_samples[:max_items]
    if not picked:
        return
    fig, axes = plt.subplots(1, len(picked), figsize=(4 * len(picked), 5))
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, picked):
        image = cv2.imread(item["path"])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(f"#{item['rank']}\nNME={item['nme']:.4f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
