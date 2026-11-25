"""Image preprocessing, controlled degradations, and quality metrics.

This module operates on RGB uint8 arrays. It does not change semantic masks.
For segmentation training, apply spatial transforms identically to image and
mask; do not apply colour/noise degradations to the mask.
"""
from __future__ import annotations

from pathlib import Path
import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

from dataset_analysis import calculate_entropy
from image_visualization import plot_degradation_previews, plot_degradation_metrics

DEGRADATION_SEVERITIES = [1, 2, 3, 4, 5]
DEGRADATION_TYPES = [
    "gaussian_blur", "motion_blur", "defocus_blur",
    "gaussian_noise", "salt_pepper_noise", "speckle_noise",
    "jpeg_compression", "down_up_sampling",
    "brightness", "contrast", "gamma",
]
_RNG = np.random.default_rng(42)


def resize_image(image: np.ndarray, size: tuple[int, int],
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize RGB image to (width, height)."""
    return cv2.resize(image, size, interpolation=interpolation)


def normalize_image(image: np.ndarray,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Convert uint8 RGB image to float32 CHW tensor-like normalized array."""
    arr = image.astype(np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))


def _clip_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image, severity):
    k = [3, 5, 7, 9, 11][severity - 1]
    return cv2.GaussianBlur(image, (k, k), sigmaX=0)


def apply_motion_blur(image, severity):
    k = [3, 5, 7, 9, 13][severity - 1]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(image, -1, kernel)


def apply_defocus_blur(image, severity):
    radius = [1, 2, 3, 4, 5][severity - 1]
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    disk = (xx - radius) ** 2 + (yy - radius) ** 2 <= radius ** 2
    kernel = disk.astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_noise(image, severity):
    std = [5, 10, 15, 25, 35][severity - 1]
    return _clip_uint8(image.astype(np.float32) + _RNG.normal(0, std, image.shape))


def apply_salt_pepper_noise(image, severity):
    amount = [0.005, 0.01, 0.02, 0.04, 0.06][severity - 1]
    out = image.copy()
    h, w = image.shape[:2]
    n = int(amount * h * w)
    ys, xs = _RNG.integers(0, h, n), _RNG.integers(0, w, n)
    half = n // 2
    out[ys[:half], xs[:half]] = 0
    out[ys[half:], xs[half:]] = 255
    return out


def apply_speckle_noise(image, severity):
    std = [0.03, 0.06, 0.10, 0.15, 0.20][severity - 1]
    return _clip_uint8(image.astype(np.float32) *
                       (1.0 + _RNG.normal(0, std, image.shape)))


def apply_jpeg_compression(image, severity):
    quality = [80, 60, 40, 20, 10][severity - 1]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def apply_down_up_sampling(image, severity):
    scale = [0.75, 0.50, 0.35, 0.25, 0.15][severity - 1]
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_brightness(image, severity):
    offset = [-10, -20, -30, -40, -50][severity - 1]
    return _clip_uint8(image.astype(np.float32) + offset)


def apply_contrast(image, severity):
    factor = [0.90, 0.80, 0.70, 0.60, 0.50][severity - 1]
    mean = image.mean(axis=(0, 1), keepdims=True)
    return _clip_uint8((image.astype(np.float32) - mean) * factor + mean)


def apply_gamma(image, severity):
    gamma = [0.90, 0.80, 0.70, 0.60, 0.50][severity - 1]
    lut = np.asarray([((i / 255.0) ** gamma) * 255 for i in range(256)],
                     dtype=np.uint8)
    return cv2.LUT(image, lut)


DEGRADATION_FUNCTIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "defocus_blur": apply_defocus_blur,
    "gaussian_noise": apply_gaussian_noise,
    "salt_pepper_noise": apply_salt_pepper_noise,
    "speckle_noise": apply_speckle_noise,
    "jpeg_compression": apply_jpeg_compression,
    "down_up_sampling": apply_down_up_sampling,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "gamma": apply_gamma,
}


def calculate_psnr(reference: np.ndarray, distorted: np.ndarray) -> float:
    ref, dist = reference.astype(np.float32), distorted.astype(np.float32)
    mse = float(np.mean((ref - dist) ** 2))
    return float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    return float(structural_similarity(reference, distorted, channel_axis=2, data_range=255))


def calculate_degradation_metrics(reference: np.ndarray, distorted: np.ndarray) -> dict:
    gray = cv2.cvtColor(distorted, cv2.COLOR_RGB2GRAY)
    return {
        "psnr_db": calculate_psnr(reference, distorted),
        "ssim": calculate_ssim(reference, distorted),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": calculate_entropy(gray),
    }


def generate_degradation_dataset(
    rgb_zip,
    source_items: list[dict],
    output_dir: str | Path,
    max_images: int = 10,
    save_images: bool = True,
    jpeg_save_quality: int = 92,
):
    """Generate 11 x 5 variants per source image and export metrics/summaries."""
    output_dir = Path(output_dir)
    image_dir, csv_dir, graph_dir = (
        output_dir / "images", output_dir / "csv", output_dir / "graphs"
    )
    for folder in (image_dir, csv_dir, graph_dir):
        folder.mkdir(parents=True, exist_ok=True)

    sources = source_items[:min(max_images, len(source_items))]
    records, previews = [], {}

    for source_index, item in enumerate(sources, start=1):
        try:
            clean = np.asarray(
                Image.open(io.BytesIO(rgb_zip.read(item["image_path"]))).convert("RGB"),
                dtype=np.uint8,
            )
            source_name = Path(item["image_path"]).stem
            for degradation_name, function in DEGRADATION_FUNCTIONS.items():
                for severity in DEGRADATION_SEVERITIES:
                    degraded = function(clean.copy(), severity)
                    metrics = calculate_degradation_metrics(clean, degraded)
                    records.append({
                        "source_filename": Path(item["image_path"]).name,
                        "source_key": item.get("key", source_name),
                        "split": item.get("split", "unknown"),
                        "degradation": degradation_name,
                        "severity": severity,
                        "width": int(degraded.shape[1]),
                        "height": int(degraded.shape[0]),
                        **metrics,
                    })
                    if severity == 3 and degradation_name not in previews:
                        previews[degradation_name] = {
                            "clean": clean.copy(), "degraded": degraded.copy(),
                            "severity": severity, "filename": source_name,
                        }
                    if save_images:
                        folder = image_dir / degradation_name / f"severity_{severity}"
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / f"{source_index:03d}_{source_name}.jpg"
                        bgr = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(path), bgr,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_save_quality])
        except Exception as exc:
            print(f"Could not process {item.get('image_path', 'unknown')}: {exc}")

    degradation_df = pd.DataFrame(records)
    if degradation_df.empty:
        raise ValueError("No degraded images were generated.")

    severity_summary = (
        degradation_df.groupby(["degradation", "severity"])
        .agg(
            image_count=("source_filename", "count"),
            mean_psnr_db=("psnr_db", "mean"),
            mean_ssim=("ssim", "mean"),
            mean_brightness=("brightness", "mean"),
            mean_contrast=("contrast", "mean"),
            mean_laplacian_variance=("laplacian_variance", "mean"),
            mean_entropy=("entropy", "mean"),
        ).reset_index()
    )
    degradation_df.to_csv(csv_dir / "ACDC_Degradation_Quality_Metrics.csv", index=False)
    severity_summary.to_csv(csv_dir / "ACDC_Degradation_Severity_Summary.csv", index=False)

    family_map = {
        "gaussian_blur": "Blur", "motion_blur": "Blur", "defocus_blur": "Blur",
        "gaussian_noise": "Noise", "salt_pepper_noise": "Noise", "speckle_noise": "Noise",
        "jpeg_compression": "Compression", "down_up_sampling": "Resolution",
        "brightness": "Lighting", "contrast": "Lighting", "gamma": "Lighting",
    }
    degradation_df["degradation_family"] = degradation_df["degradation"].map(family_map)
    family_summary = (
        degradation_df.groupby(["degradation_family", "degradation", "severity"])
        [["psnr_db", "ssim", "brightness", "contrast", "laplacian_variance", "entropy"]]
        .mean().reset_index()
    )
    family_summary.to_csv(csv_dir / "Degradation_Family_Summary.csv", index=False)

    plot_degradation_previews(previews, graph_dir)
    plot_degradation_metrics(severity_summary, graph_dir)

    report = [
        "CONTROLLED IMAGE DEGRADATION EDA REPORT",
        "=" * 55,
        f"Source images used: {len(sources)}",
        f"Degradation types: {len(DEGRADATION_FUNCTIONS)}",
        f"Severity levels: {DEGRADATION_SEVERITIES}",
        f"Total degraded variants measured: {len(degradation_df)}",
        f"Degraded images saved: {save_images}",
        "",
        severity_summary.to_string(index=False),
    ]
    (output_dir / "Controlled_Degradation_EDA_Report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return degradation_df, severity_summary, family_summary
