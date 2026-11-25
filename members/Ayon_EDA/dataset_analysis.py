"""Image-quality EDA for ACDC fog and Cityscapes-C."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from data_loading import (
    ACDC_RGB_URL, ACDC_GT_URL, MAX_ACDC_IMAGES, MAX_CITYC_IMAGES,
    MAX_PER_GROUP, connect_acdc_archives, list_acdc_fog_files,
    match_acdc_images, sample_acdc_images, iter_cityscapes_c_images,
)
from image_visualization import (
    plot_quality_distributions, plot_cityscapes_corruption_metrics,
    plot_dataset_comparison, save_image_previews,
)


def calculate_entropy(gray: np.ndarray) -> float:
    """Calculate Shannon entropy (bits) from a grayscale image."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    probabilities = hist / max(hist.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def analyse_image(image_bytes: bytes) -> dict:
    """Calculate dimensions and simple image-quality statistics."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "width": int(image.width),
        "height": int(image.height),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": calculate_entropy(gray),
        "file_size_kb": len(image_bytes) / 1024.0,
    }


def analyse_acdc_images(rgb_zip, sample: list, preview_limit: int = 6):
    """Analyse sampled ACDC RGB images; returns DataFrame and preview records."""
    records, previews = [], []
    for item in tqdm(sample, desc="Analysing ACDC images"):
        try:
            payload = rgb_zip.read(item["image_path"])
            stats = analyse_image(payload)
            record = {
                "filename": Path(item["image_path"]).name,
                "split": item.get("split", "unknown"),
                **stats,
            }
            records.append(record)
            if len(previews) < preview_limit:
                previews.append({
                    "filename": record["filename"],
                    "image": Image.open(io.BytesIO(payload)).convert("RGB").copy(),
                })
        except Exception as exc:
            print(f"Skipping ACDC image {item['image_path']}: {exc}")
    return pd.DataFrame(records), previews


def analyse_cityscapes_c(
    max_images: int = MAX_CITYC_IMAGES,
    max_per_group: int = MAX_PER_GROUP,
    preview_limit: int = 6,
):
    """Stream and analyse a bounded sample of Cityscapes-C images."""
    records, previews = [], []
    skipped = 0
    for item in tqdm(
        iter_cityscapes_c_images(max_images=max_images, max_per_group=max_per_group),
        total=max_images,
        desc="Analysing Cityscapes-C",
    ):
        try:
            stats = analyse_image(item["image_bytes"])
            records.append({
                "filename": item["filename"],
                "archive_path": item["archive_path"],
                "corruption": item["corruption"],
                "severity": item["severity"],
                **stats,
            })
            if len(previews) < preview_limit:
                previews.append({
                    "filename": item["filename"],
                    "corruption": item["corruption"],
                    "severity": item["severity"],
                    "image": Image.open(io.BytesIO(item["image_bytes"])).convert("RGB").copy(),
                })
        except Exception as exc:
            skipped += 1
            print(f"Skipping Cityscapes-C entry {item.get('archive_path')}: {exc}")
    return pd.DataFrame(records), previews, skipped


def run_dataset_analysis(
    output_dir: str | Path = "Cityscapes_EDA/results",
    max_acdc_images: int = MAX_ACDC_IMAGES,
    max_cityc_images: int = MAX_CITYC_IMAGES,
    max_per_group: int = MAX_PER_GROUP,
    rgb_url: str = ACDC_RGB_URL,
    gt_url: str = ACDC_GT_URL,
):
    """Run the image-quality EDA and export CSVs/plots/reports.

    Returns a dictionary with archive handles, sampled pairs, dataframes,
    previews, and output directory paths for use by the other modules.
    """
    output_dir = Path(output_dir)
    csv_dir, graph_dir, report_dir = (
        output_dir / "csv", output_dir / "graphs", output_dir / "reports"
    )
    for folder in (csv_dir, graph_dir, report_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rgb_zip, gt_zip = connect_acdc_archives(rgb_url, gt_url)
    rgb_files, gt_files = list_acdc_fog_files(rgb_zip, gt_zip)
    matched = match_acdc_images(rgb_files, gt_files)
    sample = sample_acdc_images(matched, max_acdc_images)

    diagnostics = {
        "rgb_files": len(rgb_files),
        "gt_files": len(gt_files),
        "matched_pairs": len(matched),
        "sample_size": len(sample),
        "split_counts": pd.Series([x["split"] for x in sample]).value_counts().to_dict(),
    }
    acdc_df, acdc_previews = analyse_acdc_images(rgb_zip, sample)
    cityc_df, cityc_previews, cityc_skipped = analyse_cityscapes_c(
        max_images=max_cityc_images, max_per_group=max_per_group
    )

    acdc_df.to_csv(csv_dir / "ACDC_Fog_Image_Analysis.csv", index=False)
    cityc_df.to_csv(csv_dir / "Cityscapes_C_Image_Analysis.csv", index=False)

    if not acdc_df.empty:
        acdc_df.select_dtypes(include=np.number).describe().T.to_csv(
            csv_dir / "ACDC_Fog_Descriptive_Statistics.csv"
        )
        plot_quality_distributions(acdc_df, graph_dir)
        save_image_previews(acdc_previews, graph_dir / "ACDC_Fog_Sample_Images.png",
                            title="ACDC Fog Sample Images")

    if not cityc_df.empty:
        cityc_df.select_dtypes(include=np.number).describe().T.to_csv(
            csv_dir / "Cityscapes_C_Descriptive_Statistics.csv"
        )
        cityc_df.groupby(["corruption", "severity"], dropna=False).size().rename(
            "image_count"
        ).reset_index().to_csv(csv_dir / "Cityscapes_C_Group_Summary.csv", index=False)
        plot_cityscapes_corruption_metrics(cityc_df, graph_dir)
        save_image_previews(cityc_previews, graph_dir / "Cityscapes_C_Sample_Images.png",
                            title="Cityscapes-C Sample Images")

    if not acdc_df.empty and not cityc_df.empty:
        plot_dataset_comparison(acdc_df, cityc_df, graph_dir)
        comparison = pd.concat([
            acdc_df.assign(dataset="ACDC Fog"),
            cityc_df.assign(dataset="Cityscapes-C"),
        ], ignore_index=True)
        comparison.to_csv(csv_dir / "ACDC_vs_CityscapesC_Comparison.csv", index=False)

    report = [
        "CITYSCAPES-RELATED DATASET EDA REPORT",
        "=" * 55,
        f"ACDC RGB fog entries: {len(rgb_files)}",
        f"ACDC annotation fog entries: {len(gt_files)}",
        f"Matched RGB-mask pairs: {len(matched)}",
        f"ACDC images analysed: {len(acdc_df)}",
        f"Cityscapes-C images analysed: {len(cityc_df)}",
        f"Cityscapes-C entries skipped: {cityc_skipped}",
        f"Cityscapes-C corruption groups: {cityc_df['corruption'].nunique() if not cityc_df.empty else 0}",
    ]
    (report_dir / "Dataset_EDA_Report.txt").write_text("\n".join(report), encoding="utf-8")

    return {
        "rgb_zip": rgb_zip, "gt_zip": gt_zip,
        "rgb_files": rgb_files, "gt_files": gt_files,
        "matched_acdc": matched, "acdc_sample": sample,
        "acdc_df": acdc_df, "cityc_df": cityc_df,
        "acdc_previews": acdc_previews, "cityc_previews": cityc_previews,
        "diagnostics": diagnostics,
        "output_dirs": {"csv": csv_dir, "graphs": graph_dir, "reports": report_dir},
    }
