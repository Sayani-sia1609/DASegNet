"""Semantic-mask inspection and class-pixel distribution analysis."""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from image_visualization import plot_class_distribution


def select_semantic_mask(mask_paths: list[str]):
    """Prefer trainIds masks; fall back to labelIds masks."""
    train_ids = [p for p in mask_paths if "_gt_labelTrainIds.png" in p]
    if train_ids:
        return train_ids[0]
    label_ids = [p for p in mask_paths if "_gt_labelIds.png" in p]
    return label_ids[0] if label_ids else None


def analyse_class_distribution(gt_zip, matched_sample: list[dict]):
    """Count pixels per label and collect per-mask dimensions/label counts.

    Note: if labelIds are used as fallback, IDs are raw dataset label IDs;
    trainIds are preferred because they represent the training class mapping.
    """
    class_pixel_counts = Counter()
    mask_records = []

    for item in tqdm(matched_sample, desc="Analysing semantic masks"):
        mask_path = select_semantic_mask(item.get("mask_paths", []))
        if mask_path is None:
            continue
        try:
            mask_bytes = gt_zip.read(mask_path)
            mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
            values, counts = np.unique(mask, return_counts=True)
            for value, count in zip(values, counts):
                class_pixel_counts[int(value)] += int(count)
            mask_records.append({
                "filename": Path(mask_path).name,
                "source_key": item.get("key"),
                "split": item.get("split", "unknown"),
                "width": int(mask.shape[1]),
                "height": int(mask.shape[0]),
                "unique_labels": int(len(values)),
                "total_pixels": int(mask.size),
            })
        except Exception as exc:
            print(f"Skipping mask {mask_path}: {exc}")

    mask_df = pd.DataFrame(mask_records)
    class_df = pd.DataFrame(
        class_pixel_counts.items(), columns=["label_id", "pixel_count"]
    ).sort_values("label_id").reset_index(drop=True)

    if not class_df.empty:
        total = int(class_df["pixel_count"].sum())
        class_df["pixel_percentage"] = class_df["pixel_count"] / total * 100.0
    return mask_df, class_df


def save_class_distribution(
    gt_zip,
    matched_sample: list[dict],
    csv_dir: str | Path,
    graph_dir: str | Path,
):
    """Run mask analysis and save class-distribution CSV and chart."""
    csv_dir, graph_dir = Path(csv_dir), Path(graph_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    mask_df, class_df = analyse_class_distribution(gt_zip, matched_sample)
    mask_df.to_csv(csv_dir / "ACDC_Fog_Mask_Analysis.csv", index=False)
    class_df.to_csv(csv_dir / "ACDC_Fog_Class_Pixel_Distribution.csv", index=False)
    if not class_df.empty:
        plot_class_distribution(
            class_df, graph_dir / "ACDC_Fog_Class_Distribution.png"
        )
    return mask_df, class_df
