"""Plotting and visual preview helpers used by the EDA modules."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_image_previews(previews: list[dict], output_path: str | Path,
                        title: str = "Image Samples"):
    """Save a grid of preview images. Each preview has an 'image' PIL object."""
    if not previews:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows = int(np.ceil(len(previews) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, previews):
        ax.imshow(item["image"])
        label = item.get("filename", "")
        if "corruption" in item:
            label += f"\n{item['corruption']} | severity {item.get('severity')}"
        ax.set_title(label)
        ax.axis("off")
    for ax in axes[len(previews):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_quality_distributions(dataframe: pd.DataFrame, graph_dir: str | Path):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in ("brightness", "contrast", "blur_score", "entropy")
               if m in dataframe.columns]
    if not metrics:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, metrics):
        sns.histplot(data=dataframe, x=metric, kde=True, ax=ax)
        ax.set_title(f"Distribution of {metric}")
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(graph_dir / "ACDC_Fog_Quality_Distributions.png", dpi=250)
    plt.close(fig)


def plot_class_distribution(class_df: pd.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=class_df, x="label_id", y="pixel_count", ax=ax)
    ax.set_title("ACDC Fog Semantic Class Pixel Distribution")
    ax.set_xlabel("Semantic label ID")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_cityscapes_corruption_metrics(dataframe: pd.DataFrame,
                                       graph_dir: str | Path):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("blur_score", "brightness", "contrast", "entropy"):
        if metric not in dataframe.columns:
            continue
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=dataframe, x="corruption", y=metric, ax=ax)
        ax.tick_params(axis="x", rotation=65)
        ax.set_title(f"Cityscapes-C {metric} by Corruption")
        fig.tight_layout()
        fig.savefig(graph_dir / f"Cityscapes_C_{metric}_by_corruption.png", dpi=250)
        plt.close(fig)
        if "severity" in dataframe.columns:
            severity_df = dataframe.copy()
            severity_df["severity"] = pd.to_numeric(severity_df["severity"], errors="coerce")
            fig, ax = plt.subplots(figsize=(13, 6))
            sns.lineplot(data=severity_df, x="severity", y=metric,
                         hue="corruption", marker="o", estimator="mean", errorbar=None, ax=ax)
            ax.set_title(f"Cityscapes-C {metric} vs Severity")
            fig.tight_layout()
            fig.savefig(graph_dir / f"Cityscapes_C_{metric}_vs_severity.png", dpi=250)
            plt.close(fig)


def plot_dataset_comparison(acdc_df: pd.DataFrame, cityc_df: pd.DataFrame,
                            graph_dir: str | Path):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["brightness", "contrast", "blur_score", "entropy"]
    a = acdc_df.assign(dataset="ACDC Fog")
    c = cityc_df.assign(dataset="Cityscapes-C")
    combined = pd.concat([a, c], ignore_index=True)
    for metric in metrics:
        if metric not in combined.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=combined, x="dataset", y=metric, ax=ax)
        ax.set_title(f"ACDC Fog vs Cityscapes-C: {metric}")
        fig.tight_layout()
        fig.savefig(graph_dir / f"ACDC_vs_CityscapesC_{metric}.png", dpi=250)
        plt.close(fig)


def plot_degradation_previews(previews: dict, graph_dir: str | Path):
    """Save one clean/degraded pair per degradation type."""
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    if not previews:
        return
    names = list(previews)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes, names):
        item = previews[name]
        ax.imshow(item["degraded"])
        ax.set_title(f"{name} | severity {item['severity']}")
        ax.axis("off")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Examples of Controlled Image Degradations")
    fig.tight_layout()
    fig.savefig(graph_dir / "Degradation_Examples.png", dpi=250)
    plt.close(fig)


def plot_degradation_metrics(severity_summary: pd.DataFrame,
                             graph_dir: str | Path):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("psnr_db", "PSNR (dB)"), ("ssim", "SSIM"),
        ("laplacian_variance", "Laplacian variance"),
        ("brightness", "Mean brightness"), ("contrast", "Contrast"),
        ("entropy", "Image entropy"),
    ]
    for metric, label in metrics:
        column = f"mean_{metric}"
        if column not in severity_summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.lineplot(data=severity_summary, x="severity", y=column,
                     hue="degradation", marker="o", ax=ax)
        ax.set_title(f"{label} vs Degradation Severity")
        ax.set_xlabel("Severity level")
        ax.set_ylabel(label)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(graph_dir / f"{metric}_vs_severity.png", dpi=250)
        plt.close(fig)
