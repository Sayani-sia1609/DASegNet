"""Dataset download, archive inspection, RGB-mask pairing, and sample loading.

This module is responsible only for obtaining dataset entries and reading bytes.
It does not perform image analysis or plotting.
"""
from __future__ import annotations

import io
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import requests
from PIL import Image

HF_REPO = "AaronCIH/Segmentation"
ACDC_RGB_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/ACDC/rgb_anon_trainvaltest.zip"
ACDC_GT_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/ACDC/gt_trainval.zip"
CITYSCAPES_C_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/Cityscape-C.tar.gz"

MAX_ACDC_IMAGES = 120
MAX_CITYC_IMAGES = 500
MAX_PER_GROUP = 30
RANDOM_SEED = 42

CORRUPTION_NAMES = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "saturate", "speckle_noise", "gaussian_blur",
]


def get_split(path: str) -> str:
    """Extract train/val/test from an archive path."""
    parts = path.lower().replace("\\", "/").split("/")
    return next((part for part in ("train", "val", "test") if part in parts), "unknown")


def get_sequence(path: str) -> str:
    return Path(path).parent.name


def get_image_key(filename: str) -> str:
    """Normalize ACDC RGB and annotation filenames to a shared key."""
    name = Path(filename).name
    suffixes = (
        "_rgb_anon.png", "_gt_labelIds.png", "_gt_labelTrainIds.png",
        "_gt_labelColor.png", "_gt_invIds.png", "_gt_invGray.png",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def connect_acdc_archives(
    rgb_url: str = ACDC_RGB_URL,
    gt_url: str = ACDC_GT_URL,
):
    """Open the ACDC RGB and GT remote ZIP archives and return both handles."""
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise ImportError("Install remotezip first: pip install remotezip") from exc
    rgb_zip = RemoteZip(rgb_url)
    try:
        gt_zip = RemoteZip(gt_url)
    except Exception:
        rgb_zip.close()
        raise
    return rgb_zip, gt_zip


def list_acdc_fog_files(rgb_zip, gt_zip) -> Tuple[List[str], List[str]]:
    """List ACDC fog RGB files (train/val) and fog annotation PNGs."""
    rgb_files = [
        name for name in rgb_zip.namelist()
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
        and "/fog/" in ("/" + name.lower())
        and ("/train/" in ("/" + name.lower()) or "/val/" in ("/" + name.lower()))
    ]
    gt_files = [
        name for name in gt_zip.namelist()
        if name.lower().endswith(".png") and "/fog/" in ("/" + name.lower())
    ]
    return rgb_files, gt_files


def match_acdc_images(rgb_files: Iterable[str], gt_files: Iterable[str]) -> List[dict]:
    """Pair each RGB image with all annotation candidates sharing its filename key."""
    lookup = defaultdict(list)
    for mask_path in gt_files:
        lookup[get_image_key(mask_path)].append(mask_path)

    matched = []
    for image_path in rgb_files:
        key = get_image_key(image_path)
        masks = lookup.get(key, [])
        if masks:
            matched.append({
                "image_path": image_path,
                "mask_paths": masks,
                "key": key,
                "split": get_split(image_path),
                "sequence": get_sequence(image_path),
            })
    return matched


def sample_acdc_images(
    matched_images: List[dict],
    max_images: int = MAX_ACDC_IMAGES,
    seed: int = RANDOM_SEED,
) -> List[dict]:
    """Reproducibly sample matched ACDC examples."""
    if not matched_images:
        raise ValueError("No matched ACDC RGB/mask pairs were found.")
    rng = random.Random(seed)
    return rng.sample(matched_images, min(max_images, len(matched_images)))


def read_image_bytes(archive, archive_path: str) -> bytes:
    """Read one file from a RemoteZip archive."""
    return archive.read(archive_path)


def decode_rgb(image_bytes: bytes):
    """Decode image bytes as an RGB uint8 NumPy array."""
    import numpy as np
    return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)


def infer_corruption_severity(filepath: str) -> Tuple[str, object]:
    """Infer Cityscapes-C corruption name and severity from archive folders."""
    parts = filepath.lower().replace("\\", "/").split("/")
    corruption, severity = "unknown", "unknown"
    for part in parts:
        if part in CORRUPTION_NAMES:
            corruption = part
        if part in ("1", "2", "3", "4", "5"):
            severity = int(part)
        match = re.search(r"severity[_-]?([1-5])", part)
        if match:
            severity = int(match.group(1))
    return corruption, severity


def iter_cityscapes_c_images(
    archive_url: str = CITYSCAPES_C_URL,
    max_images: int = MAX_CITYC_IMAGES,
    max_per_group: int = MAX_PER_GROUP,
    timeout: Tuple[int, int] = (30, 180),
) -> Iterator[dict]:
    """Stream image entries from Cityscapes-C tar.gz without extracting the archive.

    Yields dictionaries containing archive_path, filename, corruption, severity,
    and image_bytes. The caller is responsible for processing each byte payload.
    """
    response = requests.get(archive_url, stream=True, timeout=timeout)
    response.raise_for_status()
    response.raw.decode_content = True
    group_counts = defaultdict(int)
    processed = 0
    try:
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                if processed >= max_images:
                    break
                corruption, severity = infer_corruption_severity(member.name)
                group = (corruption, severity)
                if group_counts[group] >= max_per_group:
                    continue
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    continue
                payload = file_obj.read()
                group_counts[group] += 1
                processed += 1
                yield {
                    "filename": Path(member.name).name,
                    "archive_path": member.name,
                    "corruption": corruption,
                    "severity": severity,
                    "image_bytes": payload,
                }
    finally:
        response.close()


def get_archive_diagnostics(rgb_files: List[str], gt_files: List[str]) -> dict:
    """Return useful archive counts and split/sequence overlap diagnostics."""
    from collections import Counter
    rgb_splits = Counter(get_split(path) for path in rgb_files)
    gt_splits = Counter(get_split(path) for path in gt_files)
    rgb_sequences = Counter(get_sequence(path) for path in rgb_files)
    gt_sequences = Counter(get_sequence(path) for path in gt_files)
    return {
        "rgb_count": len(rgb_files),
        "gt_count": len(gt_files),
        "rgb_splits": dict(rgb_splits),
        "gt_splits": dict(gt_splits),
        "rgb_sequences": dict(rgb_sequences),
        "gt_sequences": dict(gt_sequences),
        "sequence_overlap": sorted(set(rgb_sequences) & set(gt_sequences)),
        "rgb_examples": rgb_files[:10],
        "gt_examples": gt_files[:10],
    }
