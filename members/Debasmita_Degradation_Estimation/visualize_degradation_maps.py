"""Display and save SpatialDegradationMapper outputs for one RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence, Union

import cv2

if __package__:
    from .spatial_degradation_map import SpatialDegradationMapper
else:
    from spatial_degradation_map import SpatialDegradationMapper


def _load_rgb_image(image_path: Union[str, Path]):
    """Load an image as RGB or raise a clear error for an invalid image."""
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    try:
        bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not load image {path}: {exc}") from exc
    if bgr_image is None:
        raise ValueError(f"Could not decode image file: {path}")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def visualize_degradation_maps(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Generate, display, and save the four spatial degradation maps.

    The saved visualization is a PNG. By default it is written next to the
    input image as ``<image_stem>_degradation_maps.png``.
    """
    source_path = Path(image_path).expanduser()
    rgb_image = _load_rgb_image(source_path)

    if output_path is None:
        destination = source_path.with_name(
            f"{source_path.stem}_degradation_maps.png"
        )
    else:
        destination = Path(output_path).expanduser()
    resolved_source_path = source_path.resolve()
    resolved_destination_path = destination.resolve()
    if resolved_destination_path == resolved_source_path:
        raise ValueError("output path must differ from the input image path")
    if destination.suffix.lower() != ".png":
        raise ValueError("output path must have a .png extension")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {destination.parent}"
        )

    maps = SpatialDegradationMapper().generate_maps(rgb_image)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to display and save degradation maps"
        ) from exc

    titles = (
        ("brightness_map", "Brightness"),
        ("contrast_map", "Contrast"),
        ("sharpness_map", "Sharpness"),
        ("noise_map", "Noise"),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        constrained_layout=True,
    )
    panels = axes.ravel()

    panels[0].imshow(rgb_image)
    panels[0].set_title("Original RGB Image")
    panels[0].axis("off")

    for axis, (map_name, title) in zip(panels[1:5], titles):
        display = axis.imshow(maps[map_name], cmap="viridis")
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(display, ax=axis, fraction=0.046, pad=0.04)

    panels[5].axis("off")
    try:
        figure.savefig(destination, format="png", dpi=150)
        plt.show()
    finally:
        plt.close(figure)

    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize brightness, contrast, sharpness, and noise maps."
    )
    parser.add_argument("image_path", help="Path to an RGB image")
    parser.add_argument(
        "--output",
        help="PNG output path (default: <image_stem>_degradation_maps.png)",
    )
    arguments = parser.parse_args(argv)

    try:
        destination = visualize_degradation_maps(
            arguments.image_path,
            arguments.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Saved degradation-map visualization to: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
