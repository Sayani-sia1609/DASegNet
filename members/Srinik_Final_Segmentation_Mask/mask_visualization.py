
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from .class_mapping import colorize_mask


def visualize_mask(
    mask,
    class_names,
    class_colors,
    ignore_index=None,
    ignore_color=(0, 0, 0),
    title="Predicted Segmentation Mask",
    save_path=None,
    batch_index=0
):
    """
    Visualize a color-coded semantic segmentation mask.

    Args:
        mask: NumPy array or PyTorch tensor of shape
              [H, W] or [B, H, W].
        class_names: Mapping of class IDs to class names.
        class_colors: Mapping of class IDs to RGB colors.
        ignore_index: Optional ignored class ID.
        ignore_color: RGB color for ignored pixels.
        title: Visualization title.
        save_path: Optional path to save the visualization.
        batch_index: Selects an image from a batch.

    Returns:
        Color-coded RGB segmentation mask.
    """

    # Convert PyTorch tensor into NumPy array
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    elif not isinstance(mask, np.ndarray):
        raise TypeError(
            "Mask must be a NumPy array or PyTorch tensor."
        )

    # Validate batch index
    if (
        isinstance(batch_index, (bool, np.bool_))
        or not isinstance(batch_index, (int, np.integer))
        or batch_index < 0
    ):
        raise ValueError("batch_index must be a non-negative integer.")

    # Handle batched and unbatched masks
    if mask.ndim == 3:

        if mask.shape[0] == 0:
            raise ValueError("Mask batch cannot be empty.")

        if batch_index >= mask.shape[0]:
            raise IndexError("batch_index exceeds batch size.")

        mask = mask[batch_index]

    elif mask.ndim != 2:
        raise ValueError(
            "Expected mask shape [H, W] or [B, H, W]."
        )

    # Convert numerical mask into RGB mask
    rgb_mask = colorize_mask(
        mask,
        class_names,
        class_colors,
        ignore_index,
        ignore_color
    )

    # Display segmentation mask
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.imshow(rgb_mask)
    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()

    # Save visualization if a path is provided
    if save_path is not None:

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150
        )

    plt.show()
    plt.close(fig)

    return rgb_mask
