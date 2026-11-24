
import numpy as np
import torch
import cv2

from pathlib import Path
from PIL import Image

from .class_mapping import colorize_mask


def create_prediction_overlay(
    image,
    mask,
    class_names,
    class_colors,
    alpha=0.5,
    ignore_index=None,
    ignore_color=(0, 0, 0),
    image_color_space="RGB",
    save_path=None,
    batch_index=0
):
    """
    Overlay a predicted segmentation mask on the original image.

    Args:
        image: Original image as NumPy array or PyTorch tensor.
        mask: Predicted mask of shape [H, W] or [B, H, W].
        class_names: Mapping of class IDs to semantic names.
        class_colors: Mapping of class IDs to RGB colors.
        alpha: Transparency of the segmentation mask (0 to 1).
        ignore_index: Optional ignored class ID.
        ignore_color: RGB colour for ignored pixels.
        image_color_space: Either RGB or BGR.
        save_path: Optional path for saving the overlay.
        batch_index: Selects an image from a batch.

    Returns:
        RGB overlay as a NumPy array.
    """

    # Validate alpha
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.number))
        or not np.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("alpha must be between 0 and 1.")

    # Validate image colour format
    if image_color_space not in ("RGB", "BGR"):
        raise ValueError("image_color_space must be RGB or BGR.")

    # Validate batch index
    if (
        isinstance(batch_index, (bool, np.bool_))
        or not isinstance(batch_index, (int, np.integer))
        or batch_index < 0
    ):
        raise ValueError("batch_index must be a non-negative integer.")

    # Convert image tensor to NumPy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array or PyTorch tensor.")

    # Handle batched images
    if image.ndim == 4:

        if image.shape[0] == 0 or batch_index >= image.shape[0]:
            raise IndexError("Invalid image batch index.")

        image = image[batch_index]

    # Convert channel-first image to channel-last
    if image.ndim == 3:

        if (
            image.shape[0] in (1, 3, 4)
            and image.shape[-1] not in (1, 3, 4)
        ):
            image = np.transpose(image, (1, 2, 0))

    # Handle grayscale images
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Image must have three colour channels.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image dimensions cannot be zero.")

    # Validate image values
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("Image must contain numerical pixel values.")

    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains non-finite values.")

    if np.min(image) < 0 or np.max(image) > 255:
        raise ValueError("Image values must be between 0 and 255.")

    # Convert floating-point images to uint8
    if np.issubdtype(image.dtype, np.floating):

        if np.max(image) <= 1:
            image = image * 255

        image = np.round(image).astype(np.uint8)

    else:
        image = image.astype(np.uint8)

    # Convert BGR image to RGB if needed
    if image_color_space == "BGR":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert mask tensor to NumPy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a NumPy array or PyTorch tensor.")

    # Handle batched masks
    if mask.ndim == 3:

        if mask.shape[0] == 0 or batch_index >= mask.shape[0]:
            raise IndexError("Invalid mask batch index.")

        mask = mask[batch_index]

    if mask.ndim != 2:
        raise ValueError("Mask must have shape [H, W] or [B, H, W].")

    # Ensure image and mask dimensions match
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "Image and mask dimensions must match. "
            f"Image: {image.shape[:2]}, Mask: {mask.shape}"
        )

    # Generate colour-coded segmentation mask
    rgb_mask = colorize_mask(
        mask,
        class_names,
        class_colors,
        ignore_index,
        ignore_color
    )

    # Blend original image with predicted mask
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        rgb_mask,
        alpha,
        0
    )

    # Save overlay if requested
    if save_path is not None:

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(overlay).save(save_path)

    return overlay
