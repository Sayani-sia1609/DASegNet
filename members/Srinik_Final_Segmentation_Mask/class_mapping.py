
import numpy as np
from collections.abc import Mapping


def validate_class_mapping(class_names, class_colors, ignore_index=None):
    """
    Validate semantic class names and their RGB colour mappings.
    """

    if not isinstance(class_names, Mapping):
        raise TypeError("class_names must be a mapping.")

    if not isinstance(class_colors, Mapping):
        raise TypeError("class_colors must be a mapping.")

    if not class_names:
        raise ValueError("Class mapping cannot be empty.")

    if set(class_names.keys()) != set(class_colors.keys()):
        raise ValueError(
            "Class IDs in class_names and class_colors must match."
        )

    for class_id, name in class_names.items():

        if isinstance(class_id, (bool, np.bool_)) or not isinstance(
            class_id, (int, np.integer)
        ):
            raise TypeError("Class IDs must be integers.")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every class must have a valid name.")

    for class_id, color in class_colors.items():

        if isinstance(color, (str, bytes)):
            raise ValueError("Colors must contain three RGB values.")

        try:
            color = tuple(color)
        except TypeError:
            raise ValueError("Invalid RGB color.")

        if len(color) != 3:
            raise ValueError("Every color must have three RGB values.")

        for channel in color:
            if (
                isinstance(channel, (bool, np.bool_))
                or not isinstance(channel, (int, np.integer))
                or not 0 <= channel <= 255
            ):
                raise ValueError("RGB values must be integers from 0 to 255.")

    if ignore_index is not None:
        if isinstance(ignore_index, (bool, np.bool_)) or not isinstance(
            ignore_index, (int, np.integer)
        ):
            raise TypeError("ignore_index must be an integer.")

        if ignore_index in class_names:
            raise ValueError("Ignore index must not overlap a class ID.")

    return True


def get_class_name(class_id, class_names):
    """
    Return the semantic category corresponding to a class ID.
    """

    if class_id not in class_names:
        raise KeyError(f"Unknown class ID: {class_id}")

    return class_names[class_id]


def colorize_mask(
    mask,
    class_names,
    class_colors,
    ignore_index=None,
    ignore_color=(0, 0, 0)
):
    """
    Convert a numerical segmentation mask into an RGB mask.
    """

    validate_class_mapping(
        class_names,
        class_colors,
        ignore_index
    )

    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a NumPy array.")

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Mask must contain integer class IDs.")

    if mask.size == 0:
        raise ValueError("Mask cannot be empty.")

    if len(ignore_color) != 3 or any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, np.integer))
        or not 0 <= v <= 255
        for v in ignore_color
    ):
        raise ValueError("ignore_color must contain three valid RGB values.")

    known_ids = set(class_colors.keys())
    allowed_ids = known_ids.copy()

    if ignore_index is not None:
        allowed_ids.add(ignore_index)

    unknown_ids = set(np.unique(mask).tolist()) - allowed_ids

    if unknown_ids:
        raise ValueError(f"Unknown class IDs found in mask: {unknown_ids}")

    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        rgb_mask[mask == class_id] = color

    if ignore_index is not None:
        rgb_mask[mask == ignore_index] = ignore_color

    return rgb_mask
