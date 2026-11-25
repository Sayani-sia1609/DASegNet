"""Spatial maps of local image-quality characteristics."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


class SpatialDegradationMapper:
    """Generate spatial image-quality maps without classifying degradation.

    The maps describe local brightness, contrast, sharpness, and a robust
    noise estimate. These indicators can be affected by scene content and are
    not definitive degradation classifications.
    """

    MAP_NAMES = (
        "brightness_map",
        "contrast_map",
        "sharpness_map",
        "noise_map",
    )
    _SUPPORTED_DTYPES = (
        np.uint8,
        np.uint16,
        np.int16,
        np.int32,
        np.float16,
        np.float32,
        np.float64,
    )
    _NOISE_KERNEL = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    _MAD_NORMALIZATION = 0.6744897501960817 * np.sqrt(20.0)

    def __init__(self, window_size: int = 15) -> None:
        """Create a mapper using the given odd local window size.

        Args:
            window_size: Odd integer window width and height, at least 3.
                Border pixels are handled by replicating the nearest image
                values, so windows may safely extend beyond small images.

        Raises:
            TypeError: If ``window_size`` is not an integer.
            ValueError: If ``window_size`` is less than 3 or even.
        """
        if isinstance(window_size, (bool, np.bool_)) or not isinstance(
            window_size, (int, np.integer)
        ):
            raise TypeError("window_size must be an integer")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer of at least 3")
        self.window_size = int(window_size)

    @classmethod
    def _to_grayscale(cls, image: np.ndarray) -> np.ndarray:
        """Validate an input image and return a float64 grayscale copy."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy ndarray")
        if image.ndim not in (2, 3):
            raise ValueError(
                "image must be a 2D grayscale image or a 3D grayscale/RGB image"
            )
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image height and width must be non-zero")
        if image.dtype not in cls._SUPPORTED_DTYPES:
            raise TypeError(
                "unsupported image data type "
                f"{image.dtype}; expected uint8, uint16, int16, int32, "
                "float16, float32, or float64"
            )

        if image.ndim == 2:
            grayscale = image
        elif image.shape[2] == 1:
            grayscale = image[:, :, 0]
        elif image.shape[2] == 3:
            grayscale = (
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]
            )
        else:
            raise ValueError("3D images must have one or three channels")

        grayscale = np.array(grayscale, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(grayscale)):
            raise ValueError("image contains NaN or infinite pixel values")
        return grayscale

    def _local_mean(self, values: np.ndarray) -> np.ndarray:
        """Return a same-sized local mean with replicated borders."""
        return cv2.boxFilter(
            values,
            ddepth=cv2.CV_64F,
            ksize=(self.window_size, self.window_size),
            normalize=True,
            borderType=cv2.BORDER_REPLICATE,
        )

    def _local_mad(self, values: np.ndarray) -> np.ndarray:
        """Compute exact same-sized local MAD values in bounded-memory tiles."""
        radius = self.window_size // 2
        padded = cv2.copyMakeBorder(
            values,
            radius,
            radius,
            radius,
            radius,
            borderType=cv2.BORDER_REPLICATE,
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (self.window_size, self.window_size)
        )
        height, width = values.shape
        mad_map = np.empty((height, width), dtype=np.float64)

        # Bound temporary absolute-deviation arrays for large images/windows.
        outputs_per_tile = max(1, 262_144 // (self.window_size**2))
        tile_width = min(width, outputs_per_tile)
        tile_height = max(1, outputs_per_tile // tile_width)

        for row_start in range(0, height, tile_height):
            row_end = min(row_start + tile_height, height)
            for col_start in range(0, width, tile_width):
                col_end = min(col_start + tile_width, width)
                tile_windows = windows[row_start:row_end, col_start:col_end]
                tile_medians = np.median(tile_windows, axis=(-2, -1))
                deviations = np.abs(
                    tile_windows - tile_medians[..., np.newaxis, np.newaxis]
                )
                mad_map[row_start:row_end, col_start:col_end] = np.median(
                    deviations, axis=(-2, -1)
                )
        return mad_map

    def generate_maps(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate four local image-quality maps for an RGB or grayscale image.

        The maps have the same height and width as the input:

        * ``brightness_map`` is the local mean grayscale intensity.
        * ``contrast_map`` is the local population standard deviation of
          grayscale intensity.
        * ``sharpness_map`` is the local variance of the 3x3 Laplacian
          response. It can be elevated by texture or noise as well as detail.
        * ``noise_map`` is the local median absolute deviation (MAD) of a
          four-neighbor 3x3 high-pass response, divided by
          ``0.6744897501960817 * sqrt(20)``. The factor follows from the
          Gaussian MAD conversion and the kernel's squared coefficient sum
          of 20. Texture and image details can influence this estimate; it is
          not a definitive measure of sensor noise.

        Pixel intensities are not normalized, so map values retain the input
        intensity scale. Edges use replicated border values, including when
        the window is larger than the image.

        Args:
            image: A finite NumPy grayscale image of shape ``(H, W)`` or an
                RGB image of shape ``(H, W, 3)``. A single-channel
                ``(H, W, 1)`` grayscale array is also accepted.

        Returns:
            A dictionary of float64 maps in the order
            ``brightness_map``, ``contrast_map``, ``sharpness_map``,
            ``noise_map``.
        """
        grayscale = self._to_grayscale(image)

        brightness_map = self._local_mean(grayscale)
        mean_square_map = self._local_mean(np.square(grayscale))
        contrast_map = np.sqrt(
            np.maximum(mean_square_map - np.square(brightness_map), 0.0)
        )

        laplacian = cv2.Laplacian(
            grayscale,
            ddepth=cv2.CV_64F,
            ksize=3,
            borderType=cv2.BORDER_REPLICATE,
        )
        laplacian_mean = self._local_mean(laplacian)
        laplacian_mean_square = self._local_mean(np.square(laplacian))
        sharpness_map = np.maximum(
            laplacian_mean_square - np.square(laplacian_mean), 0.0
        )

        high_pass_response = cv2.filter2D(
            grayscale,
            ddepth=cv2.CV_64F,
            kernel=self._NOISE_KERNEL,
            borderType=cv2.BORDER_REPLICATE,
        )
        noise_map = self._local_mad(high_pass_response) / self._MAD_NORMALIZATION

        return {
            "brightness_map": brightness_map,
            "contrast_map": contrast_map,
            "sharpness_map": sharpness_map,
            "noise_map": noise_map,
        }

    def stack_maps(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack generated maps in channel-first ``(4, H, W)`` order.

        Args:
            maps: Mapping containing all four maps returned by
                :meth:`generate_maps`.

        Returns:
            A float64 array ordered brightness, contrast, sharpness, then noise.

        Raises:
            TypeError: If ``maps`` is not a dictionary, a map is not an array,
                or a map contains non-numeric or complex values.
            ValueError: If a required map is missing, is not 2D, or has a
                different spatial shape from the other maps, or contains
                non-finite values.
        """
        if not isinstance(maps, dict):
            raise TypeError("maps must be a dictionary returned by generate_maps")

        arrays = []
        expected_shape = None
        for name in self.MAP_NAMES:
            if name not in maps:
                raise ValueError(f"maps is missing required key {name!r}")
            current = maps[name]
            if not isinstance(current, np.ndarray):
                raise TypeError(f"map {name!r} must be a NumPy ndarray")
            if current.ndim != 2:
                raise ValueError(f"map {name!r} must be a 2D array")
            if not np.issubdtype(current.dtype, np.number):
                raise TypeError(f"map {name!r} must contain real numeric values")
            if np.issubdtype(current.dtype, np.complexfloating):
                raise TypeError(f"map {name!r} must not contain complex values")
            if not np.isfinite(current).all():
                raise ValueError(f"map {name!r} contains NaN or infinite values")
            if expected_shape is None:
                expected_shape = current.shape
            elif current.shape != expected_shape:
                raise ValueError("all maps must have the same height and width")
            converted = current.astype(np.float64, copy=False)
            if not np.isfinite(converted).all():
                raise ValueError(
                    f"map {name!r} contains values that are not finite as float64"
                )
            arrays.append(converted)

        return np.stack(arrays, axis=0)