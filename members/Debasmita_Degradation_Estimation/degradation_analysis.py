"""Numerical image-quality indicators related to image degradation.

The returned measurements are intended as lightweight inputs to a degradation
estimation module. They are image-quality indicators, not definitive
degradation classifications. RGB inputs are interpreted in RGB channel order.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


class DegradationAnalyzer:
    """Calculate image-quality indicators, not definitive degradation labels.

    Brightness, contrast, sharpness, and noise measurements can help characterize
    an image, but none alone determines the degradation type or severity.
    """

    _SUPPORTED_DTYPES = (
        np.uint8,
        np.uint16,
        np.int16,
        np.int32,
        np.float16,
        np.float32,
        np.float64,
    )

    @classmethod
    def _validate_and_convert(cls, image: np.ndarray) -> np.ndarray:
        """Validate an image and return an independent float64 grayscale array.

        Grayscale images may have shape ``(H, W)`` or ``(H, W, 1)``. RGB images
        must have shape ``(H, W, 3)``. Both spatial dimensions must be at least 3.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy ndarray")
        if image.ndim not in (2, 3):
            raise ValueError(
                "unsupported image dimensions: expected 2D grayscale or "
                "3D grayscale/RGB data"
            )
        if image.dtype not in cls._SUPPORTED_DTYPES:
            raise TypeError(
                "unsupported image data type "
                f"{image.dtype}; expected uint8, uint16, int16, int32, "
                "float16, float32, or float64"
            )
        if image.shape[0] < 3 or image.shape[1] < 3:
            raise ValueError("image height and width must both be at least 3")
        if image.ndim == 3:
            if image.shape[2] not in (1, 3):
                raise ValueError(
                    "3D images must contain one grayscale or three RGB channels"
                )
            if image.shape[2] == 1:
                grayscale = image[:, :, 0]
            else:
                grayscale = (
                    0.299 * image[:, :, 0]
                    + 0.587 * image[:, :, 1]
                    + 0.114 * image[:, :, 2]
                )
        else:
            grayscale = image

        # Conversion creates an independent array and makes calculations stable.
        grayscale = np.array(grayscale, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(grayscale)):
            raise ValueError("image contains NaN or infinite pixel values")
        return grayscale

    def calculate_brightness(self, image: np.ndarray) -> float:
        """Return the arithmetic mean grayscale intensity.

        Integer images retain their native scale (for example 0--255).
        Floating-point images retain their supplied scale (commonly 0--1 or
        0--255); no global normalization is performed.
        """
        grayscale = self._validate_and_convert(image)
        return float(np.mean(grayscale, dtype=np.float64))

    def calculate_contrast(self, image: np.ndarray) -> float:
        """Return the population standard deviation of grayscale intensity."""
        grayscale = self._validate_and_convert(image)
        return float(np.std(grayscale, dtype=np.float64))

    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Return the variance of a 3x3 Laplacian response.

        Larger values generally indicate greater edge and texture content.
        The result is sensitive to noise and scene texture.
        """
        grayscale = self._validate_and_convert(image)
        laplacian = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=3)
        return float(np.var(laplacian, dtype=np.float64))

    def estimate_noise(self, image: np.ndarray) -> float:
        """Estimate additive Gaussian noise standard deviation robustly.

        For noise estimation, a four-neighbor 3x3 high-pass kernel
        ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]`` is applied to valid interior
        neighborhoods. For independent, zero-mean Gaussian pixel noise with
        standard deviation ``sigma``, the filtered response is Gaussian with
        standard deviation ``sqrt(20) * sigma`` because the sum of squared
        kernel coefficients is 20. A Gaussian variable's median absolute
        deviation is ``0.6744897501960817`` times its standard deviation, so
        the estimate is
        ``MAD / (0.6744897501960817 * sqrt(20))``.

        This assumes approximately additive, zero-mean Gaussian noise in the
        supplied grayscale scale and independent pixel noise. At least a 5x5
        image is required to provide a 3x3 set of valid interior responses for
        the MAD estimate. Texture, compression artifacts, and low-light sensor
        patterns can bias the estimate. It does not estimate colored noise or
        provide a pixel-wise noise map.
        """
        grayscale = self._validate_and_convert(image)
        if grayscale.shape[0] < 5 or grayscale.shape[1] < 5:
            raise ValueError(
                "image must be at least 5x5 to estimate noise "
                "(at least 3x3 valid interior responses are required)"
            )
        high_pass_kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        response = cv2.filter2D(
            grayscale, cv2.CV_64F, high_pass_kernel, borderType=cv2.BORDER_REPLICATE
        )
        # Exclude padded-border responses so every sample uses the complete
        # kernel over distinct source pixels and has variance 20 * sigma**2.
        response = response[1:-1, 1:-1]
        response_median = np.median(response)
        mad = np.median(np.abs(response - response_median))
        noise_sigma = mad / (0.6744897501960817 * np.sqrt(20.0))
        return float(noise_sigma)

    def analyze_image(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate image-quality indicators for a validated input image.

        These numerical measurements do not provide definitive degradation
        classifications. Complete analysis requires an image of at least 5x5
        pixels because noise estimation requires a 3x3 interior response region.

        Returns:
            A dictionary containing ``brightness``, ``contrast``,
            ``sharpness``, and ``noise`` float values.
        """
        grayscale = self._validate_and_convert(image)
        laplacian = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=3)

        return {
            "brightness": float(np.mean(grayscale, dtype=np.float64)),
            "contrast": float(np.std(grayscale, dtype=np.float64)),
            "sharpness": float(np.var(laplacian, dtype=np.float64)),
            "noise": self.estimate_noise(image),
        }