"""Feature-wise Linear Modulation conditioned on degradation embeddings."""

from __future__ import annotations

import torch
from torch import nn


class FiLMModule(nn.Module):
    """Apply spatially varying affine modulation to fused CNN features.

    A convolutional conditioner predicts per-location scale ``gamma`` and
    shift ``beta`` maps from an aligned degradation embedding. Its final
    convolution is zero-initialized, making the initial transformation an
    identity while allowing the parameters to learn through backpropagation.

    Inputs and the output have shape ``(B, C, H, W)``; the two inputs must
    share batch size, spatial dimensions, device, and floating-point dtype.
    """

    def __init__(
        self,
        feature_channels: int = 128,
        degradation_channels: int = 128,
    ) -> None:
        """Create the FiLM conditioner.

        Args:
            feature_channels: Channel count of the fused features and the
                generated ``gamma`` and ``beta`` maps.
            degradation_channels: Channel count of the degradation embedding.

        Raises:
            TypeError: If a channel count is not an integer or is a boolean.
            ValueError: If a channel count is not positive.
        """
        super().__init__()
        self._validate_channel_count("feature_channels", feature_channels)
        self._validate_channel_count("degradation_channels", degradation_channels)

        self.feature_channels = feature_channels
        self.degradation_channels = degradation_channels
        self.conditioner = nn.Sequential(
            nn.Conv2d(
                degradation_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                feature_channels,
                2 * feature_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    @staticmethod
    def _validate_channel_count(name: str, value: int) -> None:
        """Require a positive integer channel count, excluding booleans."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_tensor(
        tensor: object,
        name: str,
        expected_channels: int,
    ) -> torch.Tensor:
        """Validate an NCHW tensor's type, dimensions, channels, and values."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape (B, C, H, W); "
                f"received {tuple(tensor.shape)}"
            )
        if tensor.shape[0] <= 0 or tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
            raise ValueError(f"{name} must have non-empty batch and spatial dimensions")
        if tensor.shape[1] != expected_channels:
            raise ValueError(
                f"{name} must have {expected_channels} channels; "
                f"received {tensor.shape[1]}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} contains NaN or infinite values")
        return tensor

    def forward(
        self,
        fused_features: torch.Tensor,
        degradation_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate fused features using an already aligned degradation map.

        Args:
            fused_features: Tensor of shape
                ``(B, feature_channels, H, W)``.
            degradation_embedding: Tensor of shape
                ``(B, degradation_channels, H, W)``. This module does not
                resize inputs; spatial alignment is handled upstream.

        Returns:
            FiLM-modulated tensor with the same shape as ``fused_features``.

        Raises:
            TypeError: If either input is not a floating-point tensor.
            ValueError: If input dimensions, channels, batch, spatial size,
                device, dtype, or finite values are invalid, or if the result
                is non-finite.
        """
        fused_features = self._validate_tensor(
            fused_features, "fused_features", self.feature_channels
        )
        degradation_embedding = self._validate_tensor(
            degradation_embedding,
            "degradation_embedding",
            self.degradation_channels,
        )

        if fused_features.shape[0] != degradation_embedding.shape[0]:
            raise ValueError(
                "fused_features and degradation_embedding must have the same batch size"
            )
        if fused_features.shape[-2:] != degradation_embedding.shape[-2:]:
            raise ValueError(
                "fused_features and degradation_embedding must have the same "
                "spatial dimensions"
            )
        if fused_features.device != degradation_embedding.device:
            raise ValueError(
                "fused_features and degradation_embedding must be on the same device"
            )
        if fused_features.dtype != degradation_embedding.dtype:
            raise ValueError(
                "fused_features and degradation_embedding must have the same dtype"
            )

        gamma, beta = self.conditioner(degradation_embedding).chunk(2, dim=1)
        modulated_features = fused_features * (1.0 + gamma) + beta
        if modulated_features.shape != fused_features.shape:
            raise ValueError("FiLM output shape does not match fused_features")
        if not torch.isfinite(modulated_features).all().item():
            raise ValueError("FiLM output contains NaN or infinite values")
        return modulated_features