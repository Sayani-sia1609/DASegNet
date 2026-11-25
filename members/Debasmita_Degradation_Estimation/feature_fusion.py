"""Fusion of multi-scale, degradation-modulated CNN features."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


class FeatureFusion(nn.Module):
    """Fuse modulated C2--C5 features.

    All CNN features are projected to ``fusion_dim`` channels, aligned to the
    C2 resolution, and concatenated. Learnable convolutions fuse these
    representations into a spatial feature map of shape
    ``(B, fusion_dim, H_C2, W_C2)``.
    """

    FEATURE_NAMES = ("C2", "C3", "C4", "C5")
    DEFAULT_INPUT_CHANNELS = {
        "C2": 64,
        "C3": 128,
        "C4": 256,
        "C5": 512,
    }

    def __init__(
        self,
        input_channels: Optional[Mapping[str, int]] = None,
        fusion_dim: int = 128,
    ) -> None:
        """Initialize feature projections and the convolutional fusion head.

        Args:
            input_channels: Expected channel counts for ``C2`` through ``C5``.
                Defaults to ResNet-34 dimensions 64, 128, 256, and 512.
            fusion_dim: Shared projection width and output channel count.

        Raises:
            TypeError: If channel dimensions are not positive integers or
                ``input_channels`` is not a mapping.
            ValueError: If the input-channel mapping has incorrect keys or a
                configured dimension is not positive.
        """
        super().__init__()
        if input_channels is None:
            channels = dict(self.DEFAULT_INPUT_CHANNELS)
        else:
            if not isinstance(input_channels, Mapping):
                raise TypeError("input_channels must be a mapping of feature names to counts")
            missing = set(self.FEATURE_NAMES) - set(input_channels)
            extra = set(input_channels) - set(self.FEATURE_NAMES)
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing {sorted(missing)}")
                if extra:
                    details.append(f"unexpected {sorted(extra)}")
                raise ValueError(
                    "input_channels must define exactly C2, C3, C4, and C5 ("
                    + "; ".join(details)
                    + ")"
                )
            channels = dict(input_channels)

        for name, value in channels.items():
            self._validate_dimension(f"input_channels[{name!r}]", value)
        self._validate_dimension("fusion_dim", fusion_dim)

        self.input_channels = channels
        self.fusion_dim = fusion_dim

        self.feature_projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(channels[name], fusion_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                for name in self.FEATURE_NAMES
            }
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                len(self.FEATURE_NAMES) * fusion_dim,
                fusion_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _validate_dimension(name: str, value: int) -> None:
        """Validate a configured channel dimension."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def _extract_features(self, features: object) -> Dict[str, torch.Tensor]:
        """Return and validate the required CNN feature tensors."""
        if isinstance(features, Mapping):
            feature_dict = features
        else:
            get_all_features = getattr(features, "get_all_features", None)
            if not callable(get_all_features):
                raise TypeError(
                    "features must be a mapping or expose get_all_features()"
                )
            feature_dict = get_all_features()
            if not isinstance(feature_dict, Mapping):
                raise TypeError("get_all_features() must return a feature mapping")

        missing = [name for name in self.FEATURE_NAMES if name not in feature_dict]
        unexpected = [name for name in feature_dict if name not in self.FEATURE_NAMES]
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing required feature maps: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected feature map keys: {unexpected!r}")
            raise ValueError("; ".join(details))

        validated: Dict[str, torch.Tensor] = {}
        batch_size = None
        device = None
        dtype = None
        for name in self.FEATURE_NAMES:
            tensor = feature_dict[name]
            self._validate_tensor(tensor, name, self.input_channels[name])
            if batch_size is None:
                batch_size = tensor.shape[0]
                device = tensor.device
                dtype = tensor.dtype
            else:
                if tensor.shape[0] != batch_size:
                    raise ValueError("all CNN feature maps must have the same batch size")
                if tensor.device != device:
                    raise ValueError("all CNN feature maps must be on the same device")
                if tensor.dtype != dtype:
                    raise ValueError("all CNN feature maps must have the same dtype")
            validated[name] = tensor
        return validated

    def _validate_parameter_compatibility(self, reference: torch.Tensor) -> None:
        """Ensure all convolution parameters match the input tensor."""
        for name, parameter in self.named_parameters():
            module_name = (
                "feature projection"
                if name.startswith("feature_projections.")
                else "fusion module"
            )
            if parameter.device != reference.device:
                raise ValueError(
                    f"{module_name} parameter {name!r} is on {parameter.device}, "
                    f"but input feature tensors are on {reference.device}"
                )
            if parameter.dtype != reference.dtype:
                raise ValueError(
                    f"{module_name} parameter {name!r} has dtype {parameter.dtype}, "
                    f"but input feature tensors have dtype {reference.dtype}"
                )

    @staticmethod
    def _validate_tensor(
        tensor: object, name: str, expected_channels: int
    ) -> None:
        """Validate a finite, floating-point NCHW tensor with expected channels."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape (B, C, H, W); received {tuple(tensor.shape)}"
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

    def forward(self, features: object) -> torch.Tensor:
        """Fuse four modulated CNN feature maps at C2 resolution.

        Args:
            features: A mapping containing ``C2``, ``C3``, ``C4``, and ``C5``
                NCHW tensors, or an object exposing ``get_all_features()``.
                With ResNet-34 defaults, their shapes are
                ``(B, 64, H/4, W/4)``, ``(B, 128, H/8, W/8)``,
                ``(B, 256, H/16, W/16)``, and ``(B, 512, H/32, W/32)``.

        Returns:
            Fused floating-point tensor with shape
            ``(B, fusion_dim, H_C2, W_C2)``.

        Raises:
            TypeError: If a feature is not a tensor or is not floating point.
            ValueError: If feature shapes, channels, batch sizes, devices,
                dtypes, or finite-value requirements are not satisfied.
        """
        feature_dict = self._extract_features(features)
        self._validate_parameter_compatibility(feature_dict["C2"])

        target_size = feature_dict["C2"].shape[-2:]
        aligned = []
        for name in self.FEATURE_NAMES:
            projected = self.feature_projections[name](feature_dict[name])
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(projected)

        return self.fusion(torch.cat(aligned, dim=1))