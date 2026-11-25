"""Learned spatial degradation embeddings from multi-scale CNN features."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


class DegradationEstimator(nn.Module):
    """Estimate a spatially varying degradation embedding from CNN features
    and handcrafted spatial degradation maps.

    The estimator accepts either a feature-map object exposing
    ``get_all_features()`` or a dictionary with ``C2``, ``C3``, ``C4``, and
    ``C5`` tensors. Each tensor must have shape ``(B, C_i, H_i, W_i)``. It also
    accepts a four-channel spatial map tensor ordered brightness, contrast,
    sharpness, and noise, with shape ``(B, 4, H, W)``.

    Each CNN scale and the resized spatial maps are projected into a common
    feature dimension at the spatial resolution of ``C2``. Their
    representations are concatenated and processed by learnable convolutions
    to produce an embedding of shape ``(B, embedding_dim, H_C2, W_C2)``. The
    embedding is not a class or severity prediction.
    """

    FEATURE_NAMES = ("C2", "C3", "C4", "C5")
    SPATIAL_MAP_CHANNELS = 4
    SPATIAL_MAP_ORDER = ("brightness", "contrast", "sharpness", "noise")
    DEFAULT_INPUT_CHANNELS = {
        "C2": 64,
        "C3": 128,
        "C4": 256,
        "C5": 512,
    }

    def __init__(
        self,
        embedding_dim: int = 128,
        input_channels: Optional[Mapping[str, int]] = None,
        projection_dim: int = 64,
    ) -> None:
        """Initialize projections and the multi-scale embedding head.

        Args:
            embedding_dim: Number of channels in the output degradation
                embedding.
            input_channels: Expected channel count for each of ``C2``--``C5``.
                Defaults to the ResNet-34 feature dimensions
                ``{C2: 64, C3: 128, C4: 256, C5: 512}``. If supplied, all four
                feature names must be present.
            projection_dim: Common channel count used to project each feature
                scale before alignment and concatenation.

        Raises:
            TypeError: If dimensions are not integers or ``input_channels`` is
                not a mapping.
            ValueError: If a dimension is not positive or the channel mapping
                does not contain exactly the required feature names.
        """
        super().__init__()
        self._validate_dimension("embedding_dim", embedding_dim)
        self._validate_dimension("projection_dim", projection_dim)

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

        for name in self.FEATURE_NAMES:
            self._validate_dimension(f"input_channels[{name!r}]", channels[name])

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.input_channels = channels

        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(channels[name], projection_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                for name in self.FEATURE_NAMES
            }
        )
        self.spatial_map_projection = nn.Sequential(
            nn.Conv2d(
                self.SPATIAL_MAP_CHANNELS,
                projection_dim,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                (len(self.FEATURE_NAMES) + 1) * projection_dim,
                2 * projection_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                2 * projection_dim,
                embedding_dim,
                kernel_size=3,
                padding=1,
            ),
        )

    @staticmethod
    def _validate_dimension(name: str, value: int) -> None:
        """Ensure a configured channel dimension is a positive integer."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def _extract_features(self, features: object) -> Dict[str, torch.Tensor]:
        """Normalize an accepted feature input into a validated dictionary."""
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
        if missing:
            raise ValueError(f"missing required feature maps: {', '.join(missing)}")

        validated: Dict[str, torch.Tensor] = {}
        batch_size = None
        device = None
        floating_dtype = None
        for name in self.FEATURE_NAMES:
            tensor = feature_dict[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"feature {name} must be a torch.Tensor")
            if tensor.ndim != 4:
                raise ValueError(
                    f"feature {name} must have shape (B, C, H, W); "
                    f"received {tuple(tensor.shape)}"
                )
            if tensor.shape[0] <= 0 or tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
                raise ValueError(
                    f"feature {name} must have non-empty batch and spatial dimensions"
                )
            if tensor.shape[1] != self.input_channels[name]:
                raise ValueError(
                    f"feature {name} must have {self.input_channels[name]} channels; "
                    f"received {tensor.shape[1]}"
                )
            if not tensor.is_floating_point():
                raise TypeError(f"feature {name} must have a floating-point dtype")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"feature {name} contains NaN or infinite values")

            if batch_size is None:
                batch_size = tensor.shape[0]
                device = tensor.device
                floating_dtype = tensor.dtype
            elif tensor.shape[0] != batch_size:
                raise ValueError("all feature maps must have the same batch size")
            elif tensor.device != device:
                raise ValueError("all feature maps must be on the same device")
            elif tensor.dtype != floating_dtype:
                raise ValueError(
                    "all feature maps must have the same floating-point dtype; "
                    f"feature {name} has {tensor.dtype}, expected {floating_dtype}"
                )

            validated[name] = tensor
        return validated

    def _validate_parameter_compatibility(self, reference: torch.Tensor) -> None:
        """Ensure all estimator parameters match the C2 input device and dtype."""
        for name, parameter in self.named_parameters():
            if parameter.device != reference.device:
                raise ValueError(
                    f"parameter {name!r} has device {parameter.device}; "
                    f"expected device {reference.device} to match the C2 input"
                )
            if parameter.dtype != reference.dtype:
                raise ValueError(
                    f"parameter {name!r} has dtype {parameter.dtype}; "
                    f"expected dtype {reference.dtype} to match the C2 input"
                )

    def forward(
        self,
        features: object,
        spatial_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Produce a learned spatial degradation embedding.

        Args:
            features: A ``FeatureMaps``-like object with ``get_all_features()``
                or a mapping containing tensors ``C2``, ``C3``, ``C4``, and
                ``C5``. Feature tensor shapes are ``(B, C_i, H_i, W_i)`` and
                their channels must match the configured input dimensions.
            spatial_maps: Floating-point Torch tensor with shape
                ``(B, 4, H, W)``. Channels are ordered brightness, contrast,
                sharpness, and noise. The maps are resized to the spatial
                resolution of ``C2`` using area interpolation. NumPy arrays
                are not accepted.

        Returns:
            Tensor with shape ``(B, embedding_dim, H_C2, W_C2)``. The output is
            a learned feature representation, not a degradation class or
            severity prediction.

        Raises:
            TypeError: If an input is not a supported feature container,
                ``spatial_maps`` is not a Torch tensor, or an input tensor is
                not floating point.
            ValueError: If feature or map shapes, channels, batch sizes,
                devices, dtypes, or finite-value requirements are invalid.
        """
        feature_dict = self._extract_features(features)
        if not isinstance(spatial_maps, torch.Tensor):
            raise TypeError("spatial_maps must be a torch.Tensor")
        if spatial_maps.ndim != 4:
            raise ValueError(
                "spatial_maps must have shape (B, 4, H, W); "
                f"received {tuple(spatial_maps.shape)}"
            )
        if (
            spatial_maps.shape[0] <= 0
            or spatial_maps.shape[2] <= 0
            or spatial_maps.shape[3] <= 0
        ):
            raise ValueError(
                "spatial_maps must have non-empty batch and spatial dimensions"
            )
        if spatial_maps.shape[1] != self.SPATIAL_MAP_CHANNELS:
            raise ValueError(
                "spatial_maps must have four channels ordered "
                "brightness, contrast, sharpness, and noise; "
                f"received {spatial_maps.shape[1]} channels"
            )
        if not spatial_maps.is_floating_point():
            raise TypeError("spatial_maps must have a floating-point dtype")
        if not torch.isfinite(spatial_maps).all().item():
            raise ValueError("spatial_maps contains NaN or infinite values")

        reference = feature_dict["C2"]
        if spatial_maps.shape[0] != reference.shape[0]:
            raise ValueError("spatial_maps and CNN features must have the same batch size")
        if spatial_maps.device != reference.device:
            raise ValueError("spatial_maps and CNN features must be on the same device")
        if spatial_maps.dtype != reference.dtype:
            raise ValueError("spatial_maps and CNN features must have the same dtype")

        self._validate_parameter_compatibility(reference)

        target_size = feature_dict["C2"].shape[-2:]

        aligned = []
        for name in self.FEATURE_NAMES:
            projected = self.projections[name](feature_dict[name])
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(projected)

        if spatial_maps.shape[-2:] != target_size:
            spatial_maps = F.interpolate(
                spatial_maps,
                size=target_size,
                mode="area",
            )
        aligned.append(self.spatial_map_projection(spatial_maps))

        return self.fusion(torch.cat(aligned, dim=1))