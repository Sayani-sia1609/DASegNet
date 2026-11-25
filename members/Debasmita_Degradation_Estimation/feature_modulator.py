"""Apply degradation-conditioned FiLM modulation to multi-scale CNN features."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .film_module import FiLMModule
else:
    from film_module import FiLMModule


class FeatureModulator(nn.Module):
    """Modulate C2--C5 CNN features with a shared degradation embedding.

    Each feature level has its own :class:`FiLMModule`, configured for that
    level's channel count and the shared degradation embedding channels. The
    embedding is resized independently to each feature's spatial dimensions.

    The returned dictionary preserves each input feature's shape and contains
    ``C2``, ``C3``, ``C4``, and ``C5`` in that order.
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
        degradation_channels: int = 128,
    ) -> None:
        """Create an independent FiLM module for each CNN feature level.

        Args:
            input_channels: Expected channel counts for ``C2`` through ``C5``.
                Defaults to ResNet-34 values 64, 128, 256, and 512.
            degradation_channels: Number of channels in the spatial
                degradation embedding shared by all four FiLM modules.

        Raises:
            TypeError: If channel counts are not integers or ``input_channels``
                is not a mapping.
            ValueError: If channel counts are not positive or the mapping does
                not define exactly ``C2``, ``C3``, ``C4``, and ``C5``.
        """
        super().__init__()
        self._validate_channel_count("degradation_channels", degradation_channels)
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
            self._validate_channel_count(f"input_channels[{name!r}]", channels[name])

        self.input_channels = channels
        self.degradation_channels = degradation_channels
        self.film_modules = nn.ModuleDict(
            {
                name: FiLMModule(
                    feature_channels=channels[name],
                    degradation_channels=degradation_channels,
                )
                for name in self.FEATURE_NAMES
            }
        )

    @staticmethod
    def _validate_channel_count(name: str, value: int) -> None:
        """Validate a positive integer channel count, excluding booleans."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def _extract_features(self, features: object) -> Dict[str, torch.Tensor]:
        """Return and validate the four NCHW feature tensors."""
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
                    raise ValueError("all feature maps must have the same batch size")
                if tensor.device != device:
                    raise ValueError("all feature maps must be on the same device")
                if tensor.dtype != dtype:
                    raise ValueError("all feature maps must have the same dtype")
            validated[name] = tensor
        return validated

    @staticmethod
    def _validate_tensor(
        tensor: object,
        name: str,
        expected_channels: int,
    ) -> torch.Tensor:
        """Validate an NCHW floating-point tensor with finite values."""
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

    @staticmethod
    def _validate_film_parameter_compatibility(
        film_module: nn.Module,
        feature: torch.Tensor,
        feature_name: str,
    ) -> None:
        """Ensure FiLM parameters match the corresponding input tensor."""
        for parameter_name, parameter in film_module.named_parameters():
            if parameter.device != feature.device:
                raise ValueError(
                    f"FiLM module for {feature_name} parameter {parameter_name} "
                    f"is on device {parameter.device}, but the input is on "
                    f"device {feature.device}"
                )
            if parameter.dtype != feature.dtype:
                raise ValueError(
                    f"FiLM module for {feature_name} parameter {parameter_name} "
                    f"has dtype {parameter.dtype}, but the input has dtype "
                    f"{feature.dtype}"
                )

    def forward(
        self,
        features: object,
        degradation_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Modulate each CNN feature level using the degradation embedding.

        Args:
            features: A mapping containing ``C2``, ``C3``, ``C4``, and ``C5``
                NCHW tensors, or an object exposing ``get_all_features()``.
                With ResNet-34 defaults, channels are 64, 128, 256, and 512.
            degradation_embedding: NCHW degradation representation with shape
                ``(B, degradation_channels, H_D, W_D)``. It is resized with
                bilinear interpolation to each feature level's spatial size.

        Returns:
            Dictionary containing modulated ``C2``, ``C3``, ``C4``, and ``C5``.
            Each tensor has the same shape as its corresponding input feature.

        Raises:
            TypeError: If an input is not a supported feature container or a
                tensor is not floating point.
            ValueError: If feature or embedding dimensions, channels, batch
                sizes, devices, dtypes, or finite-value requirements are
                invalid, or a FiLM output has an invalid shape or values.
        """
        feature_dict = self._extract_features(features)
        degradation_embedding = self._validate_tensor(
            degradation_embedding,
            "degradation_embedding",
            self.degradation_channels,
        )

        reference = feature_dict["C2"]
        if degradation_embedding.shape[0] != reference.shape[0]:
            raise ValueError(
                "CNN features and degradation_embedding must have the same batch size"
            )
        if degradation_embedding.device != reference.device:
            raise ValueError(
                "CNN features and degradation_embedding must be on the same device"
            )
        if degradation_embedding.dtype != reference.dtype:
            raise ValueError(
                "CNN features and degradation_embedding must have the same dtype"
            )

        for name in self.FEATURE_NAMES:
            self._validate_film_parameter_compatibility(
                self.film_modules[name],
                feature_dict[name],
                name,
            )

        modulated_features: Dict[str, torch.Tensor] = {}
        for name in self.FEATURE_NAMES:
            feature = feature_dict[name]
            aligned_embedding = degradation_embedding
            if aligned_embedding.shape[-2:] != feature.shape[-2:]:
                aligned_embedding = F.interpolate(
                    degradation_embedding,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            modulated = self.film_modules[name](feature, aligned_embedding)
            if not isinstance(modulated, torch.Tensor):
                raise TypeError(f"FiLM module for {name} must return a torch.Tensor")
            if modulated.shape != feature.shape:
                raise ValueError(
                    f"FiLM module for {name} must return a tensor with the same "
                    "shape as its input feature"
                )
            if modulated.device != feature.device or modulated.dtype != feature.dtype:
                raise ValueError(
                    f"FiLM module for {name} must preserve the input device and dtype"
                )
            if not torch.isfinite(modulated).all().item():
                raise ValueError(
                    f"FiLM module output for {name} contains NaN or infinite values"
                )
            modulated_features[name] = modulated

        return modulated_features