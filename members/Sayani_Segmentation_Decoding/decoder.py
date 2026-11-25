from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .upsampling import Upsample
from .skip_connections import SkipConnection
from .segmentation_head import SegmentationHead


class SegmentationDecoder(nn.Module):
    """Decode ResNet C2--C5 feature maps into per-class segmentation logits.

    The output has shape ``(B, num_classes, H, W)``. ``H`` and ``W`` match
    C2 unless ``output_size`` is provided. Values are raw logits; apply
    ``argmax(dim=1)`` downstream to create a class-ID mask.
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
        num_classes: int = 2,
        input_channels: Optional[Mapping[str, int]] = None,
    ) -> None:
        """Create a decoder for the configured class count and backbone.

        ``input_channels`` defaults to ResNet-18/34 dimensions. For ResNet-50,
        pass ``{"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}``.
        """
        super().__init__()

        self._validate_channel_count("num_classes", num_classes)
        if input_channels is None:
            channels = dict(self.DEFAULT_INPUT_CHANNELS)
        else:
            if not isinstance(input_channels, Mapping):
                raise TypeError("input_channels must map C2, C3, C4, and C5 to counts")
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

        self.upsample = Upsample()
        self.skip4 = SkipConnection(
            in_channels=channels["C5"],
            skip_channels=channels["C4"],
            out_channels=256,
        )
        self.skip3 = SkipConnection(
            in_channels=256,
            skip_channels=channels["C3"],
            out_channels=128,
        )
        self.skip2 = SkipConnection(
            in_channels=128,
            skip_channels=channels["C2"],
            out_channels=64,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            num_classes=num_classes,
        )

    @staticmethod
    def _validate_channel_count(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def _get_features(self, features: object) -> Dict[str, torch.Tensor]:
        if isinstance(features, Mapping):
            feature_map = features
        else:
            get_all_features = getattr(features, "get_all_features", None)
            if not callable(get_all_features):
                raise TypeError(
                    "features must be a mapping or expose get_all_features()"
                )
            feature_map = get_all_features()
            if not isinstance(feature_map, Mapping):
                raise TypeError("get_all_features() must return a feature mapping")

        normalized: Dict[str, torch.Tensor] = {}
        for name in self.FEATURE_NAMES:
            lower_name = name.lower()
            if name in feature_map:
                tensor = feature_map[name]
            elif lower_name in feature_map:
                tensor = feature_map[lower_name]
            else:
                raise ValueError(
                    f"Missing feature map {name!r} (also accepts {lower_name!r})"
                )

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Feature map {name} must be a torch.Tensor")
            if tensor.ndim != 4:
                raise ValueError(
                    f"Feature map {name} must have shape (B, C, H, W); "
                    f"received {tuple(tensor.shape)}"
                )
            if any(size <= 0 for size in tensor.shape):
                raise ValueError(f"Feature map {name} dimensions must be non-empty")
            if tensor.shape[1] != self.input_channels[name]:
                raise ValueError(
                    f"Feature map {name} must have {self.input_channels[name]} "
                    f"channels; received {tensor.shape[1]}"
                )
            if not tensor.is_floating_point():
                raise TypeError(f"Feature map {name} must have a floating-point dtype")

            if normalized:
                reference = normalized["C2"]
                if tensor.shape[0] != reference.shape[0]:
                    raise ValueError("All feature maps must have the same batch size")
                if tensor.device != reference.device:
                    raise ValueError("All feature maps must be on the same device")
                if tensor.dtype != reference.dtype:
                    raise ValueError("All feature maps must have the same dtype")
            normalized[name] = tensor

        return normalized

    def forward(
        self,
        features: object,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return full-class logits from feature maps or a FeatureMaps object.

        Feature mappings may use the extractor's ``C2``--``C5`` keys or the
        lowercase equivalents. By default, logits remain at C2 resolution.
        Set ``output_size`` to the original image's ``(height, width)`` to
        restore input resolution before passing logits to the mask generator.
        """
        if output_size is not None and (
            not isinstance(output_size, (tuple, list, torch.Size))
            or len(output_size) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in output_size
            )
        ):
            raise ValueError("output_size must contain two positive integers")

        feature_map = self._get_features(features)
        c2, c3, c4, c5 = (feature_map[name] for name in self.FEATURE_NAMES)

        for name, parameter in self.named_parameters():
            if parameter.device != c2.device:
                raise ValueError(
                    f"Decoder parameter {name!r} is on {parameter.device}, "
                    f"but feature maps are on {c2.device}"
                )
            if parameter.dtype != c2.dtype:
                raise ValueError(
                    f"Decoder parameter {name!r} has dtype {parameter.dtype}, "
                    f"but feature maps have dtype {c2.dtype}"
                )

        x = self.upsample(c5, target_size=c4.shape[-2:])
        x = self.skip4(x, c4)
        x = self.upsample(x, target_size=c3.shape[-2:])
        x = self.skip3(x, c3)
        x = self.upsample(x, target_size=c2.shape[-2:])
        x = self.skip2(x, c2)

        logits = self.segmentation_head(x)
        if output_size is not None:
            logits = self.upsample(logits, target_size=output_size)

        return logits
