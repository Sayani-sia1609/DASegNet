"""Integrated degradation-aware semantic segmentation model."""

from typing import Mapping, Optional, Tuple

import torch
from torch import nn

from members.Chiranjib_CNN_Feature_Extraction.feature_extractor import (
    FeatureExtractor,
)
from members.Debasmita_Degradation_Estimation.degradation_estimator import (
    DegradationEstimator,
)
from members.Debasmita_Degradation_Estimation.feature_modulator import (
    FeatureModulator,
)
from members.Sayani_Segmentation_Decoding.decoder import SegmentationDecoder


class DegradationAwareSegmentationModel(nn.Module):
    """Run backbone, degradation estimation, feature modulation, and decoding.

    The model returns raw segmentation logits at input resolution by default,
    ready for ``CrossEntropyLoss`` with targets shaped ``(B, H, W)``.
    """

    BACKBONE_CHANNELS: Mapping[str, Mapping[str, int]] = {
        "resnet18": {"C2": 64, "C3": 128, "C4": 256, "C5": 512},
        "resnet34": {"C2": 64, "C3": 128, "C4": 256, "C5": 512},
        "resnet50": {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048},
    }

    def __init__(
        self,
        arch: str = "resnet34",
        num_classes: int = 2,
        pretrained: bool = False,
        trainable_backbone: bool = True,
        embedding_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if arch not in self.BACKBONE_CHANNELS:
            raise ValueError(
                f"Unsupported architecture {arch!r}; "
                f"choose from {tuple(self.BACKBONE_CHANNELS)}"
            )

        input_channels = dict(self.BACKBONE_CHANNELS[arch])
        self.feature_extractor = FeatureExtractor(
            arch=arch,
            pretrained=pretrained,
            trainable=trainable_backbone,
        )
        self.degradation_estimator = DegradationEstimator(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            projection_dim=projection_dim,
        )
        self.feature_modulator = FeatureModulator(
            input_channels=input_channels,
            degradation_channels=embedding_dim,
        )
        self.decoder = SegmentationDecoder(
            num_classes=num_classes,
            input_channels=input_channels,
        )

    def forward(
        self,
        images: torch.Tensor,
        spatial_maps: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return segmentation logits for an image batch.

        Args:
            images: RGB image tensor shaped ``(B, 3, H, W)``.
            spatial_maps: Four degradation maps shaped ``(B, 4, H_D, W_D)``
                ordered brightness, contrast, sharpness, and noise.
            output_size: Optional ``(height, width)`` for logits. Defaults to
                the input image resolution.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images must have shape (B, 3, H, W); got {tuple(images.shape)}"
            )
        if output_size is None:
            output_size = images.shape[-2:]

        features = self.feature_extractor(images).get_all_features()
        degradation_embedding = self.degradation_estimator(features, spatial_maps)
        modulated_features = self.feature_modulator(
            features,
            degradation_embedding,
        )
        return self.decoder(modulated_features, output_size=output_size)