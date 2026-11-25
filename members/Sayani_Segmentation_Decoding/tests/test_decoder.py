import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from members.Debasmita_Degradation_Estimation.degradation_estimator import (
    DegradationEstimator,
)
from members.Debasmita_Degradation_Estimation.feature_modulator import (
    FeatureModulator,
)
from members.Sayani_Segmentation_Decoding.decoder import SegmentationDecoder
from members.Srinik_Final_Segmentation_Mask.evaluation_metrics import (
    SegmentationMetrics,
)
from members.Srinik_Final_Segmentation_Mask.mask_generator import generate_mask


CHANNELS = {"C2": 4, "C3": 6, "C4": 8, "C5": 10}
FEATURE_SIZES = {
    "C2": (7, 9),
    "C3": (4, 5),
    "C4": (2, 3),
    "C5": (1, 2),
}


class FeatureMapsStub:
    def __init__(self, feature_maps):
        self.feature_maps = feature_maps

    def get_all_features(self):
        return self.feature_maps


def make_features():
    return {
        name: torch.randn(2, CHANNELS[name], *FEATURE_SIZES[name])
        for name in ("C2", "C3", "C4", "C5")
    }


def make_decoder():
    decoder = SegmentationDecoder(input_channels=CHANNELS)
    decoder.eval()
    return decoder


def test_logits_are_class_channels_at_c2_or_requested_resolution():
    decoder = make_decoder()
    features = make_features()

    c2_logits = decoder(features)
    full_resolution_logits = decoder(features, output_size=(28, 36))

    assert c2_logits.shape == (2, 2, 7, 9)
    assert full_resolution_logits.shape == (2, 2, 28, 36)
    assert generate_mask(full_resolution_logits).shape == (2, 28, 36)


def test_accepts_feature_maps_object_and_lowercase_feature_keys():
    decoder = make_decoder()
    features = make_features()
    lowercase_features = {name.lower(): value for name, value in features.items()}

    from_mapping = decoder(features)
    from_object = decoder(FeatureMapsStub(features))
    from_lowercase = decoder(lowercase_features)

    torch.testing.assert_close(from_mapping, from_object)
    torch.testing.assert_close(from_mapping, from_lowercase)


def test_rejects_missing_features_and_mismatched_channels():
    decoder = make_decoder()
    features = make_features()

    with pytest.raises(ValueError, match="C4"):
        decoder({name: value for name, value in features.items() if name != "C4"})

    features["C3"] = torch.randn(2, 7, *FEATURE_SIZES["C3"])
    with pytest.raises(ValueError, match="channels"):
        decoder(features)


class TinyPyramidCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            (
                nn.Conv2d(3, CHANNELS["C2"], kernel_size=3, stride=2, padding=1),
                nn.Conv2d(
                    CHANNELS["C2"],
                    CHANNELS["C3"],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    CHANNELS["C3"],
                    CHANNELS["C4"],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    CHANNELS["C4"],
                    CHANNELS["C5"],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
        )

    def forward(self, images):
        features = {}
        x = images
        for name, stage in zip(("C2", "C3", "C4", "C5"), self.stages):
            x = torch.relu(stage(x))
            features[name] = x
        return features


def test_cross_entropy_and_gradients_flow_through_segmentation_pipeline():
    torch.manual_seed(7)
    batch_size = 2
    cnn = TinyPyramidCNN()
    estimator = DegradationEstimator(
        embedding_dim=5,
        input_channels=CHANNELS,
        projection_dim=4,
    )
    modulator = FeatureModulator(
        input_channels=CHANNELS,
        degradation_channels=5,
    )
    decoder = SegmentationDecoder(input_channels=CHANNELS, num_classes=2)

    images = torch.randn(batch_size, 3, 32, 32)
    spatial_maps = torch.rand(batch_size, 4, 32, 32)
    targets = torch.zeros(batch_size, 32, 32, dtype=torch.long)
    targets[:, :, 16:] = 1
    assert set(torch.unique(targets).tolist()) == {0, 1}

    features = cnn(images)
    degradation_embedding = estimator(features, spatial_maps)
    modulated_features = modulator(features, degradation_embedding)
    logits = decoder(modulated_features, output_size=images.shape[-2:])

    assert logits.shape == (batch_size, 2, 32, 32)
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()

    gradients = (
        cnn.stages[0].weight.grad,
        modulator.film_modules["C2"].conditioner[-1].weight.grad,
        decoder.skip2.conv[0].weight.grad,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)

    predicted_mask = generate_mask(logits.detach())
    assert predicted_mask.shape == targets.shape
    assert set(torch.unique(predicted_mask).tolist()) <= {0, 1}


def test_mask_can_contain_both_classes_when_logits_support_both():
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 0, :, :2] = 2
    logits[:, 1, :, 2:] = 2

    predicted_mask = generate_mask(logits)

    assert set(torch.unique(predicted_mask).tolist()) == {0, 1}


def test_binary_segmentation_metrics_and_ground_truth_id_validation():
    targets = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long)
    predictions = torch.tensor([[[0, 1, 1, 1]]], dtype=torch.long)
    metrics = SegmentationMetrics(num_classes=2)

    metrics.update(predictions, targets)
    results = metrics.compute()

    assert results["pixel_accuracy"] == pytest.approx(0.75)
    assert results["mean_iou"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert results["mean_dice"] == pytest.approx((2 / 3 + 0.8) / 2)

    invalid_targets = targets.clone()
    invalid_targets[0, 0, 0] = 2
    with pytest.raises(ValueError, match="invalid class IDs"):
        SegmentationMetrics(num_classes=2).update(predictions, invalid_targets)


def test_integrated_model_returns_input_resolution_logits():
    from integration.model import DegradationAwareSegmentationModel

    model = DegradationAwareSegmentationModel(
        pretrained=False,
        embedding_dim=8,
        projection_dim=4,
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    spatial_maps = torch.rand(2, 4, 32, 32)

    with torch.no_grad():
        logits = model(images, spatial_maps)

    assert logits.shape == (2, 2, 32, 32)
