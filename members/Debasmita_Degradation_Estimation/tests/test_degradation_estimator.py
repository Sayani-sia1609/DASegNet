import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degradation_estimator import DegradationEstimator


def make_features(requires_grad=False):
    return {
        "C2": torch.randn(2, 64, 7, 9, requires_grad=requires_grad),
        "C3": torch.randn(2, 128, 4, 5, requires_grad=requires_grad),
        "C4": torch.randn(2, 256, 2, 3, requires_grad=requires_grad),
        "C5": torch.randn(2, 512, 1, 2, requires_grad=requires_grad),
    }


def make_spatial_maps(
    batch_size=2,
    height=28,
    width=36,
    dtype=torch.float32,
    device="cpu",
    requires_grad=False,
):
    return torch.randn(
        batch_size,
        4,
        height,
        width,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


class FeatureMapsStub:
    def __init__(self, feature_maps):
        self.feature_maps = feature_maps

    def get_all_features(self):
        return self.feature_maps


def test_dictionary_and_feature_maps_inputs_produce_expected_embedding():
    estimator = DegradationEstimator(embedding_dim=24, projection_dim=8)
    features = make_features()
    spatial_maps = make_spatial_maps()

    from_dict = estimator(features, spatial_maps)
    from_object = estimator(FeatureMapsStub(features), spatial_maps)

    assert from_dict.shape == (2, 24, 7, 9)
    assert from_dict.dtype == torch.float32
    assert estimator.SPATIAL_MAP_ORDER == (
        "brightness",
        "contrast",
        "sharpness",
        "noise",
    )
    torch.testing.assert_close(from_dict, from_object)


def test_spatial_maps_are_area_resized_to_c2_resolution():
    estimator = DegradationEstimator(embedding_dim=12, projection_dim=4)
    features = make_features()
    spatial_maps = make_spatial_maps(height=14, width=18)
    projected_inputs = []
    hook = estimator.spatial_map_projection.register_forward_pre_hook(
        lambda _module, args: projected_inputs.append(args[0].detach())
    )
    try:
        output = estimator(features, spatial_maps)
    finally:
        hook.remove()

    assert output.shape == (2, 12, 7, 9)
    assert projected_inputs[0].shape == (2, 4, 7, 9)
    torch.testing.assert_close(
        projected_inputs[0],
        torch.nn.functional.interpolate(
            spatial_maps,
            size=(7, 9),
            mode="area",
        ),
    )


def test_custom_input_channels_and_embedding_dimension():
    channels = {"C2": 4, "C3": 6, "C4": 8, "C5": 10}
    features = {
        "C2": torch.randn(1, 4, 5, 8),
        "C3": torch.randn(1, 6, 3, 4),
        "C4": torch.randn(1, 8, 2, 2),
        "C5": torch.randn(1, 10, 1, 1),
    }

    output = DegradationEstimator(
        embedding_dim=7, input_channels=channels, projection_dim=5
    )(features, make_spatial_maps(batch_size=1, height=20, width=32))

    assert output.shape == (1, 7, 5, 8)


def test_gradients_reach_all_feature_inputs_spatial_maps_and_parameters():
    estimator = DegradationEstimator(embedding_dim=16, projection_dim=8)
    features = make_features(requires_grad=True)
    spatial_maps = make_spatial_maps(requires_grad=True)

    estimator(features, spatial_maps).square().mean().backward()

    assert all(tensor.grad is not None for tensor in features.values())
    assert spatial_maps.grad is not None
    assert torch.count_nonzero(spatial_maps.grad) > 0
    assert estimator.projections["C3"][0].weight.grad is not None
    assert estimator.spatial_map_projection[0].weight.grad is not None
    assert estimator.fusion[0].weight.grad is not None


def test_missing_features_wrong_channels_and_rank_are_rejected():
    estimator = DegradationEstimator()
    features = make_features()
    spatial_maps = make_spatial_maps()
    del features["C4"]
    with pytest.raises(ValueError, match="C4"):
        estimator(features, spatial_maps)

    features = make_features()
    features["C3"] = torch.randn(2, 127, 4, 5)
    with pytest.raises(ValueError, match="channels"):
        estimator(features, spatial_maps)

    features = make_features()
    features["C5"] = torch.randn(2, 512, 1)
    with pytest.raises(ValueError, match="shape"):
        estimator(features, spatial_maps)


def test_batch_dtype_and_finite_value_validation():
    estimator = DegradationEstimator()

    features = make_features()
    spatial_maps = make_spatial_maps()
    features["C4"] = torch.randn(1, 256, 2, 3)
    with pytest.raises(ValueError, match="batch size"):
        estimator(features, spatial_maps)

    features = make_features()
    features["C3"] = features["C3"].double()
    with pytest.raises(ValueError, match="dtype"):
        estimator(features, spatial_maps)

    features = make_features()
    features["C2"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="C2"):
        estimator(features, spatial_maps)

    features = make_features()
    features["C5"][0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="C5"):
        estimator(features, spatial_maps)


def test_non_tensor_and_non_floating_features_are_rejected():
    estimator = DegradationEstimator()
    features = make_features()
    spatial_maps = make_spatial_maps()
    features["C2"] = None
    with pytest.raises(TypeError, match="C2.*torch.Tensor"):
        estimator(features, spatial_maps)

    features = make_features()
    features["C2"] = torch.ones(2, 64, 7, 9, dtype=torch.int32)
    with pytest.raises(TypeError, match="floating-point"):
        estimator(features, spatial_maps)


def test_invalid_spatial_map_rank_channels_and_batch_are_rejected():
    estimator = DegradationEstimator()
    features = make_features()

    with pytest.raises(ValueError, match="shape"):
        estimator(features, torch.randn(2, 4, 28))
    with pytest.raises(ValueError, match="four channels"):
        estimator(features, torch.randn(2, 3, 28, 36))
    with pytest.raises(ValueError, match="same batch size"):
        estimator(features, torch.randn(1, 4, 28, 36))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_spatial_maps_are_rejected(bad_value):
    estimator = DegradationEstimator()
    features = make_features()
    spatial_maps = make_spatial_maps()
    spatial_maps[0, 2, 0, 0] = bad_value

    with pytest.raises(ValueError, match="spatial_maps.*NaN or infinite"):
        estimator(features, spatial_maps)


def test_spatial_map_dtype_mismatch_is_rejected():
    estimator = DegradationEstimator()

    with pytest.raises(ValueError, match="same dtype"):
        estimator(make_features(), make_spatial_maps(dtype=torch.float64))


def test_numpy_spatial_maps_are_rejected():
    estimator = DegradationEstimator()

    with pytest.raises(TypeError, match="spatial_maps must be a torch.Tensor"):
        estimator(make_features(), make_spatial_maps().numpy())


def test_model_parameter_dtype_mismatch_reports_parameter_and_expected_dtype():
    estimator = DegradationEstimator().double()

    with pytest.raises(ValueError) as exc_info:
        estimator(make_features(), make_spatial_maps())

    message = str(exc_info.value)
    assert "parameter 'projections.C2.0.weight'" in message
    assert "dtype torch.float64" in message
    assert "expected dtype torch.float32" in message


def test_model_parameter_device_mismatch_reports_parameter_and_expected_device():
    estimator = DegradationEstimator()
    estimator.spatial_map_projection[0].to(device="meta")

    with pytest.raises(ValueError) as exc_info:
        estimator(make_features(), make_spatial_maps())

    message = str(exc_info.value)
    assert "parameter 'spatial_map_projection.0.weight'" in message
    assert "device meta" in message
    assert "expected device cpu" in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_feature_device_mismatch_is_rejected():
    features = make_features()
    features["C4"] = features["C4"].cuda()

    with pytest.raises(ValueError, match="same device"):
        DegradationEstimator()(features, make_spatial_maps())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_spatial_map_device_mismatch_is_rejected():
    estimator = DegradationEstimator()

    with pytest.raises(ValueError, match="same device"):
        estimator(make_features(), make_spatial_maps(device="cuda"))
