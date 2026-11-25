import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_modulator import FeatureModulator
from film_module import FiLMModule


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


class WrongShapeFiLM(torch.nn.Module):
    def forward(self, fused_features, degradation_embedding):
        return fused_features[:, :, :-1, :]


class NonTensorFiLM(torch.nn.Module):
    def forward(self, fused_features, degradation_embedding):
        return None


class NonFiniteFiLM(torch.nn.Module):
    def forward(self, fused_features, degradation_embedding):
        return torch.full_like(fused_features, float("nan"))


def make_features(requires_grad=False, channels=None, sizes=None, dtype=torch.float32):
    channels = channels or CHANNELS
    sizes = sizes or FEATURE_SIZES
    return {
        name: torch.randn(
            2,
            channels[name],
            *sizes[name],
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for name in ("C2", "C3", "C4", "C5")
    }


def make_modulator(channels=None, degradation_channels=5):
    return FeatureModulator(
        input_channels=channels or CHANNELS,
        degradation_channels=degradation_channels,
    )


def test_output_keys_and_each_feature_shape_are_preserved():
    modulator = make_modulator()
    features = make_features()
    embedding = torch.randn(2, 5, 3, 4)

    output = modulator(features, embedding)

    assert tuple(output) == ("C2", "C3", "C4", "C5")
    for name in output:
        assert output[name].shape == features[name].shape
        assert output[name].dtype == features[name].dtype
        torch.testing.assert_close(output[name], features[name])


def test_dictionary_and_feature_maps_style_inputs_are_equivalent():
    modulator = make_modulator()
    features = make_features()
    embedding = torch.randn(2, 5, 3, 4)

    from_dict = modulator(features, embedding)
    from_object = modulator(FeatureMapsStub(features), embedding)

    assert tuple(from_dict) == tuple(from_object)
    for name in from_dict:
        torch.testing.assert_close(from_dict[name], from_object[name])


def test_four_independent_film_modules_match_level_channels():
    modulator = make_modulator()

    assert tuple(modulator.film_modules) == ("C2", "C3", "C4", "C5")
    assert all(
        isinstance(module, FiLMModule)
        for module in modulator.film_modules.values()
    )
    assert len({id(module) for module in modulator.film_modules.values()}) == 4
    for name, module in modulator.film_modules.items():
        assert module.feature_channels == CHANNELS[name]
        assert module.degradation_channels == 5

    parameters = [
        module.conditioner[0].weight
        for module in modulator.film_modules.values()
    ]
    assert len({parameter.data_ptr() for parameter in parameters}) == 4


def test_embedding_is_bilinearly_resized_for_each_feature_level():
    modulator = make_modulator()
    features = make_features()
    embedding = torch.randn(2, 5, 3, 4)
    aligned_inputs = {}
    handles = []

    for name, film_module in modulator.film_modules.items():
        def capture_embedding(_module, args, feature_name=name):
            aligned_inputs[feature_name] = args[1].detach()

        handles.append(
            film_module.register_forward_pre_hook(capture_embedding)
        )

    try:
        modulator(features, embedding)
    finally:
        for handle in handles:
            handle.remove()

    for name, (height, width) in FEATURE_SIZES.items():
        assert aligned_inputs[name].shape == (2, 5, height, width)
        expected = torch.nn.functional.interpolate(
            embedding,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        torch.testing.assert_close(aligned_inputs[name], expected)


def test_gradients_reach_all_features_embedding_and_each_film_module():
    modulator = make_modulator()
    for film_module in modulator.film_modules.values():
        with torch.no_grad():
            film_module.conditioner[0].weight.fill_(0.01)
            film_module.conditioner[0].bias.fill_(1.0)
            film_module.conditioner[-1].weight.fill_(0.001)
            film_module.conditioner[-1].bias.zero_()

    features = {
        name: torch.ones(
            2,
            CHANNELS[name],
            *FEATURE_SIZES[name],
            requires_grad=True,
        )
        for name in ("C2", "C3", "C4", "C5")
    }
    embedding = torch.ones(2, 5, 3, 4, requires_grad=True)
    output = modulator(features, embedding)
    assert any(
        not torch.allclose(output[name], features[name])
        for name in output
    )

    sum(value.square().mean() for value in output.values()).backward()

    assert all(feature.grad is not None for feature in features.values())
    assert all(torch.count_nonzero(feature.grad) > 0 for feature in features.values())
    assert embedding.grad is not None
    assert torch.count_nonzero(embedding.grad) > 0
    for film_module in modulator.film_modules.values():
        assert torch.count_nonzero(film_module.conditioner[-1].weight.grad) > 0


def test_custom_channel_configuration_is_supported():
    channels = {"C2": 2, "C3": 3, "C4": 4, "C5": 5}
    sizes = {
        "C2": (9, 11),
        "C3": (5, 6),
        "C4": (3, 3),
        "C5": (2, 2),
    }
    modulator = make_modulator(channels=channels, degradation_channels=7)
    features = make_features(channels=channels, sizes=sizes)
    embedding = torch.randn(2, 7, 4, 5)

    output = modulator(features, embedding)

    for name in channels:
        assert output[name].shape == features[name].shape
        assert modulator.film_modules[name].feature_channels == channels[name]
        assert modulator.film_modules[name].degradation_channels == 7


def test_missing_features_invalid_container_rank_and_channels_are_rejected():
    modulator = make_modulator()
    embedding = torch.randn(2, 5, 3, 4)
    features = make_features()
    del features["C4"]
    with pytest.raises(ValueError, match="C4"):
        modulator(features, embedding)

    with pytest.raises(TypeError, match="mapping or expose get_all_features"):
        modulator(None, embedding)

    features = make_features()
    features["C2"] = torch.randn(2, CHANNELS["C2"], 7)
    with pytest.raises(ValueError, match="shape"):
        modulator(features, embedding)

    features = make_features()
    features["C3"] = torch.randn(2, CHANNELS["C3"] + 1, 4, 5)
    with pytest.raises(ValueError, match="channels"):
        modulator(features, embedding)


def test_feature_batch_dtype_device_and_nonfinite_validation():
    modulator = make_modulator()
    embedding = torch.randn(2, 5, 3, 4)

    features = make_features()
    features["C4"] = torch.randn(1, CHANNELS["C4"], 2, 3)
    with pytest.raises(ValueError, match="same batch size"):
        modulator(features, embedding)

    features = make_features()
    features["C3"] = features["C3"].double()
    with pytest.raises(ValueError, match="same dtype"):
        modulator(features, embedding)

    features = make_features()
    features["C2"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="C2.*NaN or infinite"):
        modulator(features, embedding)


def test_embedding_type_rank_channels_batch_dtype_and_finite_validation():
    modulator = make_modulator()
    features = make_features()

    with pytest.raises(TypeError, match="degradation_embedding must be a torch.Tensor"):
        modulator(features, None)
    with pytest.raises(ValueError, match="shape"):
        modulator(features, torch.randn(2, 5, 3))
    with pytest.raises(ValueError, match="channels"):
        modulator(features, torch.randn(2, 4, 3, 4))
    with pytest.raises(ValueError, match="same batch size"):
        modulator(features, torch.randn(1, 5, 3, 4))
    with pytest.raises(ValueError, match="same dtype"):
        modulator(features, torch.randn(2, 5, 3, 4, dtype=torch.float64))

    embedding = torch.randn(2, 5, 3, 4)
    embedding[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="degradation_embedding.*NaN or infinite"):
        modulator(features, embedding)


def test_integer_feature_and_embedding_tensors_are_rejected():
    modulator = make_modulator()
    features = make_features()
    embedding = torch.randn(2, 5, 3, 4)
    features["C2"] = torch.ones(2, CHANNELS["C2"], *FEATURE_SIZES["C2"], dtype=torch.int32)

    with pytest.raises(TypeError, match="C2 must have a floating-point dtype"):
        modulator(features, embedding)

    with pytest.raises(TypeError, match="degradation_embedding must have a floating-point dtype"):
        modulator(make_features(), torch.ones(2, 5, 3, 4, dtype=torch.int32))


def test_film_parameter_dtype_and_device_mismatches_are_rejected():
    features = make_features()
    embedding = torch.randn(2, 5, 3, 4)

    modulator = make_modulator()
    modulator.film_modules["C2"].to(dtype=torch.float64)
    with pytest.raises(ValueError, match="parameter.*dtype"):
        modulator(features, embedding)

    modulator = make_modulator()
    modulator.film_modules["C3"].to(device="meta")
    with pytest.raises(ValueError, match="parameter.*device"):
        modulator(features, embedding)


@pytest.mark.parametrize(
    ("film_module", "exception", "message"),
    [
        (WrongShapeFiLM(), ValueError, "same shape"),
        (NonTensorFiLM(), TypeError, "torch.Tensor"),
        (NonFiniteFiLM(), ValueError, "NaN or infinite"),
    ],
)
def test_modulator_validates_film_outputs(film_module, exception, message):
    modulator = make_modulator()
    modulator.film_modules["C2"] = film_module

    with pytest.raises(exception, match=message):
        modulator(make_features(), torch.randn(2, 5, 3, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_feature_and_embedding_device_mismatches_are_rejected():
    modulator = make_modulator()
    features = make_features()

    features["C4"] = features["C4"].cuda()
    with pytest.raises(ValueError, match="same device"):
        modulator(features, torch.randn(2, 5, 3, 4))

    features = make_features()
    with pytest.raises(ValueError, match="same device"):
        modulator(features, torch.randn(2, 5, 3, 4, device="cuda"))
