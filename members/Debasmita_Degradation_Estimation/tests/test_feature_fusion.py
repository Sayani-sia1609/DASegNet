import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_fusion import FeatureFusion


CHANNELS = {"C2": 4, "C3": 6, "C4": 8, "C5": 10}
FEATURE_SIZES = {
    "C2": (7, 9),
    "C3": (4, 5),
    "C4": (2, 3),
    "C5": (1, 2),
}
FEATURE_NAMES = ("C2", "C3", "C4", "C5")


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
        for name in FEATURE_NAMES
    }


class FeatureMapsStub:
    def __init__(self, feature_maps):
        self.feature_maps = feature_maps

    def get_all_features(self):
        return self.feature_maps


def make_fusion(channels=None, fusion_dim=8):
    return FeatureFusion(
        input_channels=channels or CHANNELS,
        fusion_dim=fusion_dim,
    )


def test_output_shape_and_mapping_and_get_all_features_inputs():
    fusion = make_fusion()
    features = make_features()

    from_mapping = fusion(features)
    from_object = fusion(FeatureMapsStub(features))

    assert from_mapping.shape == (2, 8, 7, 9)
    assert from_mapping.dtype == torch.float32
    torch.testing.assert_close(from_mapping, from_object)


def test_each_feature_level_is_projected_to_fusion_dim():
    fusion = make_fusion()
    features = make_features()
    projected = {}
    handles = []

    with torch.no_grad():
        for name in FEATURE_NAMES:
            convolution = fusion.feature_projections[name][0]
            convolution.weight.zero_()
            convolution.bias.fill_(1.0)
            for output_channel in range(fusion.fusion_dim):
                input_channel = output_channel % CHANNELS[name]
                convolution.weight[output_channel, input_channel, 0, 0] = 2.0

    for name in FEATURE_NAMES:
        def capture_projection(_module, _inputs, output, feature_name=name):
            projected[feature_name] = output.detach()

        handles.append(
            fusion.feature_projections[name].register_forward_hook(capture_projection)
        )

    try:
        fusion(features)
    finally:
        for handle in handles:
            handle.remove()

    for name in FEATURE_NAMES:
        expected = torch.stack(
            [
                2.0 * features[name][:, channel % CHANNELS[name]] + 1.0
                for channel in range(fusion.fusion_dim)
            ],
            dim=1,
        ).relu()
        assert projected[name].shape == (
            2,
            fusion.fusion_dim,
            *FEATURE_SIZES[name],
        )
        torch.testing.assert_close(projected[name], expected)


def test_bilinear_alignment_concatenation_order_and_fusion():
    fusion = make_fusion()
    features = make_features()
    projected = {}
    fusion_input = {}
    handles = []

    for name in FEATURE_NAMES:
        def capture_projection(_module, _inputs, output, feature_name=name):
            projected[feature_name] = output.detach()

        handles.append(
            fusion.feature_projections[name].register_forward_hook(capture_projection)
        )

    def capture_fusion_input(_module, args):
        fusion_input["tensor"] = args[0].detach()

    handles.append(
        fusion.fusion.register_forward_pre_hook(capture_fusion_input)
    )

    try:
        output = fusion(features)
    finally:
        for handle in handles:
            handle.remove()

    target_size = FEATURE_SIZES["C2"]
    assert fusion_input["tensor"].shape == (2, 4 * fusion.fusion_dim, *target_size)
    for index, name in enumerate(FEATURE_NAMES):
        expected = projected[name]
        if expected.shape[-2:] != target_size:
            expected = torch.nn.functional.interpolate(
                expected,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        actual = fusion_input["tensor"][
            :, index * fusion.fusion_dim : (index + 1) * fusion.fusion_dim
        ]
        torch.testing.assert_close(actual, expected)

    torch.testing.assert_close(output, fusion.fusion(fusion_input["tensor"]))
    assert output.shape == (2, fusion.fusion_dim, *target_size)


def test_custom_input_channels_and_fusion_dimension():
    channels = {"C2": 2, "C3": 3, "C4": 5, "C5": 7}
    sizes = {
        "C2": (9, 11),
        "C3": (5, 6),
        "C4": (3, 3),
        "C5": (2, 2),
    }
    fusion = make_fusion(channels=channels, fusion_dim=12)
    output = fusion(make_features(channels=channels, sizes=sizes))

    assert output.shape == (2, 12, 9, 11)
    for name in FEATURE_NAMES:
        assert fusion.feature_projections[name][0].in_channels == channels[name]
        assert fusion.feature_projections[name][0].out_channels == 12
    assert fusion.fusion[0].in_channels == 4 * 12
    assert fusion.fusion[0].out_channels == 12


def test_all_feature_and_fusion_parameters_receive_gradients():
    fusion = make_fusion()
    with torch.no_grad():
        for parameter in fusion.parameters():
            parameter.fill_(0.01)
        for module in fusion.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.bias.fill_(1.0)

    features = {
        name: torch.ones(
            2,
            CHANNELS[name],
            *FEATURE_SIZES[name],
            requires_grad=True,
        )
        for name in FEATURE_NAMES
    }
    fusion(features).square().mean().backward()

    assert all(feature.grad is not None for feature in features.values())
    assert all(torch.count_nonzero(feature.grad) > 0 for feature in features.values())
    for name in FEATURE_NAMES:
        assert fusion.feature_projections[name][0].weight.grad is not None
        assert torch.count_nonzero(
            fusion.feature_projections[name][0].weight.grad
        ) > 0
    assert all(parameter.grad is not None for parameter in fusion.fusion.parameters())
    assert all(
        torch.count_nonzero(parameter.grad) > 0
        for parameter in fusion.fusion.parameters()
    )


def test_missing_unexpected_keys_and_invalid_feature_containers_are_rejected():
    fusion = make_fusion()
    features = make_features()

    missing = dict(features)
    del missing["C3"]
    with pytest.raises(ValueError, match="missing required feature maps.*C3"):
        fusion(missing)

    extra = {**features, "extra": torch.randn(2, 1, 1, 1)}
    with pytest.raises(ValueError, match="unexpected feature map keys.*extra"):
        fusion(extra)

    with pytest.raises(TypeError, match="mapping or expose get_all_features"):
        fusion(None)
    with pytest.raises(TypeError, match="get_all_features.*feature mapping"):
        fusion(FeatureMapsStub(None))


@pytest.mark.parametrize(
    ("replacement", "exception", "message"),
    [
        (torch.randn(2, 4, 7), ValueError, "shape"),
        (torch.randn(2, 5, 7, 9), ValueError, "channels"),
        (torch.randn(2, 4, 0, 9), ValueError, "non-empty"),
        ("not a tensor", TypeError, "torch.Tensor"),
    ],
)
def test_invalid_feature_ranks_channels_dimensions_and_types_are_rejected(
    replacement, exception, message
):
    features = make_features()
    features["C2"] = replacement

    with pytest.raises(exception, match=message):
        make_fusion()(features)


def test_feature_batch_size_dtype_and_finite_values_are_validated():
    fusion = make_fusion()
    features = make_features()
    features["C4"] = torch.randn(1, CHANNELS["C4"], *FEATURE_SIZES["C4"])
    with pytest.raises(ValueError, match="same batch size"):
        fusion(features)

    features = make_features()
    features["C3"] = features["C3"].double()
    with pytest.raises(ValueError, match="same dtype"):
        fusion(features)

    features = make_features()
    features["C5"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="C5.*NaN or infinite"):
        fusion(features)


def test_integer_feature_tensors_are_rejected():
    features = make_features()
    features["C2"] = torch.ones(
        2, CHANNELS["C2"], *FEATURE_SIZES["C2"], dtype=torch.int32
    )

    with pytest.raises(TypeError, match="C2 must have a floating-point dtype"):
        make_fusion()(features)


@pytest.mark.parametrize(
    ("module_name", "message"),
    [
        ("feature_projection", "feature projection parameter"),
        ("fusion_module", "fusion module parameter"),
    ],
)
def test_parameter_dtype_mismatches_are_rejected(module_name, message):
    fusion = make_fusion()
    if module_name == "feature_projection":
        fusion.feature_projections["C2"][0].double()
    else:
        fusion.fusion[0].double()

    with pytest.raises(ValueError, match=f"{message}.*dtype"):
        fusion(make_features())


@pytest.mark.parametrize(
    ("module_name", "message"),
    [
        ("feature_projection", "feature projection parameter"),
        ("fusion_module", "fusion module parameter"),
    ],
)
def test_parameter_device_mismatches_are_rejected(module_name, message):
    fusion = make_fusion()
    if module_name == "feature_projection":
        fusion.feature_projections["C2"][0].to(device="meta")
    else:
        fusion.fusion[0].to(device="meta")

    with pytest.raises(ValueError, match=f"{message}.*on meta"):
        fusion(make_features())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_feature_device_mismatch_is_rejected():
    features = make_features()
    features["C4"] = features["C4"].cuda()

    with pytest.raises(ValueError, match="same device"):
        make_fusion()(features)


def test_invalid_configuration_values_are_rejected():
    with pytest.raises(TypeError, match="input_channels must be a mapping"):
        FeatureFusion(input_channels=[4, 6, 8, 10])

    with pytest.raises(ValueError, match="missing.*C5"):
        FeatureFusion(input_channels={"C2": 4, "C3": 6, "C4": 8})
    with pytest.raises(ValueError, match="unexpected.*extra"):
        FeatureFusion(input_channels={**CHANNELS, "extra": 1})
    with pytest.raises(TypeError, match=r"input_channels\['C2'\].*integer"):
        FeatureFusion(input_channels={**CHANNELS, "C2": 4.0})
    with pytest.raises(ValueError, match=r"input_channels\['C2'\].*positive"):
        FeatureFusion(input_channels={**CHANNELS, "C2": 0})
    with pytest.raises(TypeError, match="fusion_dim must be an integer"):
        FeatureFusion(input_channels=CHANNELS, fusion_dim=8.0)
    with pytest.raises(ValueError, match="fusion_dim must be positive"):
        FeatureFusion(input_channels=CHANNELS, fusion_dim=0)
