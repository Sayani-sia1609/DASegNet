import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from film_module import FiLMModule


def test_zero_initialized_film_is_identity_for_non_square_inputs():
    film = FiLMModule(feature_channels=4, degradation_channels=3)
    features = torch.randn(2, 4, 5, 9)
    embedding = torch.randn(2, 3, 5, 9)

    output = film(features, embedding)

    assert output.shape == features.shape
    assert output.dtype == features.dtype
    torch.testing.assert_close(output, features, rtol=0, atol=0)
    assert torch.count_nonzero(film.conditioner[-1].weight) == 0
    assert torch.count_nonzero(film.conditioner[-1].bias) == 0


def test_nonzero_conditioning_changes_output_and_propagates_gradients():
    film = FiLMModule(feature_channels=4, degradation_channels=3)
    with torch.no_grad():
        film.conditioner[0].weight.fill_(0.01)
        film.conditioner[0].bias.fill_(1.0)
        film.conditioner[-1].weight.fill_(0.001)
        film.conditioner[-1].bias.zero_()
    features = torch.ones(2, 4, 5, 9, requires_grad=True)
    embedding = torch.ones(2, 3, 5, 9, requires_grad=True)

    output = film(features, embedding)
    assert not torch.allclose(output, features)
    output.mean().backward()

    assert features.grad is not None
    assert embedding.grad is not None
    assert torch.count_nonzero(embedding.grad) > 0
    assert torch.count_nonzero(film.conditioner[0].weight.grad) > 0
    assert torch.count_nonzero(film.conditioner[-1].weight.grad) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_channels": 0},
        {"degradation_channels": -1},
        {"feature_channels": True},
        {"degradation_channels": False},
    ],
)
def test_invalid_channel_dimensions_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FiLMModule(**kwargs)


def test_mismatched_spatial_size_batch_dtype_and_nonfinite_values_are_rejected():
    film = FiLMModule(feature_channels=4, degradation_channels=3)
    features = torch.randn(2, 4, 5, 9)

    with pytest.raises(ValueError, match="spatial dimensions"):
        film(features, torch.randn(2, 3, 4, 9))
    with pytest.raises(ValueError, match="batch size"):
        film(features, torch.randn(1, 3, 5, 9))
    with pytest.raises(ValueError, match="same dtype"):
        film(features, torch.randn(2, 3, 5, 9, dtype=torch.float64))

    embedding = torch.randn(2, 3, 5, 9)
    embedding[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="degradation_embedding"):
        film(features, embedding)


def test_film_rejects_non_tensor_non_floating_and_invalid_rank_inputs():
    film = FiLMModule(feature_channels=4, degradation_channels=3)
    embedding = torch.randn(2, 3, 5, 9)

    with pytest.raises(TypeError, match="fused_features must be a torch.Tensor"):
        film(None, embedding)
    with pytest.raises(TypeError, match="floating-point"):
        film(torch.ones(2, 4, 5, 9, dtype=torch.int32), embedding)
    with pytest.raises(ValueError, match="shape"):
        film(torch.ones(2, 4, 5), embedding)


def test_nonfinite_film_output_is_rejected():
    film = FiLMModule(feature_channels=2, degradation_channels=2)
    with torch.no_grad():
        film.conditioner[-1].bias.fill_(float("inf"))

    with pytest.raises(ValueError, match="output contains"):
        film(torch.ones(1, 2, 3, 5), torch.ones(1, 2, 3, 5))
