import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_degradation_map import SpatialDegradationMapper


def test_constant_grayscale_maps_have_correct_shape_dtype_and_values():
    image = np.full((5, 8), 11, dtype=np.uint8)
    maps = SpatialDegradationMapper(window_size=3).generate_maps(image)

    assert tuple(maps) == (
        "brightness_map",
        "contrast_map",
        "sharpness_map",
        "noise_map",
    )
    for result in maps.values():
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.isfinite(result).all()

    np.testing.assert_allclose(maps["brightness_map"], 11.0)
    np.testing.assert_allclose(maps["contrast_map"], 0.0)
    np.testing.assert_allclose(maps["sharpness_map"], 0.0)
    np.testing.assert_allclose(maps["noise_map"], 0.0)


def test_contrast_and_sharpness_maps_detect_a_controlled_edge():
    image = np.zeros((7, 9), dtype=np.float64)
    image[:, 4:] = 10.0

    maps = SpatialDegradationMapper(window_size=3).generate_maps(image)

    assert np.max(maps["contrast_map"]) > 0.0
    assert np.max(maps["sharpness_map"]) > 0.0


def test_brightness_map_uses_replicated_values_at_the_border():
    image = np.array([[0.0, 0.0], [0.0, 9.0]])

    maps = SpatialDegradationMapper(window_size=3).generate_maps(image)

    # Replicating the top and left edges gives nine samples whose sum is 9.
    assert maps["brightness_map"][0, 0] == pytest.approx(1.0)


def test_noise_map_matches_controlled_local_mad_reference():
    rows, columns = np.indices((7, 9))
    image = (((rows * 3 + columns * 7) % 11) ** 2).astype(np.float64)
    window_size = 3
    radius = window_size // 2
    padded_image = np.pad(image, 1, mode="edge")
    response = (
        padded_image[1:-1, :-2]
        + padded_image[1:-1, 2:]
        + padded_image[:-2, 1:-1]
        + padded_image[2:, 1:-1]
        - 4.0 * padded_image[1:-1, 1:-1]
    )
    padded_response = np.pad(response, radius, mode="edge")
    expected = np.empty_like(image)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            local = padded_response[
                row : row + window_size, column : column + window_size
            ]
            median = np.median(local)
            expected[row, column] = np.median(np.abs(local - median)) / (
                0.6744897501960817 * np.sqrt(20.0)
            )

    actual = SpatialDegradationMapper(window_size=window_size).generate_maps(image)[
        "noise_map"
    ]

    assert np.max(expected) > 0.0
    np.testing.assert_allclose(actual, expected)


def test_rgb_non_square_input_preserves_shape_and_intensity_scale():
    image = np.zeros((6, 9, 3), dtype=np.uint8)
    image[..., 1] = 100
    maps = SpatialDegradationMapper(window_size=5).generate_maps(image)

    assert all(result.shape == (6, 9) for result in maps.values())
    np.testing.assert_allclose(maps["brightness_map"], 58.7, atol=1e-5)
    assert np.isfinite(maps["noise_map"]).all()


def test_tiny_image_and_window_larger_than_image_are_supported():
    image = np.array([[3, 7]], dtype=np.uint8)

    maps = SpatialDegradationMapper(window_size=5).generate_maps(image)

    assert all(result.shape == image.shape for result in maps.values())
    assert all(np.isfinite(result).all() for result in maps.values())


def test_stack_maps_preserves_order_and_casts_to_float64():
    maps = {
        "brightness_map": np.full((2, 3), 1, dtype=np.uint8),
        "contrast_map": np.full((2, 3), 2.5, dtype=np.float32),
        "sharpness_map": np.full((2, 3), 3, dtype=np.int16),
        "noise_map": np.full((2, 3), 4.5, dtype=np.float64),
    }

    stacked = SpatialDegradationMapper().stack_maps(maps)

    assert stacked.shape == (4, 2, 3)
    assert stacked.dtype == np.float64
    for channel, value in enumerate((1.0, 2.5, 3.0, 4.5)):
        np.testing.assert_allclose(stacked[channel], value)


@pytest.mark.parametrize(
    "bad_map",
    [
        np.full((2, 3), "text", dtype=object),
        np.full((2, 3), 1 + 2j, dtype=np.complex64),
        np.ones((2, 3), dtype=np.bool_),
    ],
)
def test_stack_maps_rejects_non_real_numeric_values(bad_map):
    maps = {
        name: np.ones((2, 3), dtype=np.float64)
        for name in SpatialDegradationMapper.MAP_NAMES
    }
    maps["noise_map"] = bad_map

    with pytest.raises(TypeError, match="real numeric|complex"):
        SpatialDegradationMapper().stack_maps(maps)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_stack_maps_rejects_nonfinite_values(bad_value):
    maps = {
        name: np.ones((2, 3), dtype=np.float32)
        for name in SpatialDegradationMapper.MAP_NAMES
    }
    maps["sharpness_map"][0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        SpatialDegradationMapper().stack_maps(maps)


def test_stack_maps_rejects_missing_and_inconsistent_maps():
    mapper = SpatialDegradationMapper()
    maps = {name: np.ones((2, 3)) for name in mapper.MAP_NAMES}
    del maps["contrast_map"]
    with pytest.raises(ValueError, match="contrast_map"):
        mapper.stack_maps(maps)

    maps = {name: np.ones((2, 3)) for name in mapper.MAP_NAMES}
    maps["noise_map"] = np.ones((3, 2))
    with pytest.raises(ValueError, match="same height and width"):
        mapper.stack_maps(maps)


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        (None, TypeError),
        (np.zeros((4, 5, 2), dtype=np.uint8), ValueError),
        (np.zeros((4, 5, 3, 1), dtype=np.uint8), ValueError),
        (np.zeros((4, 5), dtype=np.int64), TypeError),
        (np.full((4, 5), np.inf, dtype=np.float32), ValueError),
    ],
)
def test_invalid_images_raise_clear_exceptions(image, exception):
    with pytest.raises(exception):
        SpatialDegradationMapper().generate_maps(image)


@pytest.mark.parametrize("window_size", [0, 2, 4, True, 3.5])
def test_invalid_window_size_is_rejected(window_size):
    with pytest.raises((TypeError, ValueError)):
        SpatialDegradationMapper(window_size=window_size)
