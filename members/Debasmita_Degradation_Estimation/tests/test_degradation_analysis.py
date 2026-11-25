import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degradation_analysis import DegradationAnalyzer


def test_brightness_and_contrast_match_controlled_grayscale_values():
    image = np.tile(np.array([0, 2, 4, 6], dtype=np.uint8), (5, 2))
    analyzer = DegradationAnalyzer()

    assert analyzer.calculate_brightness(image) == pytest.approx(3.0)
    assert analyzer.calculate_contrast(image) == pytest.approx(np.sqrt(5.0))


def test_rgb_brightness_uses_rgb_channel_order():
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    image[..., 0] = 100

    assert DegradationAnalyzer().calculate_brightness(image) == pytest.approx(29.9)


def test_constant_image_has_zero_contrast_sharpness_and_noise():
    image = np.full((7, 9), 12, dtype=np.uint8)
    analyzer = DegradationAnalyzer()

    assert analyzer.calculate_brightness(image) == pytest.approx(12.0)
    assert analyzer.calculate_contrast(image) == pytest.approx(0.0)
    assert analyzer.calculate_sharpness(image) == pytest.approx(0.0)
    assert analyzer.estimate_noise(image) == pytest.approx(0.0)


def test_noise_estimate_matches_controlled_nonconstant_laplacian_mad():
    rows, columns = np.indices((7, 9))
    image = (((rows * 7 + columns * 11) % 19) ** 2).astype(np.float64)

    response = (
        image[1:-1, :-2]
        + image[1:-1, 2:]
        + image[:-2, 1:-1]
        + image[2:, 1:-1]
        - 4.0 * image[1:-1, 1:-1]
    )
    median = np.median(response)
    mad = np.median(np.abs(response - median))
    expected = mad / (0.6744897501960817 * np.sqrt(20.0))

    actual = DegradationAnalyzer().estimate_noise(image)

    assert expected > 0.0
    assert actual == pytest.approx(expected)


def test_sharpness_detects_a_controlled_edge():
    image = np.zeros((7, 9), dtype=np.uint8)
    image[:, 4:] = 255

    assert DegradationAnalyzer().calculate_sharpness(image) > 0.0


def test_analyze_image_returns_finite_python_scalars_and_preserves_input():
    image = np.arange(35, dtype=np.uint8).reshape(5, 7)
    original = image.copy()

    result = DegradationAnalyzer().analyze_image(image)

    assert set(result) == {"brightness", "contrast", "sharpness", "noise"}
    assert all(type(value) is float and np.isfinite(value) for value in result.values())
    np.testing.assert_array_equal(image, original)


def test_noise_and_complete_analysis_reject_images_smaller_than_five_by_five():
    analyzer = DegradationAnalyzer()
    image = np.zeros((4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="5x5"):
        analyzer.estimate_noise(image)
    with pytest.raises(ValueError, match="5x5"):
        analyzer.analyze_image(image)


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        (None, TypeError),
        (np.zeros((3, 3, 2), dtype=np.uint8), ValueError),
        (np.zeros((3, 3, 3, 1), dtype=np.uint8), ValueError),
        (np.zeros((3, 3), dtype=np.int64), TypeError),
        (np.full((3, 3), np.nan, dtype=np.float32), ValueError),
    ],
)
def test_invalid_images_raise_clear_exceptions(image, exception):
    with pytest.raises(exception):
        DegradationAnalyzer().calculate_brightness(image)
