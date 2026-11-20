"""Preprocessing contract for the ONNX graph (decision D8)."""

from __future__ import annotations

import numpy as np
import pytest

from rdv.adapters.inference.preprocess import INPUT_SIZE, preprocess_image


def _gradient(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :]
    image[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, np.newaxis]
    image[..., 2] = 42
    return image


def test_tensor_follows_the_contract_shape_dtype_and_range() -> None:
    tensor = preprocess_image(_gradient(37, 53))

    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_channels_keep_rgb_order_and_rescale_by_255() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30

    tensor = preprocess_image(image, size=16)

    expected = np.array([10.0, 20.0, 30.0], dtype=np.float32) / np.float32(255.0)
    for channel, value in enumerate(expected):
        assert tensor[0, channel] == pytest.approx(value, abs=1e-6)


def test_odd_aspect_ratios_are_stretched_to_a_square() -> None:
    tensor = preprocess_image(_gradient(100, 10), size=8)

    assert tensor.shape == (1, 3, 8, 8)


def test_center_region_matches_bilinear_downscale() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[320:, :, :] = 255

    tensor = preprocess_image(image)

    # A hard edge at the exact middle of a 640 -> 640 resize stays at row 320.
    assert tensor[0, 0, 319, 0] == pytest.approx(0.0)
    assert tensor[0, 0, 320, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float64),
    ],
)
def test_invalid_images_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        preprocess_image(image)


def test_invalid_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        preprocess_image(_gradient(10, 10), size=0)
