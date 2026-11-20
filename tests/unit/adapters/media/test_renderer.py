"""Overlay rendering: palette, line width and label chips (D11)."""

from __future__ import annotations

import numpy as np
import pytest

from rdv.adapters.media.renderer import CLASS_COLORS, PillowRenderer, line_width, text_color
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox


def _image(width: int = 200, height: int = 120) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)
    return image


def _detection(code: DamageClass = DamageClass.D00) -> Detection:
    return Detection(code, 0.91, BBox(20.0, 30.0, 120.0, 100.0))


def test_palette_is_the_documented_okabe_ito_set() -> None:
    assert CLASS_COLORS == {
        DamageClass.D00: (0x00, 0x72, 0xB2),
        DamageClass.D10: (0xE6, 0x9F, 0x00),
        DamageClass.D20: (0xCC, 0x79, 0xA7),
        DamageClass.D40: (0xD5, 0x5E, 0x00),
    }


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(100, 100, 2), (600, 600, 2), (1000, 500, 4), (3650, 2044, 15)],
)
def test_line_width_follows_the_formula(width: int, height: int, expected: int) -> None:
    assert line_width(width, height) == expected


@pytest.mark.parametrize("color", list(CLASS_COLORS.values()))
def test_chip_text_has_contrast_on_every_class_color(color: tuple[int, int, int]) -> None:
    red, green, blue = text_color(color)
    assert (red, green, blue) in {(0, 0, 0), (255, 255, 255)}


def test_render_draws_a_box_in_the_class_color_without_mutating_the_input() -> None:
    image = _image()
    original = image.copy()
    detection = _detection(DamageClass.D00)

    rendered = PillowRenderer().render(image, [detection])

    assert rendered.shape == image.shape
    assert rendered.dtype == np.uint8
    assert np.array_equal(image, original)
    assert not np.array_equal(rendered, image)
    assert rendered[30, 60].tolist() == list(CLASS_COLORS[DamageClass.D00])


def test_render_uses_the_color_of_each_class() -> None:
    detections = [
        _detection(DamageClass.D00),
        Detection(DamageClass.D40, 0.8, BBox(150.0, 30.0, 280.0, 100.0)),
    ]

    rendered = PillowRenderer().render(_image(300, 200), detections)

    colors = {tuple(pixel) for pixel in rendered.reshape(-1, 3)}
    assert CLASS_COLORS[DamageClass.D00] in colors
    assert CLASS_COLORS[DamageClass.D40] in colors


def test_render_without_detections_returns_an_identical_copy() -> None:
    image = _image()

    rendered = PillowRenderer().render(image, [])

    assert np.array_equal(rendered, image)
    assert rendered is not image


def test_label_chip_mentions_the_class_and_score() -> None:
    rendered = PillowRenderer().render(_image(), [_detection(DamageClass.D40)])

    # The chip sits above the box and fills pixels that were background before.
    chip_region = rendered[15:28, 20:110]
    assert np.any(np.all(chip_region == CLASS_COLORS[DamageClass.D40], axis=2))


def test_render_rejects_non_rgb_arrays() -> None:
    with pytest.raises(ValueError):
        PillowRenderer().render(np.zeros((10, 10), dtype=np.uint8), [])
