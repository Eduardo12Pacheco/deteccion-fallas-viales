"""Draw detections with the Okabe-Ito palette and score chips (D11, S5).

The renderer is deliberately simple and deterministic: a rectangle per box
plus a label chip with ``CODE score``. Fonts come from Pillow's bundled default
so no font file is downloaded or bundled beyond the wheel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from rdv.application.ports import RgbImage
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection

CLASS_COLORS: dict[DamageClass, tuple[int, int, int]] = {
    DamageClass.D00: (0x00, 0x72, 0xB2),
    DamageClass.D10: (0xE6, 0x9F, 0x00),
    DamageClass.D20: (0xCC, 0x79, 0xA7),
    DamageClass.D40: (0xD5, 0x5E, 0x00),
}
"""Okabe-Ito colorblind-safe palette from decision D11."""

_CHIP_PADDING = 4


def line_width(width: int, height: int) -> int:
    """Box outline width for an image, ``max(2, round(0.004 * max(w, h)))``."""
    return max(2, round(0.004 * max(width, height)))


def _relative_luminance(color: tuple[int, int, int]) -> float:
    channels = []
    for value in color:
        normalized = value / 255.0
        channels.append(
            normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text for maximum contrast on a class chip."""
    return (0, 0, 0) if _relative_luminance(background) > 0.35 else (255, 255, 255)


class PillowRenderer:
    """Renderer adapter that draws boxes and label chips (``Renderer`` port)."""

    def _font(self, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # pragma: no cover - Pillow < 10.1 has no size argument
            return ImageFont.load_default()

    def render(
        self,
        image: NDArray[np.uint8],
        detections: Sequence[Detection],
    ) -> RgbImage:
        """Return a copy of ``image`` with every detection drawn on top."""
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Expected an HxWx3 uint8 RGB image, got {array.shape}/{array.dtype}")
        height, width = int(array.shape[0]), int(array.shape[1])
        canvas = Image.fromarray(array).convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        outline = line_width(width, height)
        font = self._font(max(12, min(28, round(max(width, height) / 45))))
        for detection in detections:
            color = CLASS_COLORS[detection.damage_class]
            x1, y1, x2, y2 = detection.bbox.to_xyxy()
            draw.rectangle((x1, y1, x2, y2), outline=color, width=outline)
            label = f"{detection.damage_class.code} {detection.score:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            chip_width = right - left + 2 * _CHIP_PADDING
            chip_height = bottom - top + 2 * _CHIP_PADDING
            chip_top = y1 - chip_height - outline
            if chip_top < 0:
                chip_top = y1 + outline
            chip_left = min(max(x1, 0.0), max(width - chip_width, 0.0))
            draw.rectangle(
                (chip_left, chip_top, chip_left + chip_width, chip_top + chip_height),
                fill=color,
            )
            draw.text(
                (chip_left + _CHIP_PADDING, chip_top + _CHIP_PADDING),
                label,
                fill=text_color(color),
                font=font,
            )
        return np.asarray(canvas, dtype=np.uint8)
