"""Axis-aligned bounding boxes, IoU and exact union area.

The geometry is deliberately image-agnostic: a box is four finite, ordered
coordinates. Operations that need an image boundary take it explicitly and
clip, so no caller can silently produce an out-of-image detection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BBox:
    """A validated axis-aligned box in ``(x1, y1, x2, y2)`` pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("BBox coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("BBox requires x1 <= x2 and y1 <= y2")
        object.__setattr__(self, "x1", float(self.x1))
        object.__setattr__(self, "y1", float(self.y1))
        object.__setattr__(self, "x2", float(self.x2))
        object.__setattr__(self, "y2", float(self.y2))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_degenerate(self) -> bool:
        """True when the box has zero width or height."""
        return self.width <= 0.0 or self.height <= 0.0

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def clipped(self, width: int, height: int) -> BBox:
        """Clamp the box to an image of ``width`` x ``height`` pixels."""
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        return BBox(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )


def _box_array(boxes: Sequence[BBox]) -> NDArray[np.float64]:
    if not boxes:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray([box.to_xyxy() for box in boxes], dtype=np.float64)


def iou_matrix(first: Sequence[BBox], second: Sequence[BBox]) -> NDArray[np.float64]:
    """Pairwise intersection-over-union as a ``(len(first), len(second))`` array.

    IoU is symmetric, lies in ``[0, 1]``, and is ``0`` when either box is
    degenerate (zero area), because the union is then zero as well.
    """
    boxes_a = _box_array(first)
    boxes_b = _box_array(second)
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    ax1 = boxes_a[:, 0][:, np.newaxis]
    ay1 = boxes_a[:, 1][:, np.newaxis]
    ax2 = boxes_a[:, 2][:, np.newaxis]
    ay2 = boxes_a[:, 3][:, np.newaxis]
    bx1 = boxes_b[:, 0][np.newaxis, :]
    by1 = boxes_b[:, 1][np.newaxis, :]
    bx2 = boxes_b[:, 2][np.newaxis, :]
    by2 = boxes_b[:, 3][np.newaxis, :]

    intersection_width = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    intersection_height = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    intersection = intersection_width * intersection_height

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    return np.asarray(iou, dtype=np.float64)


def union_area(boxes: Sequence[BBox], bounds: tuple[int, int] | None = None) -> float:
    """Exact area of the union of ``boxes``, optionally clipped to an image.

    Uses coordinate compression so overlapping boxes are counted once and the
    result is independent of input order. Degenerate boxes contribute nothing.
    """
    prepared: list[BBox] = []
    for box in boxes:
        clipped = box.clipped(*bounds) if bounds is not None else box
        if not clipped.is_degenerate:
            prepared.append(clipped)
    if not prepared:
        return 0.0

    xs = np.unique(np.asarray([value for box in prepared for value in (box.x1, box.x2)]))
    ys = np.unique(np.asarray([value for box in prepared for value in (box.y1, box.y2)]))
    total = 0.0
    for left, right in pairwise(xs):
        x_mid = float(left + right) / 2.0
        for top, bottom in pairwise(ys):
            y_mid = float(top + bottom) / 2.0
            if any(box.x1 <= x_mid <= box.x2 and box.y1 <= y_mid <= box.y2 for box in prepared):
                total += (right - left) * (bottom - top)
    return total
