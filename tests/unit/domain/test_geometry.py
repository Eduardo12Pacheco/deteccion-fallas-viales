"""Bounding boxes and IoU/union math are pure, validated geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rdv.domain.geometry import BBox, iou_matrix, union_area


def test_bbox_exposes_width_height_and_area() -> None:
    box = BBox(10.0, 20.0, 30.0, 50.0)

    assert box.width == 20.0
    assert box.height == 30.0
    assert box.area == 600.0
    assert box.to_xyxy() == (10.0, 20.0, 30.0, 50.0)


def test_bbox_normalises_integer_coordinates_to_float() -> None:
    box = BBox(0, 0, 5, 5)  # type: ignore[arg-type]

    assert box.x2 == 5.0
    assert box.width == 5.0
    assert box.area == 25.0


@pytest.mark.parametrize(
    ("coords", "message"),
    [
        ((10.0, 0.0, 5.0, 5.0), "x2"),
        ((0.0, 10.0, 5.0, 5.0), "y2"),
        ((math.nan, 0.0, 5.0, 5.0), "finite"),
        ((0.0, 0.0, math.inf, 5.0), "finite"),
    ],
)
def test_bbox_rejects_invalid_coordinates(
    coords: tuple[float, float, float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BBox(*coords)


def test_bbox_allows_zero_area_boxes() -> None:
    box = BBox(5.0, 5.0, 5.0, 5.0)

    assert box.is_degenerate
    assert box.area == 0.0


def test_bbox_clipped_to_image_bounds() -> None:
    clipped = BBox(-10.0, -5.0, 120.0, 90.0).clipped(100, 80)

    assert clipped == BBox(0.0, 0.0, 100.0, 80.0)


@pytest.mark.parametrize(("width", "height"), [(0, 80), (100, 0), (-1, 80)])
def test_bbox_clipping_requires_positive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        BBox(0, 0, 1, 1).clipped(width, height)


def test_iou_of_identical_boxes_is_one() -> None:
    box = BBox(0.0, 0.0, 10.0, 10.0)

    matrix = iou_matrix([box], [box])

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou_matrix([BBox(0, 0, 1, 1)], [BBox(2, 2, 3, 3)])[0, 0] == 0.0


def test_iou_of_overlapping_boxes_is_hand_computed() -> None:
    # Intersection 5x5 = 25; each box is 10x10 -> union 175.
    matrix = iou_matrix([BBox(0, 0, 10, 10)], [BBox(5, 5, 15, 15)])

    assert matrix[0, 0] == pytest.approx(25 / 175)


def test_iou_matrix_handles_empty_inputs() -> None:
    box = BBox(0, 0, 1, 1)

    assert iou_matrix([], []).shape == (0, 0)
    assert iou_matrix([box], []).shape == (1, 0)
    assert iou_matrix([], [box]).shape == (0, 1)


def test_union_area_counts_overlap_once() -> None:
    boxes = [BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)]

    assert union_area(boxes) == pytest.approx(150.0)


def test_union_area_of_disjoint_and_touching_boxes_adds_up() -> None:
    assert union_area([BBox(0, 0, 1, 1), BBox(1, 1, 2, 2)]) == pytest.approx(2.0)


def test_union_area_is_zero_without_boxes() -> None:
    assert union_area([]) == 0.0


def test_union_area_clips_to_the_image_bounds() -> None:
    assert union_area([BBox(-10, -10, 150, 150)], bounds=(100, 50)) == pytest.approx(5000.0)


def test_union_area_ignores_degenerate_boxes() -> None:
    assert union_area([BBox(1, 1, 1, 1)]) == 0.0


@st.composite
def _non_degenerate_boxes(draw: st.DrawFn) -> BBox:
    x1 = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
    y1 = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
    width = draw(st.floats(min_value=0.001, max_value=1000.0, allow_nan=False))
    height = draw(st.floats(min_value=0.001, max_value=1000.0, allow_nan=False))
    return BBox(x1, y1, x1 + width, y1 + height)


@given(a=_non_degenerate_boxes(), b=_non_degenerate_boxes())
def test_iou_is_symmetric_and_bounded(a: BBox, b: BBox) -> None:
    forward = iou_matrix([a], [b])[0, 0]
    backward = iou_matrix([b], [a])[0, 0]

    assert forward == pytest.approx(backward)
    assert 0.0 <= forward <= 1.0


@given(box=_non_degenerate_boxes())
def test_iou_with_itself_is_one(box: BBox) -> None:
    assert iou_matrix([box], [box])[0, 0] == pytest.approx(1.0)


@given(
    first=st.lists(_non_degenerate_boxes(), min_size=0, max_size=6),
    second=st.lists(_non_degenerate_boxes(), min_size=0, max_size=6),
)
def test_iou_matrix_shape_and_range(first: list[BBox], second: list[BBox]) -> None:
    matrix = iou_matrix(first, second)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(first), len(second))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


@given(boxes=st.lists(_non_degenerate_boxes(), min_size=1, max_size=6))
def test_union_area_is_bounded_by_sum_and_largest_area(boxes: list[BBox]) -> None:
    total = union_area(boxes)

    assert max(box.area for box in boxes) - 1e-6 <= total <= sum(box.area for box in boxes) + 1e-6
