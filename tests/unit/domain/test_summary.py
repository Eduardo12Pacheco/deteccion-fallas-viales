"""Damage coverage ratio and per-image summaries (decision D21)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.summary import DamageSummary, damage_coverage


def test_coverage_is_zero_without_boxes() -> None:
    assert damage_coverage([], (600, 600)) == 0.0


def test_coverage_of_a_full_image_box_is_one() -> None:
    assert damage_coverage([BBox(0, 0, 600, 400)], (600, 400)) == pytest.approx(1.0)


def test_coverage_counts_overlap_once() -> None:
    coverage = damage_coverage([BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)], (20, 20))

    assert coverage == pytest.approx(150 / 400)


def test_coverage_clips_boxes_to_the_image() -> None:
    coverage = damage_coverage([BBox(-50, -50, 700, 700)], (600, 400))

    assert coverage == pytest.approx(1.0)


@pytest.mark.parametrize("image_size", [(0, 400), (600, 0), (-1, 400)])
def test_coverage_rejects_invalid_image_sizes(image_size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="image dimensions"):
        damage_coverage([BBox(0, 0, 10, 10)], image_size)


def test_summary_from_detections_counts_every_class() -> None:
    detections = [
        Detection(DamageClass.D00, 0.9, BBox(0, 0, 10, 10)),
        Detection(DamageClass.D00, 0.4, BBox(20, 20, 30, 30)),
        Detection(DamageClass.D40, 0.7, BBox(40, 40, 50, 50)),
    ]

    summary = DamageSummary.from_detections(detections, (100, 100))

    assert summary.counts == {
        DamageClass.D00: 2,
        DamageClass.D10: 0,
        DamageClass.D20: 0,
        DamageClass.D40: 1,
    }
    assert summary.total == 3
    assert summary.max_score == pytest.approx(0.9)
    assert summary.coverage_ratio == pytest.approx(300 / 10_000)


def test_summary_without_detections_is_empty() -> None:
    summary = DamageSummary.from_detections([], (100, 100))

    assert summary.total == 0
    assert summary.max_score == 0.0
    assert summary.coverage_ratio == 0.0
    assert all(count == 0 for count in summary.counts.values())


def test_summary_normalises_partial_counts() -> None:
    summary = DamageSummary(counts={DamageClass.D20: 2}, max_score=0.5, coverage_ratio=0.25)

    assert summary.counts[DamageClass.D20] == 2
    assert summary.counts[DamageClass.D00] == 0
    assert summary.total == 2


def test_summary_counts_are_read_only() -> None:
    summary = DamageSummary(counts={DamageClass.D00: 1}, max_score=0.5)

    assert isinstance(summary.counts, Mapping)
    with pytest.raises(TypeError):
        summary.counts[DamageClass.D00] = 5  # type: ignore[index]


@pytest.mark.parametrize(
    ("counts", "max_score", "coverage", "message"),
    [
        ({DamageClass.D00: -1}, 0.5, 0.0, "counts"),
        ({DamageClass.D00: 1}, 1.1, 0.0, "max_score"),
        ({DamageClass.D00: 1}, -0.1, 0.0, "max_score"),
        ({DamageClass.D00: 1}, math.nan, 0.0, "max_score"),
        ({DamageClass.D00: 1}, 0.5, 1.1, "coverage_ratio"),
        ({DamageClass.D00: 1}, 0.5, math.nan, "coverage_ratio"),
        ({DamageClass.D00: 1, "D99": 1}, 0.5, 0.0, "unknown class"),  # type: ignore[dict-item]
    ],
)
def test_summary_rejects_invalid_values(
    counts: Mapping[DamageClass, int],
    max_score: float,
    coverage: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DamageSummary(counts=counts, max_score=max_score, coverage_ratio=coverage)


@st.composite
def _boxes(draw: st.DrawFn) -> BBox:
    x1 = draw(st.floats(min_value=-100.0, max_value=200.0, allow_nan=False))
    y1 = draw(st.floats(min_value=-100.0, max_value=200.0, allow_nan=False))
    width = draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False))
    height = draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False))
    return BBox(x1, y1, x1 + width, y1 + height)


@given(boxes=st.lists(_boxes(), max_size=20), extra=_boxes())
def test_coverage_is_bounded_and_monotone(boxes: list[BBox], extra: BBox) -> None:
    image_size = (600, 400)

    coverage = damage_coverage(boxes, image_size)
    with_extra = damage_coverage([*boxes, extra], image_size)

    assert 0.0 <= coverage <= 1.0
    assert 0.0 <= with_extra <= 1.0
    assert with_extra >= coverage - 1e-12
