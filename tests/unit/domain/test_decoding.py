"""Pure DETR post-processing (decision D8)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rdv.domain.classes import DamageClass
from rdv.domain.decoding import decode_detr_outputs

# One query, one confident class, a box covering the middle tenth of the image.
LOGITS = np.full((1, 4), -10.0, dtype=np.float32)
LOGITS[0, 0] = 0.0
BOXES = np.array([[0.5, 0.5, 0.1, 0.2]], dtype=np.float32)


def test_single_query_scores_sigmoid_of_its_logit() -> None:
    detections = decode_detr_outputs(LOGITS, BOXES, (600, 600), score_threshold=0.4)

    assert len(detections) == 1
    assert detections[0].damage_class is DamageClass.D00
    assert detections[0].score == pytest.approx(0.5)


def test_single_query_decodes_to_the_expected_box_on_a_square_image() -> None:
    (detection,) = decode_detr_outputs(LOGITS, BOXES, (600, 600), score_threshold=0.4)

    assert detection.bbox.to_xyxy() == pytest.approx((270.0, 240.0, 330.0, 360.0))


def test_single_query_decodes_to_the_expected_box_on_a_wide_image() -> None:
    (detection,) = decode_detr_outputs(LOGITS, BOXES, (3650, 2044), score_threshold=0.4)

    assert detection.bbox.to_xyxy() == pytest.approx(
        (1825.0 - 182.5, 1022.0 - 204.4, 1825.0 + 182.5, 1022.0 + 204.4)
    )


def test_scores_are_non_increasing_and_limited_to_the_top_detections() -> None:
    queries = 20
    logits = np.zeros((queries, 4), dtype=np.float32)
    logits[:, 1] = np.linspace(0.0, 5.0, queries)
    boxes = np.tile(BOXES, (queries, 1))

    detections = decode_detr_outputs(
        logits, boxes, (600, 600), score_threshold=0.0, max_detections=7
    )

    assert len(detections) == 7
    assert [detection.damage_class for detection in detections] == [DamageClass.D10] * 7
    scores = [detection.score for detection in detections]
    assert scores == sorted(scores, reverse=True)


def test_top_limit_defaults_to_one_hundred() -> None:
    queries = 120
    logits = np.full((queries, 4), 10.0, dtype=np.float32)
    boxes = np.tile(BOXES, (queries, 1))

    assert len(decode_detr_outputs(logits, boxes, (600, 600))) == 100


def test_threshold_filters_low_scores() -> None:
    logits = np.full((2, 4), -10.0, dtype=np.float32)
    logits[0, 0] = 0.0  # 0.5
    logits[1, 0] = -2.0  # 0.12
    boxes = np.tile(BOXES, (2, 1))

    detections = decode_detr_outputs(logits, boxes, (600, 600), score_threshold=0.3)

    assert [detection.score for detection in detections] == [pytest.approx(0.5)]


def test_one_query_can_produce_multiple_class_detections() -> None:
    logits = np.zeros((1, 4), dtype=np.float32)  # all classes score 0.5

    detections = decode_detr_outputs(logits, BOXES, (600, 600), score_threshold=0.4)

    assert [detection.damage_class for detection in detections] == list(DamageClass)


def test_boxes_are_clipped_to_the_image() -> None:
    boxes = np.array([[0.05, 0.05, 0.5, 0.5]], dtype=np.float32)

    (detection,) = decode_detr_outputs(LOGITS, boxes, (600, 600), score_threshold=0.4)

    x1, y1, x2, y2 = detection.bbox.to_xyxy()
    assert 0.0 <= x1 < x2 <= 600.0
    assert 0.0 <= y1 < y2 <= 600.0


def test_boxes_smaller_than_one_pixel_are_dropped() -> None:
    boxes = np.array([[0.5, 0.5, 0.001, 0.001]], dtype=np.float32)

    assert decode_detr_outputs(LOGITS, boxes, (600, 600), score_threshold=0.4) == []


def test_non_positive_box_sizes_are_dropped() -> None:
    logits = np.zeros((2, 4), dtype=np.float32)
    boxes = np.array([[0.5, 0.5, 0.0, 0.2], [0.5, 0.5, 0.1, -0.2]], dtype=np.float32)

    assert decode_detr_outputs(logits, boxes, (600, 600), score_threshold=0.4) == []


def test_decoding_is_deterministic() -> None:
    logits = np.zeros((30, 4), dtype=np.float32)
    boxes = np.tile(BOXES, (30, 1))

    first = decode_detr_outputs(logits, boxes, (640, 480), score_threshold=0.2)
    second = decode_detr_outputs(logits, boxes, (640, 480), score_threshold=0.2)

    assert first == second


def test_ties_are_broken_by_query_then_class_index() -> None:
    logits = np.zeros((2, 4), dtype=np.float32)
    boxes = np.array([[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]], dtype=np.float32)

    detections = decode_detr_outputs(logits, boxes, (600, 600), score_threshold=0.4)

    assert [(detection.damage_class, detection.bbox.x1) for detection in detections] == [
        (DamageClass.D00, pytest.approx(90.0)),
        (DamageClass.D10, pytest.approx(90.0)),
        (DamageClass.D20, pytest.approx(90.0)),
        (DamageClass.D40, pytest.approx(90.0)),
        (DamageClass.D00, pytest.approx(450.0)),
        (DamageClass.D10, pytest.approx(450.0)),
        (DamageClass.D20, pytest.approx(450.0)),
        (DamageClass.D40, pytest.approx(450.0)),
    ]


def test_empty_query_dimension_returns_no_detections() -> None:
    assert decode_detr_outputs(np.empty((0, 4)), np.empty((0, 4)), (600, 600)) == []


@pytest.mark.parametrize(
    ("logits", "boxes", "message"),
    [
        (np.zeros((1, 4)), np.zeros((1, 3)), "pred_boxes"),
        (np.zeros(4), BOXES, "two-dimensional"),
        (np.zeros((1, 4)), np.zeros((2, 4)), "pred_boxes"),
        (np.zeros((1, 5)), BOXES, "classes"),
        (np.zeros((1, 4)), np.full((1, 4), np.nan), "finite"),
    ],
)
def test_invalid_arrays_are_rejected(logits: np.ndarray, boxes: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_detr_outputs(logits, boxes, (600, 600))


@pytest.mark.parametrize("image_size", [(0, 600), (600, 0), (-1, 600)])
def test_invalid_image_dimensions_are_rejected(image_size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="image dimensions"):
        decode_detr_outputs(LOGITS, BOXES, image_size)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        decode_detr_outputs(LOGITS, BOXES, (600, 600), score_threshold=threshold)


@pytest.mark.parametrize("max_detections", [0, -1])
def test_invalid_max_detections_are_rejected(max_detections: int) -> None:
    with pytest.raises(ValueError, match="max_detections"):
        decode_detr_outputs(LOGITS, BOXES, (600, 600), max_detections=max_detections)


@given(
    logits=st.lists(
        st.lists(
            st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
        min_size=0,
        max_size=20,
    ),
    box_values=st.lists(
        st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
        min_size=0,
        max_size=20,
    ),
    width=st.integers(min_value=1, max_value=4000),
    height=st.integers(min_value=1, max_value=4000),
)
def test_decoded_boxes_stay_inside_the_image(
    logits: list[list[float]],
    box_values: list[list[float]],
    width: int,
    height: int,
) -> None:
    if len(box_values) != len(logits):
        box_values = box_values[: len(logits)] + [[0.5, 0.5, 0.1, 0.1]] * (
            len(logits) - len(box_values)
        )

    detections = decode_detr_outputs(
        np.asarray(logits, dtype=np.float64),
        np.asarray(box_values, dtype=np.float64),
        (width, height),
        score_threshold=0.1,
    )

    assert len(detections) <= 100
    scores = [detection.score for detection in detections]
    assert scores == sorted(scores, reverse=True)
    for detection in detections:
        box = detection.bbox
        assert 0.0 <= box.x1 <= box.x2 <= width
        assert 0.0 <= box.y1 <= box.y2 <= height
