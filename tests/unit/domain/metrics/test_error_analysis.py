"""TIDE-style false-positive taxonomy and missed ground truth (decision D10)."""

from __future__ import annotations

import pytest

from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.metrics.error_analysis import (
    ErrorBreakdown,
    ErrorCategory,
    error_breakdown,
)


def _image(image_id: str, *annotations: tuple[DamageClass, tuple[float, float, float, float]]):
    return AnnotatedImage(
        image_id=image_id,
        width=100,
        height=100,
        annotations=tuple(
            Annotation(damage_class, BBox(*box)) for damage_class, box in annotations
        ),
    )


def _detection(damage_class: DamageClass, box: tuple[float, float, float, float], score=0.9):
    return Detection(damage_class, score, BBox(*box))


def test_perfect_predictions_have_no_errors() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 10, 10))]}

    breakdown = error_breakdown(gt, preds)

    assert isinstance(breakdown, ErrorBreakdown)
    assert breakdown.cases == ()
    assert breakdown.total_false_positives == 0
    assert breakdown.total_missed == 0
    assert all(count == 0 for count in breakdown.counts.values())


def test_missed_ground_truth_is_counted_per_box() -> None:
    gt = [_image("img_1", (DamageClass.D10, (0, 0, 10, 10)))]

    breakdown = error_breakdown(gt, {})

    assert breakdown.counts[ErrorCategory.MISSED] == 1
    assert breakdown.total_missed == 1
    case = breakdown.cases[0]
    assert case.image_id == "img_1"
    assert case.category is ErrorCategory.MISSED
    assert case.damage_class is DamageClass.D10
    assert case.score is None
    assert case.bbox == BBox(0, 0, 10, 10)


def test_far_false_positive_is_background() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (50, 50, 60, 60))]}

    breakdown = error_breakdown(gt, preds)

    assert breakdown.counts[ErrorCategory.BACKGROUND] == 1
    assert breakdown.cases[0].matched_gt_bbox is None
    assert breakdown.cases[0].iou is None


def test_overlap_with_another_class_is_classification() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D40, (0, 0, 10, 10))]}

    breakdown = error_breakdown(gt, preds)

    assert breakdown.counts[ErrorCategory.CLASSIFICATION] == 1
    assert breakdown.counts[ErrorCategory.DUPLICATE] == 0
    case = breakdown.cases[0]
    assert case.category is ErrorCategory.CLASSIFICATION
    assert case.matched_gt_bbox == BBox(0, 0, 10, 10)
    assert case.iou == pytest.approx(1.0)


def test_same_class_partial_overlap_is_localization() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 4, 10))]}

    breakdown = error_breakdown(gt, preds)

    assert breakdown.counts[ErrorCategory.LOCALIZATION] == 1
    case = breakdown.cases[0]
    assert case.iou == pytest.approx(0.4)


def test_localization_lower_bound_is_inclusive() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 1, 10))]}

    breakdown = error_breakdown(gt, preds)

    assert breakdown.counts[ErrorCategory.LOCALIZATION] == 1


def test_tiny_overlap_is_background() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 1, 1))]}

    breakdown = error_breakdown(gt, preds)

    assert breakdown.counts[ErrorCategory.BACKGROUND] == 1


def test_second_match_of_an_already_matched_ground_truth_is_duplicate() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {
        "img_1": [
            _detection(DamageClass.D00, (0, 0, 10, 10), score=0.9),
            _detection(DamageClass.D00, (0, 0, 10, 10), score=0.8),
        ]
    }

    breakdown = error_breakdown(gt, preds)

    assert breakdown.counts[ErrorCategory.DUPLICATE] == 1
    assert breakdown.counts[ErrorCategory.MISSED] == 0
    assert breakdown.cases[0].score == pytest.approx(0.8)


def test_classification_takes_priority_over_duplicate() -> None:
    gt = [
        _image(
            "img_1",
            (DamageClass.D00, (0, 0, 10, 10)),
            (DamageClass.D40, (0, 0, 10, 10)),
        )
    ]
    preds = {
        "img_1": [
            _detection(DamageClass.D40, (0, 0, 10, 10), score=0.9),
            _detection(DamageClass.D40, (0, 0, 10, 10), score=0.8),
        ]
    }

    breakdown = error_breakdown(gt, preds)

    assert breakdown.counts[ErrorCategory.CLASSIFICATION] == 1
    assert breakdown.counts[ErrorCategory.DUPLICATE] == 0


def test_low_score_predictions_are_filtered_by_the_operating_threshold() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (50, 50, 60, 60), score=0.2)]}

    breakdown = error_breakdown(gt, preds, score_threshold=0.5)

    assert breakdown.total_false_positives == 0
    assert breakdown.counts[ErrorCategory.MISSED] == 1


def test_predictions_for_unknown_images_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown image"):
        error_breakdown([], {"nope": [_detection(DamageClass.D00, (0, 0, 1, 1))]})


@pytest.mark.parametrize("iou", [-0.1, 0.0, 1.1, float("nan")])
def test_rejects_invalid_iou(iou: float) -> None:
    with pytest.raises(ValueError, match="iou"):
        error_breakdown([], {}, iou=iou)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_score_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        error_breakdown([], {}, score_threshold=threshold)
