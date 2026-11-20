"""PR curve view models and F1-optimal operating threshold (decisions D10)."""

from __future__ import annotations

import pytest

from rdv.domain.classes import DamageClass
from rdv.domain.metrics.average_precision import compute_average_precision
from rdv.domain.metrics.curves import ThresholdSelection, best_f1_threshold, build_pr_curve


def test_best_f1_threshold_of_perfect_predictions() -> None:
    selection = best_f1_threshold([0.9, 0.8], [True, True], 2)

    assert isinstance(selection, ThresholdSelection)
    assert selection.threshold == pytest.approx(0.8)
    assert selection.precision == pytest.approx(1.0)
    assert selection.recall == pytest.approx(1.0)
    assert selection.f1 == pytest.approx(1.0)
    assert selection.true_positives == 2
    assert selection.false_positives == 0


def test_best_f1_threshold_without_detections_is_zero() -> None:
    selection = best_f1_threshold([], [], 3)

    assert selection.threshold == 0.0
    assert selection.precision == 0.0
    assert selection.recall == 0.0
    assert selection.f1 == 0.0
    assert selection.true_positives == 0
    assert selection.false_positives == 0


def test_best_f1_threshold_ignores_false_positives_after_the_last_true_positive() -> None:
    selection = best_f1_threshold([0.9, 0.7, 0.5], [True, True, False], 2)

    assert selection.threshold == pytest.approx(0.7)
    assert selection.f1 == pytest.approx(1.0)


def test_best_f1_threshold_sorts_scores_first() -> None:
    first = best_f1_threshold([0.7, 0.9], [True, False], 1)
    second = best_f1_threshold([0.9, 0.7], [False, True], 1)

    assert first.threshold == pytest.approx(second.threshold)
    assert first.f1 == pytest.approx(second.f1)


def test_best_f1_threshold_returns_a_threshold_that_includes_the_selected_rank() -> None:
    selection = best_f1_threshold([0.9, 0.8, 0.7], [True, False, True], 2)

    assert selection.threshold == pytest.approx(0.7)
    assert selection.true_positives == 2
    assert selection.false_positives == 1
    assert selection.f1 == pytest.approx(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))


@pytest.mark.parametrize("num_gt", [0, -1])
def test_best_f1_threshold_requires_ground_truth(num_gt: int) -> None:
    with pytest.raises(ValueError, match="num_gt"):
        best_f1_threshold([0.9], [True], num_gt)


def test_best_f1_threshold_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        best_f1_threshold([0.9], [True, False], 1)


def test_pr_curve_carries_raw_and_interpolated_points() -> None:
    ap = compute_average_precision([0.9, 0.8], [True, False], 1)

    curve = build_pr_curve(DamageClass.D10, ap, iou_threshold=0.5)

    assert curve.damage_class is DamageClass.D10
    assert curve.iou_threshold == pytest.approx(0.5)
    assert curve.ap == pytest.approx(ap.ap)
    assert curve.recall == ap.recall
    assert curve.precision == ap.precision
    assert curve.scores == ap.scores
    assert curve.recall_101 == ap.recall_101
    assert curve.precision_101 == ap.precision_101
    assert curve.scores_101 == ap.scores_101


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.1, float("nan")])
def test_pr_curve_rejects_invalid_iou_threshold(threshold: float) -> None:
    ap = compute_average_precision([0.9], [True], 1)

    with pytest.raises(ValueError, match="iou_threshold"):
        build_pr_curve(DamageClass.D00, ap, iou_threshold=threshold)
