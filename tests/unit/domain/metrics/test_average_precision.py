"""COCO 101-point average precision from ranked matches (decision D10)."""

from __future__ import annotations

import numpy as np
import pytest

from rdv.domain.metrics.average_precision import (
    RECALL_GRID,
    compute_average_precision,
)


def _ap(scores: list[float], true_positives: list[bool], num_gt: int) -> float:
    return compute_average_precision(scores, true_positives, num_gt).ap


def test_perfect_predictions_score_one() -> None:
    result = compute_average_precision([0.9, 0.8], [True, True], 2)

    assert result.ap == pytest.approx(1.0, abs=1e-12)
    assert result.recall == (pytest.approx(0.5), pytest.approx(1.0))
    assert result.precision == (pytest.approx(1.0), pytest.approx(1.0))
    assert result.recall_101[0] == 0.0
    assert result.recall_101[-1] == 1.0
    assert len(result.recall_101) == len(RECALL_GRID) == 101


def test_no_predictions_score_zero() -> None:
    result = compute_average_precision([], [], 3)

    assert result.ap == 0.0
    assert result.recall == ()
    assert result.precision == ()
    assert result.recall_101 == RECALL_GRID
    assert all(value == 0.0 for value in result.precision_101)


def test_false_positive_before_true_positive_halves_the_ap() -> None:
    # Ranked: FP (0.9), TP (0.8). Precision after the envelope is 1/2 at
    # every recall point, so the 101-point average is exactly 0.5.
    result = compute_average_precision([0.9, 0.8], [False, True], 1)

    assert result.ap == pytest.approx(0.5)
    assert result.recall == (pytest.approx(0.0), pytest.approx(1.0))
    assert result.precision == (pytest.approx(0.0), pytest.approx(0.5))
    assert result.precision_101 == pytest.approx([0.5] * 101)


def test_true_positive_before_false_positive_still_scores_one() -> None:
    result = compute_average_precision([0.9, 0.8], [True, False], 1)

    assert result.ap == pytest.approx(1.0)
    assert result.precision == (pytest.approx(1.0), pytest.approx(0.5))


def test_two_true_positives_and_one_false_positive_match_manual_101_points() -> None:
    # Ranked: TP (0.9), FP (0.8), TP (0.7) -> recall [0.5, 0.5, 1.0],
    # raw precision [1, 0.5, 2/3]; the envelope lifts rank 2 to 2/3.
    # The 101-point grid has 51 points at recall <= 0.5 (precision 1) and
    # 50 points above 0.5 (precision 2/3).
    result = compute_average_precision([0.9, 0.8, 0.7], [True, False, True], 2)

    expected = (51 * 1.0 + 50 * (2.0 / 3.0)) / 101
    assert result.ap == pytest.approx(expected, abs=1e-12)
    assert result.recall_101 == RECALL_GRID
    assert result.precision_101[0] == pytest.approx(1.0)
    assert result.precision_101[50] == pytest.approx(1.0)
    assert result.precision_101[51] == pytest.approx(2.0 / 3.0)


def test_scores_at_recall_points_follow_the_interpolation_index() -> None:
    result = compute_average_precision([0.9, 0.8], [False, True], 1)

    # Recall 0 resolves to the first ranked detection (the FP); any recall
    # above zero resolves to the first detection that reaches it (the TP).
    assert result.scores_101[0] == pytest.approx(0.9)
    assert result.scores_101[1] == pytest.approx(0.8)
    assert result.scores_101[-1] == pytest.approx(0.8)


def test_input_order_does_not_matter() -> None:
    first = compute_average_precision([0.4, 0.9], [True, False], 1)
    second = compute_average_precision([0.9, 0.4], [False, True], 1)

    assert first.ap == pytest.approx(second.ap)
    assert first.recall == second.recall
    assert first.precision == second.precision


def test_precision_grid_is_monotone_non_increasing() -> None:
    result = compute_average_precision([0.9, 0.8, 0.7, 0.6], [True, False, True, False], 2)

    pairs = zip(result.precision_101, result.precision_101[1:], strict=False)
    assert all(left >= right for left, right in pairs)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_average_precision([0.9], [True, False], 1)


def test_rejects_non_positive_num_gt() -> None:
    with pytest.raises(ValueError, match="num_gt"):
        compute_average_precision([0.9], [True], 0)


def test_rejects_non_binary_true_positives() -> None:
    with pytest.raises(ValueError, match="true_positives"):
        compute_average_precision([0.9], [2], 1)  # type: ignore[list-item]


def test_returns_numpy_free_tuples() -> None:
    result = compute_average_precision(np.array([0.9]), np.array([True]), 1)

    assert isinstance(result.recall_101, tuple)
    assert isinstance(result.precision_101, tuple)
    assert isinstance(result.scores_101, tuple)
