"""COCO-style greedy matching per image and class (decision D10)."""

from __future__ import annotations

import numpy as np
import pytest

from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.metrics.matching import match_detections


def _detection(score: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(DamageClass.D00, score, BBox(*box))


def test_perfect_predictions_match_every_ground_truth() -> None:
    gt = [BBox(0, 0, 10, 10), BBox(50, 50, 70, 80)]
    detections = [_detection(0.9, (0, 0, 10, 10)), _detection(0.8, (50, 50, 70, 80))]

    result = match_detections(gt, detections, (0.5,))

    assert result.num_gt == 2
    assert result.scores.tolist() == pytest.approx([0.9, 0.8])
    assert result.matches[:, 0].tolist() == [True, True]
    assert result.matched_gt[:, 0].tolist() == [0, 1]


def test_unmatched_detection_is_a_false_positive() -> None:
    result = match_detections([BBox(0, 0, 10, 10)], [_detection(0.7, (50, 50, 60, 60))], (0.5,))

    assert result.matches[:, 0].tolist() == [False]
    assert result.matched_gt[:, 0].tolist() == [-1]


def test_duplicate_detection_is_a_false_positive_after_the_first_match() -> None:
    gt = [BBox(0, 0, 10, 10)]
    detections = [_detection(0.5, (0, 0, 10, 10)), _detection(0.9, (0, 0, 10, 10))]

    result = match_detections(gt, detections, (0.5,))

    assert result.scores.tolist() == pytest.approx([0.9, 0.5])
    assert result.matches[:, 0].tolist() == [True, False]


def test_matching_sorts_detections_by_descending_score() -> None:
    gt = [BBox(0, 0, 10, 10), BBox(20, 20, 30, 30)]
    detections = [_detection(0.2, (20, 20, 30, 30)), _detection(0.9, (0, 0, 10, 10))]

    result = match_detections(gt, detections, (0.5,))

    assert result.scores.tolist() == pytest.approx([0.9, 0.2])
    assert result.matches[:, 0].tolist() == [True, True]


def test_max_detections_keeps_only_the_highest_scores() -> None:
    detections = [
        _detection(0.1, (0, 0, 1, 1)),
        _detection(0.6, (2, 2, 3, 3)),
        _detection(0.9, (4, 4, 5, 5)),
    ]

    result = match_detections([], detections, (0.5,), max_dets=2)

    assert result.scores.tolist() == pytest.approx([0.9, 0.6])


def test_matches_exactly_at_the_iou_threshold() -> None:
    gt = [BBox(0, 0, 10, 10)]
    detections = [_detection(0.9, (0, 0, 10, 5))]

    result = match_detections(gt, detections, (0.5,))

    assert result.matches[0, 0]


def test_thresholds_are_applied_independently() -> None:
    gt = [BBox(0, 0, 10, 10)]
    detections = [_detection(0.9, (0, 0, 10, 6))]

    result = match_detections(gt, detections, (0.5, 0.7))

    assert result.matches[:, 0].tolist() == [True]
    assert result.matches[:, 1].tolist() == [False]


def test_no_ground_truth_means_no_matches() -> None:
    result = match_detections([], [_detection(0.9, (0, 0, 1, 1))], (0.5,))

    assert result.num_gt == 0
    assert result.matches[:, 0].tolist() == [False]


def test_no_detections_returns_empty_arrays() -> None:
    result = match_detections([BBox(0, 0, 1, 1)], [], (0.5, 0.75))

    assert result.scores.shape == (0,)
    assert result.matches.shape == (0, 2)
    assert result.matched_gt.shape == (0, 2)


@pytest.mark.parametrize("thresholds", [(), (0.0,), (-0.1,), (1.1,), (float("nan"),)])
def test_rejects_invalid_iou_thresholds(thresholds: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="iou_thresholds"):
        match_detections([], [], thresholds)


def test_rejects_non_positive_max_detections() -> None:
    with pytest.raises(ValueError, match="max_dets"):
        match_detections([], [], (0.5,), max_dets=0)


def test_matches_array_is_boolean() -> None:
    result = match_detections([BBox(0, 0, 1, 1)], [_detection(0.5, (0, 0, 1, 1))], (0.5,))

    assert result.matches.dtype == np.bool_
