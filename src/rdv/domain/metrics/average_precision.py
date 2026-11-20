"""COCO 101-point average precision from ranked match decisions (D10).

Given a globally score-ranked list of true-positive flags and the number of
ground-truth boxes of one class, this module computes the raw precision/recall
trace, its monotone precision envelope, the interpolated 101-point precision
used by COCO, and the resulting average precision. It never looks at pixels or
files: the evaluator feeds it arrays.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

RECALL_GRID: tuple[float, ...] = tuple(index / 100.0 for index in range(101))
_SCORE_EPSILON = float(np.spacing(1))


@dataclass(frozen=True, slots=True)
class AveragePrecision:
    """Ranked precision/recall trace plus its 101-point interpolation."""

    ap: float
    num_gt: int
    recall: tuple[float, ...]
    precision: tuple[float, ...]
    scores: tuple[float, ...]
    recall_101: tuple[float, ...]
    precision_101: tuple[float, ...]
    scores_101: tuple[float, ...]


def _validate_flags(true_positives: ArrayLike) -> NDArray[np.bool_]:
    flags = np.asarray(true_positives)
    if flags.ndim != 1:
        raise ValueError("true_positives must be one-dimensional")
    if flags.size and not np.all(np.isin(flags, (0, 1))):
        raise ValueError("true_positives must contain only 0/False and 1/True")
    return flags.astype(np.bool_)


def compute_average_precision(
    scores: ArrayLike,
    true_positives: ArrayLike,
    num_gt: int,
    *,
    recall_grid: Sequence[float] = RECALL_GRID,
) -> AveragePrecision:
    """Compute 101-point AP for one class at one IoU threshold.

    Args:
        scores: detection scores; sorted descending internally (stable).
        true_positives: one flag per score, ``True`` when the detection was
            matched to a previously unmatched ground-truth box.
        num_gt: number of ground-truth boxes of the class in the full set.
        recall_grid: recall points used for interpolation; defaults to COCO's
            0.00, 0.01, ..., 1.00.

    Returns:
        AP and the raw/interpolated traces. With no detections, AP is 0 and
        the raw traces are empty.

    Raises:
        ValueError: the inputs are not one-dimensional, their lengths differ,
            scores are not finite, flags are not binary, or ``num_gt`` is not
            positive.
    """
    if num_gt < 1:
        raise ValueError(f"num_gt must be positive, got {num_gt!r}")
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    flags = _validate_flags(true_positives)
    if score_array.shape != flags.shape:
        raise ValueError("scores and true_positives must have the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")

    grid = np.asarray(tuple(recall_grid), dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("recall_grid must be a non-empty sequence of finite values")

    order = np.argsort(-score_array, kind="stable")
    ranked_scores = score_array[order]
    ranked_flags = flags[order]
    num_detections = int(ranked_scores.size)

    raw_recall = np.zeros(num_detections, dtype=np.float64)
    raw_precision = np.zeros(num_detections, dtype=np.float64)
    envelope = np.zeros(num_detections, dtype=np.float64)
    if num_detections:
        true_positives_sum = np.cumsum(ranked_flags, dtype=np.float64)
        false_positives_sum = np.cumsum(~ranked_flags, dtype=np.float64)
        raw_recall = true_positives_sum / num_gt
        raw_precision = true_positives_sum / (
            false_positives_sum + true_positives_sum + _SCORE_EPSILON
        )
        # COCO's monotone precision envelope: a later rank can only improve
        # the precision reported at an earlier recall.
        envelope = np.maximum.accumulate(raw_precision[::-1])[::-1]

    precision_101 = np.zeros(grid.size, dtype=np.float64)
    scores_101 = np.zeros(grid.size, dtype=np.float64)
    if num_detections:
        indices = np.searchsorted(raw_recall, grid, side="left")
        valid = indices < num_detections
        safe_indices = np.minimum(indices, num_detections - 1)
        precision_101[valid] = envelope[safe_indices[valid]]
        scores_101[valid] = ranked_scores[safe_indices[valid]]

    ap = float(np.mean(precision_101)) if grid.size else 0.0

    return AveragePrecision(
        ap=ap,
        num_gt=num_gt,
        recall=tuple(float(value) for value in raw_recall),
        precision=tuple(float(value) for value in raw_precision),
        scores=tuple(float(value) for value in ranked_scores),
        recall_101=tuple(float(value) for value in grid),
        precision_101=tuple(float(value) for value in precision_101),
        scores_101=tuple(float(value) for value in scores_101),
    )


def mean_average_precision(values: Sequence[float]) -> float:
    """Mean of per-class AP values; empty input means 0.0."""
    if not values:
        return 0.0
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"AP values must be finite, got {value!r}")
    return float(np.mean(np.asarray(values, dtype=np.float64)))
