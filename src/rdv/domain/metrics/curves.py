"""PR curve view models and F1-optimal operating threshold (decision D10).

The evaluation split is chosen on validation data: the operating threshold is
the score at the rank that maximises F1 at IoU 0.5. This module owns that
calculation and the immutable curve object that reports consume, so the
selection can be unit-tested without any dataset or detector.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from rdv.domain.classes import DamageClass
from rdv.domain.metrics.average_precision import AveragePrecision


@dataclass(frozen=True, slots=True)
class PRCurve:
    """Precision/recall trace of one class at one IoU threshold."""

    damage_class: DamageClass
    iou_threshold: float
    ap: float
    recall: tuple[float, ...]
    precision: tuple[float, ...]
    scores: tuple[float, ...]
    recall_101: tuple[float, ...]
    precision_101: tuple[float, ...]
    scores_101: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """Operating threshold chosen by maximum F1 at IoU 0.5."""

    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int


def build_pr_curve(
    damage_class: DamageClass,
    average_precision: AveragePrecision,
    *,
    iou_threshold: float,
) -> PRCurve:
    """Attach class and IoU metadata to a computed AP trace.

    Raises:
        ValueError: the IoU threshold is not finite or outside ``(0, 1]``.
    """
    if not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be within (0, 1], got {iou_threshold!r}")
    return PRCurve(
        damage_class=damage_class,
        iou_threshold=float(iou_threshold),
        ap=average_precision.ap,
        recall=average_precision.recall,
        precision=average_precision.precision,
        scores=average_precision.scores,
        recall_101=average_precision.recall_101,
        precision_101=average_precision.precision_101,
        scores_101=average_precision.scores_101,
    )


def best_f1_threshold(
    scores: ArrayLike,
    true_positives: ArrayLike,
    num_gt: int,
) -> ThresholdSelection:
    """Return the score threshold that maximises F1 at one IoU.

    The threshold is the score of the highest-ranked detection at the winning
    rank (first maximum, so ties prefer the higher score). With no detections
    the selection is all zeros and a threshold of ``0.0``.

    Args:
        scores: detection scores of one class at one IoU threshold.
        true_positives: match flags aligned with ``scores``.
        num_gt: number of ground-truth boxes of that class.

    Raises:
        ValueError: the arrays are malformed, ``num_gt`` is not positive, or
            the flags are not binary.
    """
    if num_gt < 1:
        raise ValueError(f"num_gt must be positive, got {num_gt!r}")
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    flag_array = np.asarray(true_positives)
    if flag_array.ndim != 1 or (flag_array.size and not np.all(np.isin(flag_array, (0, 1)))):
        raise ValueError("true_positives must be a one-dimensional 0/1 array")
    flags = flag_array.astype(np.bool_)
    if score_array.shape != flags.shape:
        raise ValueError("scores and true_positives must have the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if score_array.size == 0:
        return ThresholdSelection(
            threshold=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            true_positives=0,
            false_positives=0,
        )

    order = np.argsort(-score_array, kind="stable")
    ranked_scores = score_array[order]
    ranked_flags = flags[order]
    cumulative_tp = np.cumsum(ranked_flags, dtype=np.float64)
    ranks = np.arange(1, ranked_scores.size + 1, dtype=np.float64)
    precision = cumulative_tp / ranks
    recall = cumulative_tp / num_gt
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    best_index = int(np.argmax(f1))
    true_positives_at_best = int(cumulative_tp[best_index])
    false_positives_at_best = int(best_index + 1 - true_positives_at_best)

    return ThresholdSelection(
        threshold=float(ranked_scores[best_index]),
        precision=float(precision[best_index]),
        recall=float(recall[best_index]),
        f1=float(f1[best_index]),
        true_positives=true_positives_at_best,
        false_positives=false_positives_at_best,
    )


def thresholds_in_order(values: Sequence[float]) -> tuple[float, ...]:
    """Normalise thresholds to a tuple of finite floats in ``(0, 1]``."""
    result = tuple(float(value) for value in values)
    for value in result:
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"thresholds must be within (0, 1], got {value!r}")
    return result
