"""Per-image, per-class greedy matching with COCO semantics (decision D10).

The evaluator matches detections to ground truth independently for every class
and IoU threshold. Inside one image the detections are ranked by descending
score; each detection is paired with the *last* still-unmatched ground-truth
box whose IoU is at least the current best, which reproduces ``pycocotools``
(ties go to the later annotation). Matching exactly at the threshold counts as
a match, except at IoU 1.0 where the reference implementation caps the
threshold at ``1 - 1e-10`` to tolerate floating-point noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox, iou_matrix

_MAX_IOU = 1.0 - 1e-10


@dataclass(frozen=True, slots=True)
class MatchedDetections:
    """Score-ranked detections and their match state for each IoU threshold.

    ``scores`` has shape ``(num_detections,)``; ``matches`` and ``matched_gt``
    have shape ``(num_detections, num_thresholds)``. ``matched_gt`` stores the
    ground-truth index or ``-1`` for an unmatched detection.
    """

    scores: NDArray[np.float64]
    matches: NDArray[np.bool_]
    matched_gt: NDArray[np.int64]
    num_gt: int

    def true_positives(self, threshold_index: int) -> NDArray[np.bool_]:
        """Boolean true-positive flags for one IoU threshold."""
        return self.matches[:, threshold_index]


def _validate_thresholds(iou_thresholds: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in iou_thresholds)
    if not values:
        raise ValueError("iou_thresholds must contain at least one threshold")
    for value in values:
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"iou_thresholds must be within (0, 1], got {value!r}")
    return values


def match_detections(
    ground_truth: Sequence[BBox],
    detections: Sequence[Detection],
    iou_thresholds: Sequence[float],
    *,
    max_dets: int = 100,
) -> MatchedDetections:
    """Match one image/class' detections against its ground-truth boxes.

    Args:
        ground_truth: boxes of one class in annotation order.
        detections: predictions of the same class.
        iou_thresholds: IoU thresholds in ``(0, 1]``.
        max_dets: maximum ranked detections considered per image and class.

    Returns:
        Ranked detections and per-threshold match arrays.

    Raises:
        ValueError: thresholds or ``max_dets`` are invalid.
    """
    thresholds = _validate_thresholds(iou_thresholds)
    if max_dets < 1:
        raise ValueError(f"max_dets must be positive, got {max_dets!r}")

    gt_boxes = tuple(ground_truth)
    num_gt = len(gt_boxes)
    scores = np.asarray([detection.score for detection in detections], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")[:max_dets]
    ranked_scores = scores[order]
    ranked_detections = [detections[int(index)] for index in order]

    ious = iou_matrix(
        [detection.bbox for detection in ranked_detections],
        gt_boxes,
    )
    num_detections = len(ranked_detections)
    num_thresholds = len(thresholds)
    matches = np.zeros((num_detections, num_thresholds), dtype=np.bool_)
    matched_gt = np.full((num_detections, num_thresholds), -1, dtype=np.int64)

    for threshold_index, threshold in enumerate(thresholds):
        remaining = np.ones(num_gt, dtype=np.bool_)
        for detection_index in range(num_detections):
            best_iou = min(threshold, _MAX_IOU)
            best_gt = -1
            for gt_index in range(num_gt):
                if not remaining[gt_index]:
                    continue
                overlap = float(ious[detection_index, gt_index])
                if overlap < best_iou:
                    continue
                best_iou = overlap
                best_gt = gt_index
            if best_gt < 0:
                continue
            remaining[best_gt] = False
            matches[detection_index, threshold_index] = True
            matched_gt[detection_index, threshold_index] = best_gt

    return MatchedDetections(
        scores=ranked_scores,
        matches=matches,
        matched_gt=matched_gt,
        num_gt=num_gt,
    )
