"""5x5 confusion matrix with a background row and column (decision D10).

Matching is class-agnostic and greedy by descending score at a fixed IoU, so
the matrix answers "what did the model call each labeled box?" independently
of the class-aware AP matching. Rows are ground truth (D00..D40, background)
and columns are predictions (D00..D40, background); ``matrix[i, j]`` counts
boxes whose truth is row ``i`` and prediction is column ``j``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rdv.domain.annotation import AnnotatedImage
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import iou_matrix
from rdv.domain.metrics._common import (
    index_ground_truth,
    ranked_predictions,
    validate_iou,
    validate_prediction_ids,
    validate_score_threshold,
)

BACKGROUND_LABEL = "background"
CONFUSION_LABELS: tuple[str, ...] = (
    *(damage_class.code for damage_class in DamageClass),
    BACKGROUND_LABEL,
)
_BACKGROUND_INDEX = len(DamageClass)


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Counts of ground-truth vs predicted classes, including background."""

    matrix: NDArray[np.int64]
    labels: tuple[str, ...] = CONFUSION_LABELS

    def __post_init__(self) -> None:
        array = np.asarray(self.matrix)
        if array.shape != (len(self.labels), len(self.labels)):
            raise ValueError(
                f"matrix must have shape ({len(self.labels)}, {len(self.labels)}), "
                f"got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("matrix must contain integer counts")
        if np.any(array < 0):
            raise ValueError("matrix counts must be non-negative")
        object.__setattr__(self, "matrix", array.astype(np.int64, copy=True))
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def total(self) -> int:
        """Total number of matched, missed and spurious boxes."""
        return int(self.matrix.sum())


def confusion_matrix(
    ground_truth: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    *,
    iou: float = 0.5,
    score_threshold: float = 0.0,
) -> ConfusionMatrix:
    """Build the class-agnostic confusion matrix at one operating point.

    Each image's predictions are processed by descending score. A prediction
    takes the unmatched ground-truth box (of any class) with the highest IoU at
    or above ``iou``; ties keep the earlier annotation. Unmatched predictions
    land in the background row and unmatched ground truth in the background
    column.

    Args:
        ground_truth: labeled images; duplicate ids are rejected.
        predictions: detections keyed by image id; unknown ids are rejected.
        iou: matching IoU threshold in ``(0, 1]``.
        score_threshold: minimum detection score, in ``[0, 1]``.

    Raises:
        ValueError: thresholds are invalid, ids are duplicated/unknown.
    """
    iou_threshold = validate_iou(iou)
    minimum_score = validate_score_threshold(score_threshold)
    images = index_ground_truth(ground_truth)
    validate_prediction_ids(predictions, images)

    matrix = np.zeros((len(CONFUSION_LABELS), len(CONFUSION_LABELS)), dtype=np.int64)
    for image in images:
        annotations = image.annotations
        detections = ranked_predictions(predictions, image.image_id, minimum_score)
        if not annotations and not detections:
            continue
        overlaps = iou_matrix(
            [detection.bbox for detection in detections],
            [annotation.bbox for annotation in annotations],
        )
        matched = np.zeros(len(annotations), dtype=np.bool_)

        for detection_index, detection in enumerate(detections):
            candidate_scores = np.where(
                (~matched) & (overlaps[detection_index] >= iou_threshold),
                overlaps[detection_index],
                -np.inf,
            )
            best = int(np.argmax(candidate_scores)) if candidate_scores.size else -1
            if best >= 0 and candidate_scores[best] >= iou_threshold:
                matched[best] = True
                matrix[annotations[best].damage_class, detection.damage_class] += 1
            else:
                matrix[_BACKGROUND_INDEX, detection.damage_class] += 1

        for annotation_index, annotation in enumerate(annotations):
            if not matched[annotation_index]:
                matrix[annotation.damage_class, _BACKGROUND_INDEX] += 1

    return ConfusionMatrix(matrix=matrix)
