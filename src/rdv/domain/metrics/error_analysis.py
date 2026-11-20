"""TIDE-style false-positive taxonomy per decision D10.

Rules at IoU 0.5, applied in this order to every unmatched prediction that
passes the operating score threshold:

1. **classification** - overlaps a ground-truth box of another class by at
   least 0.5 (checked first, so it wins over ``duplicate`` when both apply);
2. **localization** - overlaps a same-class box with ``0.1 <= IoU < 0.5``;
3. **duplicate** - overlaps a same-class box already matched by a
   higher-scored prediction (``IoU >= 0.5``);
4. **background** - anything else.

Unmatched ground truth becomes ``missed``. The cases carry the metadata the
error gallery needs (image, class, score, box) and are deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from rdv.domain.annotation import AnnotatedImage
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox, iou_matrix
from rdv.domain.metrics._common import (
    index_ground_truth,
    ranked_predictions,
    validate_iou,
    validate_prediction_ids,
    validate_score_threshold,
)

_LOCALIZATION_LOWER_BOUND = 0.1


class ErrorCategory(StrEnum):
    """False-positive categories plus missed ground truth."""

    CLASSIFICATION = "classification"
    LOCALIZATION = "localization"
    DUPLICATE = "duplicate"
    BACKGROUND = "background"
    MISSED = "missed"


@dataclass(frozen=True, slots=True)
class ErrorCase:
    """One false positive or missed ground-truth box."""

    image_id: str
    category: ErrorCategory
    damage_class: DamageClass
    bbox: BBox
    score: float | None = None
    matched_gt_bbox: BBox | None = None
    iou: float | None = None


@dataclass(frozen=True, slots=True)
class ErrorBreakdown:
    """Error counts by category and the underlying deterministic cases."""

    counts: Mapping[ErrorCategory, int]
    cases: tuple[ErrorCase, ...] = ()

    def __post_init__(self) -> None:
        normalized: dict[ErrorCategory, int] = {}
        for category in ErrorCategory:
            count = self.counts.get(category, 0)
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"counts must be non-negative integers, got {count!r}")
            normalized[category] = count
        if (
            len(self.cases)
            != sum(
                count
                for category, count in normalized.items()
                if category is not ErrorCategory.MISSED
            )
            + normalized[ErrorCategory.MISSED]
        ):
            raise ValueError("cases do not match the category counts")
        object.__setattr__(self, "counts", MappingProxyType(normalized))
        object.__setattr__(self, "cases", tuple(self.cases))

    @property
    def total_false_positives(self) -> int:
        """False positives across the four FP categories."""
        return sum(
            count for category, count in self.counts.items() if category is not ErrorCategory.MISSED
        )

    @property
    def total_missed(self) -> int:
        """Ground-truth boxes that no prediction matched."""
        return self.counts[ErrorCategory.MISSED]


def error_breakdown(
    ground_truth: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    *,
    iou: float = 0.5,
    score_threshold: float = 0.0,
) -> ErrorBreakdown:
    """Classify every false positive and count missed ground-truth boxes.

    Args:
        ground_truth: labeled images; duplicate ids are rejected.
        predictions: detections keyed by image id; unknown ids are rejected.
        iou: matching IoU threshold in ``(0, 1]`` (D10 uses 0.5).
        score_threshold: minimum detection score, in ``[0, 1]``.

    Raises:
        ValueError: thresholds are invalid or ids are unknown/duplicated.
    """
    iou_threshold = validate_iou(iou)
    minimum_score = validate_score_threshold(score_threshold)
    images = index_ground_truth(ground_truth)
    validate_prediction_ids(predictions, images)

    counts = dict.fromkeys(ErrorCategory, 0)
    cases: list[ErrorCase] = []

    for image in images:
        annotations = image.annotations
        detections = ranked_predictions(predictions, image.image_id, minimum_score)
        if not annotations and not detections:
            continue
        overlaps = iou_matrix(
            [detection.bbox for detection in detections],
            [annotation.bbox for annotation in annotations],
        )
        matched_gt = np.zeros(len(annotations), dtype=np.bool_)

        for detection_index, detection in enumerate(detections):
            candidates = [
                index
                for index, annotation in enumerate(annotations)
                if not matched_gt[index] and annotation.damage_class is detection.damage_class
            ]
            if candidates:
                candidate_values = [float(overlaps[detection_index, index]) for index in candidates]
                ranked = int(np.argmax(candidate_values))
                if candidate_values[ranked] >= iou_threshold:
                    gt_index = candidates[ranked]
                    matched_gt[gt_index] = True
                    continue

            other_class_iou = [
                (float(overlaps[detection_index, index]), index)
                for index, annotation in enumerate(annotations)
                if annotation.damage_class is not detection.damage_class
            ]
            same_class_iou = [
                (float(overlaps[detection_index, index]), index)
                for index, annotation in enumerate(annotations)
                if annotation.damage_class is detection.damage_class
            ]

            category = ErrorCategory.BACKGROUND
            matched_gt_bbox: BBox | None = None
            matched_iou: float | None = None

            if other_class_iou:
                best_iou, best_index = max(other_class_iou, key=lambda item: item[0])
                if best_iou >= iou_threshold:
                    category = ErrorCategory.CLASSIFICATION
                    matched_gt_bbox = annotations[best_index].bbox
                    matched_iou = best_iou
            if matched_gt_bbox is None and same_class_iou:
                best_iou, best_index = max(same_class_iou, key=lambda item: item[0])
                if _LOCALIZATION_LOWER_BOUND <= best_iou < iou_threshold:
                    category = ErrorCategory.LOCALIZATION
                    matched_gt_bbox = annotations[best_index].bbox
                    matched_iou = best_iou
                elif best_iou >= iou_threshold:
                    category = ErrorCategory.DUPLICATE
                    matched_gt_bbox = annotations[best_index].bbox
                    matched_iou = best_iou

            counts[category] += 1
            cases.append(
                ErrorCase(
                    image_id=image.image_id,
                    category=category,
                    damage_class=detection.damage_class,
                    bbox=detection.bbox,
                    score=detection.score,
                    matched_gt_bbox=matched_gt_bbox,
                    iou=matched_iou,
                )
            )

        for annotation_index, annotation in enumerate(annotations):
            if not matched_gt[annotation_index]:
                counts[ErrorCategory.MISSED] += 1
                cases.append(
                    ErrorCase(
                        image_id=image.image_id,
                        category=ErrorCategory.MISSED,
                        damage_class=annotation.damage_class,
                        bbox=annotation.bbox,
                    )
                )

    return ErrorBreakdown(counts=counts, cases=tuple(cases))
