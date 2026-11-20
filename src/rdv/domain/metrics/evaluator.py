"""Evaluation orchestrator: COCO mAP, per-class AP and PR curves (D10).

The public entry point is :func:`evaluate`. It indexes ground truth by image
and class, matches predictions with :mod:`rdv.domain.metrics.matching`, and
reduces the ranked matches with the 101-point average precision. Categories
without ground truth are excluded from the means, exactly like
``pycocotools``; the returned per-class mapping still contains them with
``None`` metrics so reports can show that they were absent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from rdv.domain.annotation import AnnotatedImage
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.metrics._common import index_ground_truth, validate_prediction_ids
from rdv.domain.metrics.average_precision import (
    AveragePrecision,
    compute_average_precision,
    mean_average_precision,
)
from rdv.domain.metrics.curves import PRCurve, build_pr_curve, thresholds_in_order
from rdv.domain.metrics.matching import MatchedDetections, match_detections

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    float(value) for value in np.linspace(0.5, 0.95, 10)
)
_CURVE_IOU = 0.5


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """AP metrics for one class; ``None`` when the class has no ground truth."""

    damage_class: DamageClass
    num_gt: int
    ap_per_iou: tuple[float, ...]
    ap50: float | None
    ap: float | None
    pr_curve: PRCurve | None

    @property
    def has_ground_truth(self) -> bool:
        """True when at least one ground-truth box of this class was evaluated."""
        return self.num_gt > 0


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """COCO-style evaluation of a labeled image set."""

    iou_thresholds: tuple[float, ...]
    per_class: Mapping[DamageClass, ClassMetrics]
    mAP50: float
    mAP50_95: float
    num_images: int
    num_gt: int

    def __post_init__(self) -> None:
        thresholds = tuple(float(value) for value in self.iou_thresholds)
        if not thresholds:
            raise ValueError("iou_thresholds must not be empty")
        per_class = dict(self.per_class)
        missing = set(DamageClass) - set(per_class)
        if missing:
            raise ValueError(f"per_class is missing classes: {sorted(missing)!r}")
        object.__setattr__(self, "iou_thresholds", thresholds)
        object.__setattr__(self, "per_class", MappingProxyType(per_class))

    def ap50_for(self, damage_class: DamageClass) -> float | None:
        """AP@0.5 for one class, or ``None`` when it has no ground truth."""
        return self.per_class[damage_class].ap50


def _empty_class_metrics(damage_class: DamageClass) -> ClassMetrics:
    return ClassMetrics(
        damage_class=damage_class,
        num_gt=0,
        ap_per_iou=(),
        ap50=None,
        ap=None,
        pr_curve=None,
    )


def _concatenate(chunks: Sequence[np.ndarray], dtype: np.dtype | type) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=dtype)
    return np.concatenate(chunks)


def _evaluate_class(
    images: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    damage_class: DamageClass,
    matching_thresholds: tuple[float, ...],
    *,
    user_threshold_count: int,
    curve_index: int,
    max_dets: int,
) -> ClassMetrics:
    per_image_gt = [
        tuple(
            annotation.bbox
            for annotation in image.annotations
            if annotation.damage_class is damage_class
        )
        for image in images
    ]
    num_gt = sum(len(boxes) for boxes in per_image_gt)
    if num_gt == 0:
        return _empty_class_metrics(damage_class)

    matches: list[MatchedDetections] = []
    for image, gt_boxes in zip(images, per_image_gt, strict=True):
        class_predictions = [
            detection
            for detection in predictions.get(image.image_id, ())
            if detection.damage_class is damage_class
        ]
        matches.append(
            match_detections(
                gt_boxes,
                class_predictions,
                matching_thresholds,
                max_dets=max_dets,
            )
        )

    all_scores = _concatenate([matched.scores for matched in matches], np.float64)
    per_threshold: list[AveragePrecision] = []
    for threshold_index in range(len(matching_thresholds)):
        flags = _concatenate([matched.matches[:, threshold_index] for matched in matches], np.bool_)
        per_threshold.append(compute_average_precision(all_scores, flags, num_gt))

    ap_per_iou = tuple(per_threshold[index].ap for index in range(user_threshold_count))
    curve = per_threshold[curve_index]
    return ClassMetrics(
        damage_class=damage_class,
        num_gt=num_gt,
        ap_per_iou=ap_per_iou,
        ap50=curve.ap,
        ap=mean_average_precision(ap_per_iou),
        pr_curve=build_pr_curve(damage_class, curve, iou_threshold=_CURVE_IOU),
    )


def evaluate(
    ground_truth: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    *,
    max_dets: int = 100,
) -> EvaluationResult:
    """Evaluate predictions against labeled images with COCO semantics.

    Args:
        ground_truth: labeled images; duplicate ids are rejected.
        predictions: detections keyed by image id; unknown ids are rejected and
            missing ids mean "no predictions".
        iou_thresholds: IoU thresholds in ``(0, 1]``; the default is COCO's
            0.50, 0.55, ..., 0.95.
        max_dets: maximum ranked detections per image and class.

    Returns:
        Per-class AP over every threshold, a PR curve at IoU 0.5, mAP@0.5 and
        mAP@0.5:0.95. Classes without ground truth are excluded from the means.

    Raises:
        ValueError: ids, thresholds or ``max_dets`` are invalid.
    """
    thresholds = thresholds_in_order(iou_thresholds)
    if not thresholds:
        raise ValueError("iou_thresholds must contain at least one threshold")
    if max_dets < 1:
        raise ValueError(f"max_dets must be positive, got {max_dets!r}")

    images = index_ground_truth(ground_truth)
    validate_prediction_ids(predictions, images)

    curve_index = next(
        (
            index
            for index, threshold in enumerate(thresholds)
            if math.isclose(threshold, _CURVE_IOU, rel_tol=0.0, abs_tol=1e-12)
        ),
        -1,
    )
    if curve_index < 0:
        matching_thresholds = (*thresholds, _CURVE_IOU)
        curve_index = len(thresholds)
    else:
        matching_thresholds = thresholds

    per_class = {
        damage_class: _evaluate_class(
            images,
            predictions,
            damage_class,
            matching_thresholds,
            user_threshold_count=len(thresholds),
            curve_index=curve_index,
            max_dets=max_dets,
        )
        for damage_class in DamageClass
    }

    present_ap50 = [metrics.ap50 for metrics in per_class.values() if metrics.ap50 is not None]
    present_ap = [metrics.ap for metrics in per_class.values() if metrics.ap is not None]
    return EvaluationResult(
        iou_thresholds=thresholds,
        per_class=per_class,
        mAP50=mean_average_precision(present_ap50),
        mAP50_95=mean_average_precision(present_ap),
        num_images=len(images),
        num_gt=sum(len(image.annotations) for image in images),
    )


def by_prefix(image_ids: Iterable[str], *, separator: str = "_") -> dict[str, tuple[str, ...]]:
    """Group image ids by the leading token of their file name.

    RDD2022 ids encode the country (for example ``Japan_000001``), so grouping
    the ids lets the evaluation use case re-run :func:`evaluate` per country
    without a country column on every annotation.

    Args:
        image_ids: arbitrary image identifiers; relative paths are allowed.
        separator: token separator, normally ``_``.

    Returns:
        Sorted prefix -> sorted ids mapping.

    Raises:
        ValueError: the separator is empty or an id is blank.
    """
    if not separator:
        raise ValueError("separator must not be empty")
    groups: dict[str, list[str]] = {}
    for raw_image_id in image_ids:
        image_id = str(raw_image_id)
        if not image_id.strip():
            raise ValueError("image ids must be non-empty strings")
        file_name = image_id.replace("\\", "/").rsplit("/", 1)[-1]
        prefix = file_name.split(separator, 1)[0] or file_name
        groups.setdefault(prefix, []).append(image_id)
    return {prefix: tuple(sorted(ids)) for prefix, ids in sorted(groups.items())}
