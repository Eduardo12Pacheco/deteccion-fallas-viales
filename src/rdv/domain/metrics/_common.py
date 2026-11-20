"""Shared validation and indexing for the metric adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rdv.domain.annotation import AnnotatedImage
from rdv.domain.detection import Detection


def validate_iou(iou: float) -> float:
    """Validate an operating IoU threshold in ``(0, 1]``."""
    value = float(iou)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"iou must be within (0, 1], got {iou!r}")
    return value


def validate_score_threshold(score_threshold: float) -> float:
    """Validate an operating score threshold in ``[0, 1]``."""
    value = float(score_threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"score_threshold must be within [0, 1], got {score_threshold!r}")
    return value


def index_ground_truth(
    ground_truth: Sequence[AnnotatedImage],
) -> tuple[AnnotatedImage, ...]:
    """Return images sorted by id, rejecting duplicates."""
    images = tuple(ground_truth)
    seen: set[str] = set()
    for image in images:
        if image.image_id in seen:
            raise ValueError(f"duplicate image_id in ground truth: {image.image_id!r}")
        seen.add(image.image_id)
    return tuple(sorted(images, key=lambda image: image.image_id))


def validate_prediction_ids(
    predictions: Mapping[str, Sequence[Detection]],
    ground_truth: Sequence[AnnotatedImage],
) -> None:
    """Reject predictions that refer to images absent from the ground truth."""
    known = {image.image_id for image in ground_truth}
    unknown = sorted(set(predictions) - known)
    if unknown:
        raise ValueError(f"predictions contain unknown image ids: {unknown!r}")


def ranked_predictions(
    predictions: Mapping[str, Sequence[Detection]],
    image_id: str,
    score_threshold: float,
) -> tuple[Detection, ...]:
    """Filter by score and sort by descending score (stable ties)."""
    kept = [
        detection
        for detection in predictions.get(image_id, ())
        if detection.score >= score_threshold
    ]
    return tuple(sorted(kept, key=lambda detection: -detection.score))
