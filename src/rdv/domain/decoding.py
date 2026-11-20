"""Pure decode of raw DETR outputs into pixel-space detections (decision D8).

The detector emits one score per query and class (pre-sigmoid) and boxes in
normalized ``(cx, cy, w, h)`` coordinates relative to the stretched network
input. Decoding is a domain concern: it must stay deterministic and must never
emit a box outside the original image.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox

_NUM_CLASSES = len(DamageClass)


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative = ~positive
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    return result


def decode_detr_outputs(
    logits: ArrayLike,
    pred_boxes: ArrayLike,
    image_size: tuple[int, int],
    *,
    score_threshold: float = 0.0,
    max_detections: int = 100,
) -> list[Detection]:
    """Decode raw model tensors into detections sorted by descending score.

    Args:
        logits: ``(num_queries, num_classes)`` pre-sigmoid class scores.
        pred_boxes: ``(num_queries, 4)`` normalized ``(cx, cy, w, h)`` boxes.
        image_size: original image size as ``(width, height)`` in pixels.
        score_threshold: minimum sigmoid score to keep, in ``[0, 1]``.
        max_detections: maximum number of detections, 1..100 is the useful range.

    Returns:
        Detections with pixel-space, clipped boxes of at least 1x1 pixel,
        ordered by descending score (ties keep query-then-class order).

    Raises:
        ValueError: the arrays are malformed, the image size is not positive,
            or the decoding parameters are out of range.
    """
    scores = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(pred_boxes, dtype=np.float64)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_size!r}")
    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError(f"score_threshold must be within [0, 1], got {score_threshold!r}")
    if max_detections < 1:
        raise ValueError(f"max_detections must be positive, got {max_detections!r}")
    if scores.size == 0 and boxes.size == 0:
        return []
    if scores.ndim != 2:
        raise ValueError(f"logits must be two-dimensional, got shape {scores.shape}")
    if scores.shape[1] != _NUM_CLASSES:
        raise ValueError(f"logits must have {_NUM_CLASSES} classes, got {scores.shape[1]}")
    if boxes.shape != (scores.shape[0], 4):
        raise ValueError(f"pred_boxes must have shape ({scores.shape[0]}, 4), got {boxes.shape}")
    if not np.all(np.isfinite(boxes)):
        raise ValueError("pred_boxes must be finite")
    if scores.shape[0] == 0:
        return []

    probabilities = _sigmoid(scores)
    flat_scores = probabilities.reshape(-1)
    # Stable sort keeps query-major, class-minor order for equal scores.
    ranked = np.argsort(-flat_scores, kind="stable")[:max_detections]

    detections: list[Detection] = []
    for flat_index in ranked:
        score = float(flat_scores[flat_index])
        if score < score_threshold:
            continue
        query_index = int(flat_index) // _NUM_CLASSES
        class_id = int(flat_index) % _NUM_CLASSES
        center_x, center_y, box_width, box_height = (float(value) for value in boxes[query_index])
        if box_width <= 0.0 or box_height <= 0.0:
            continue

        half_width = box_width * width / 2.0
        half_height = box_height * height / 2.0
        center_x *= width
        center_y *= height
        box = BBox(
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ).clipped(width, height)
        if box.width < 1.0 or box.height < 1.0:
            continue
        detections.append(Detection(DamageClass(class_id), score, box))
    return detections
