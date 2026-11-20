"""Evaluate a detector on a labeled split and assemble the S7 report set.

The use case owns orchestration only: the detector, dataset repository and
report writer are ports. Metrics come from the pure domain engine, the
operating threshold is selected on a separate validation repository so the
test split is never used for selection (decision D10), and the rendered files
are produced by the reporting adapter.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rdv import __version__
from rdv.application.events import ProgressEvent
from rdv.application.ports import (
    DatasetRepository,
    Detector,
    EvaluationArtifacts,
    EvaluationReport,
    EvaluationReportWriter,
    GalleryCase,
    ProgressSink,
    RgbImage,
)
from rdv.domain.annotation import AnnotatedImage
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.metrics.confusion import ConfusionMatrix, confusion_matrix
from rdv.domain.metrics.curves import ThresholdSelection, best_f1_threshold
from rdv.domain.metrics.error_analysis import (
    ErrorBreakdown,
    ErrorCase,
    ErrorCategory,
    error_breakdown,
)
from rdv.domain.metrics.evaluator import EvaluationResult, by_prefix, evaluate
from rdv.domain.metrics.matching import match_detections
from rdv.errors import ConfigError

SCHEMA = "rdv.metrics/v1"
ERROR_SCHEMA = "rdv.error-analysis/v1"
TOOL_NAME = "road-damage-vision"
EVALUATION_SCORE_THRESHOLD = 0.001
"""Low cutoff used while collecting detections for the metrics engine (S7)."""

MAX_GALLERY_CASES = 24
GALLERY_PER_CATEGORY = 6
_CATEGORY_ORDER: tuple[ErrorCategory, ...] = (
    ErrorCategory.CLASSIFICATION,
    ErrorCategory.LOCALIZATION,
    ErrorCategory.DUPLICATE,
    ErrorCategory.BACKGROUND,
    ErrorCategory.MISSED,
)
_CLASS_ORDER = tuple(DamageClass)


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Result of one ``rdv evaluate`` run."""

    output_dir: Path
    metrics: EvaluationResult
    operating_threshold: float
    operating_source: str
    images_evaluated: int
    artifacts: EvaluationArtifacts


def round_value(value: float | None, digits: int = 6) -> float | None:
    """Round a metric for stable JSON, preserving ``None``."""
    if value is None:
        return None
    return round(float(value), digits)


def _class_metrics_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        damage_class.code: {
            "num_gt": result.per_class[damage_class].num_gt,
            "ap_50": round_value(result.per_class[damage_class].ap50),
            "ap_50_95": round_value(result.per_class[damage_class].ap),
        }
        for damage_class in _CLASS_ORDER
    }


def _metrics_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "map_50": round_value(result.mAP50),
        "map_50_95": round_value(result.mAP50_95),
        "iou_thresholds": [round_value(value) for value in result.iou_thresholds],
        "per_class": _class_metrics_payload(result),
    }


def _class_counts(images: Sequence[AnnotatedImage]) -> dict[str, int]:
    counts = dict.fromkeys((damage_class.code for damage_class in _CLASS_ORDER), 0)
    for image in images:
        for ground_truth in image.annotations:
            counts[ground_truth.damage_class.code] += 1
    return counts


def _country_counts(images: Sequence[AnnotatedImage]) -> dict[str, int]:
    groups = by_prefix([image.image_id for image in images])
    return {prefix: len(ids) for prefix, ids in groups.items()}


def _group_metrics(
    images: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
) -> dict[str, Any]:
    groups = by_prefix([image.image_id for image in images])
    by_id = {image.image_id: image for image in images}
    result: dict[str, Any] = {}
    for prefix, ids in groups.items():
        subset = [by_id[image_id] for image_id in ids]
        group_predictions = {image_id: predictions.get(image_id, ()) for image_id in ids}
        evaluation = evaluate(subset, group_predictions)
        result[prefix] = {
            "images": len(subset),
            "ground_truth_boxes": evaluation.num_gt,
            **_metrics_payload(evaluation),
        }
    return result


def _run_detector(
    repository: DatasetRepository,
    detector: Detector,
    *,
    stage: str,
    max_images: int | None = None,
    progress: ProgressSink | None = None,
) -> tuple[tuple[AnnotatedImage, ...], dict[str, tuple[Detection, ...]]]:
    images = tuple(sorted(repository, key=lambda image: image.image_id))
    if not images:
        raise ConfigError("The selected dataset split contains no labeled images")
    if max_images is not None:
        images = images[:max_images]
    predictions: dict[str, tuple[Detection, ...]] = {}
    total = len(images)
    for index, image in enumerate(images, start=1):
        pixels: RgbImage = repository.load_image(image.image_id)
        batch = detector.detect([pixels])
        if len(batch) != 1:
            raise ValueError(f"Detector returned {len(batch)} results for one image")
        predictions[image.image_id] = tuple(batch[0])
        if progress is not None:
            progress.emit(
                ProgressEvent(
                    stage=stage,
                    current=index,
                    total=total,
                    message=f"{image.image_id}: {len(batch[0])} detection(s)",
                )
            )
    return images, predictions


def _ranked_matches(
    ground_truth: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    *,
    iou: float,
    score_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    images = tuple(sorted(ground_truth, key=lambda image: image.image_id))
    num_gt = sum(len(image.annotations) for image in images)
    scores: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    for damage_class in _CLASS_ORDER:
        for image in images:
            gt_boxes = [
                annotation.bbox
                for annotation in image.annotations
                if annotation.damage_class is damage_class
            ]
            detections = [
                detection
                for detection in predictions.get(image.image_id, ())
                if detection.damage_class is damage_class
                and (score_threshold is None or detection.score >= score_threshold)
            ]
            if not gt_boxes and not detections:
                continue
            matched = match_detections(gt_boxes, detections, (iou,))
            scores.append(matched.scores)
            flags.append(matched.matches[:, 0])
    return (
        np.concatenate(scores) if scores else np.empty(0, dtype=np.float64),
        np.concatenate(flags) if flags else np.empty(0, dtype=np.bool_),
        num_gt,
    )


def select_operating_threshold(
    ground_truth: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    *,
    iou: float = 0.5,
) -> ThresholdSelection:
    """Choose the global F1-optimal threshold at IoU 0.5 (D10).

    All classes are pooled so a single threshold can be applied to the report
    and the confusion matrix. The caller is responsible for passing the
    validation split, never the test split.
    """
    scores, flags, num_gt = _ranked_matches(ground_truth, predictions, iou=iou)
    if num_gt == 0:
        return ThresholdSelection(
            threshold=0.0, precision=0.0, recall=0.0, f1=0.0, true_positives=0, false_positives=0
        )
    return best_f1_threshold(scores, flags, num_gt)


def operating_point(
    ground_truth: Sequence[AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    threshold: float,
    *,
    iou: float = 0.5,
) -> ThresholdSelection:
    """Compute precision/recall/F1 on an explicit operating threshold."""
    _scores, flags, num_gt = _ranked_matches(
        ground_truth, predictions, iou=iou, score_threshold=threshold
    )
    true_positives = int(flags.sum())
    false_positives = int(flags.size - true_positives)
    precision = true_positives / flags.size if flags.size else 0.0
    recall = true_positives / num_gt if num_gt else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return ThresholdSelection(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
    )


def _case_sort_key(case: ErrorCase) -> tuple[Any, ...]:
    box = case.bbox
    if case.category is ErrorCategory.MISSED:
        return (-box.area, case.image_id, int(case.damage_class), box.to_xyxy())
    return (-(case.score or 0.0), case.image_id, int(case.damage_class), box.to_xyxy())


def select_gallery_cases(
    cases: Sequence[ErrorCase],
    images: Mapping[str, AnnotatedImage],
    predictions: Mapping[str, Sequence[Detection]],
    *,
    per_category: int = GALLERY_PER_CATEGORY,
    total: int = MAX_GALLERY_CASES,
) -> tuple[GalleryCase, ...]:
    """Select deterministic gallery cases, balanced across non-empty categories.

    False positives are ranked by score and misses by ground-truth area; the
    round-robin pass guarantees a non-empty category is represented before any
    category gets its sixth case.
    """
    buckets: dict[ErrorCategory, list[ErrorCase]] = {category: [] for category in _CATEGORY_ORDER}
    for case in cases:
        buckets[case.category].append(case)
    for category in _CATEGORY_ORDER:
        buckets[category] = sorted(buckets[category], key=_case_sort_key)[:per_category]

    selected: list[ErrorCase] = []
    cursors = dict.fromkeys(_CATEGORY_ORDER, 0)
    while len(selected) < total:
        added = False
        for category in _CATEGORY_ORDER:
            bucket = buckets[category]
            cursor = cursors[category]
            if cursor >= len(bucket):
                continue
            selected.append(bucket[cursor])
            cursors[category] = cursor + 1
            added = True
            if len(selected) >= total:
                break
        if not added:
            break

    gallery: list[GalleryCase] = []
    for case in selected:
        image = images[case.image_id]
        is_missed = case.category is ErrorCategory.MISSED
        gallery.append(
            GalleryCase(
                image_id=case.image_id,
                category=case.category,
                damage_class=case.damage_class,
                ground_truth=image.annotations,
                detections=tuple(predictions.get(case.image_id, ())),
                detection_bbox=None if is_missed else case.bbox,
                ground_truth_bbox=case.matched_gt_bbox if not is_missed else case.bbox,
                score=case.score,
                iou=case.iou,
            )
        )
    return tuple(gallery)


def _error_case_payload(case: ErrorCase) -> dict[str, Any]:
    return {
        "image_id": case.image_id,
        "category": case.category.value,
        "class": case.damage_class.code,
        "score": round_value(case.score),
        "box": [round_value(value) for value in case.bbox.to_xyxy()],
        "matched_gt_box": (
            [round_value(value) for value in case.matched_gt_bbox.to_xyxy()]
            if case.matched_gt_bbox is not None
            else None
        ),
        "iou": round_value(case.iou),
    }


def _build_document(
    *,
    images: Sequence[AnnotatedImage],
    result: EvaluationResult,
    predictions: Mapping[str, Sequence[Detection]],
    model_id: str,
    model_sha256: str | None,
    model_input_size: int,
    device: str,
    split: str,
    score_threshold: float,
    operating: ThresholdSelection,
    operating_source: str,
    max_images: int | None,
    confusion: ConfusionMatrix,
    errors: ErrorBreakdown,
    ood: EvaluationResult | None,
    gallery_count: int,
    gallery_by_category: Mapping[ErrorCategory, int],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": __version__},
        "model": {
            "id": model_id,
            "sha256": model_sha256,
            "input_size": model_input_size,
        },
        "dataset": {
            "split": split,
            "images": len(images),
            "ground_truth_boxes": result.num_gt,
            "classes": _class_counts(images),
            "countries": _country_counts(images),
            "score_threshold": round_value(score_threshold),
        },
        "params": {
            "split": split,
            "score_threshold": round_value(score_threshold),
            "operating_threshold": round_value(operating.threshold),
            "operating_source": operating_source,
            "device": device,
            "max_images": max_images,
        },
        "metrics": _metrics_payload(result),
        "operating_point": {
            "threshold": round_value(operating.threshold),
            "precision": round_value(operating.precision),
            "recall": round_value(operating.recall),
            "f1": round_value(operating.f1),
            "true_positives": operating.true_positives,
            "false_positives": operating.false_positives,
        },
        "confusion_matrix": {
            "labels": list(confusion.labels),
            "values": confusion.matrix.tolist(),
        },
        "errors": {
            "counts": {category.value: errors.counts[category] for category in _CATEGORY_ORDER},
            "total_false_positives": errors.total_false_positives,
            "total_missed": errors.total_missed,
        },
        "per_country": _group_metrics(images, predictions),
        "ood": (
            {
                "images": ood.num_images,
                "ground_truth_boxes": ood.num_gt,
                **_metrics_payload(ood),
            }
            if ood is not None
            else None
        ),
        "gallery": {
            "total_cases": gallery_count,
            "per_category": {
                category.value: gallery_by_category.get(category, 0) for category in _CATEGORY_ORDER
            },
        },
    }


def _build_error_document(
    *,
    errors: ErrorBreakdown,
    split: str,
    score_threshold: float,
    operating_threshold: float,
) -> dict[str, Any]:
    return {
        "schema": ERROR_SCHEMA,
        "tool": {"name": TOOL_NAME, "version": __version__},
        "params": {
            "split": split,
            "score_threshold": round_value(score_threshold),
            "operating_threshold": round_value(operating_threshold),
            "iou": 0.5,
        },
        "counts": {category.value: errors.counts[category] for category in _CATEGORY_ORDER},
        "total_false_positives": errors.total_false_positives,
        "total_missed": errors.total_missed,
        "cases": [_error_case_payload(case) for case in errors.cases],
    }


def evaluate_model(
    dataset: DatasetRepository,
    *,
    detector: Detector,
    writer: EvaluationReportWriter,
    output_dir: Path,
    selection_dataset: DatasetRepository | None = None,
    ood_dataset: DatasetRepository | None = None,
    split: str = "test",
    score_threshold: float = EVALUATION_SCORE_THRESHOLD,
    operating_threshold: float | None = None,
    max_images: int | None = None,
    progress: ProgressSink | None = None,
) -> EvaluationOutcome:
    """Evaluate the detector on one split and write the S7 artifacts.

    Args:
        dataset: repository over the split being evaluated.
        detector: inference port; its own score cutoff should be
            :data:`EVALUATION_SCORE_THRESHOLD` so the metrics engine sees the
            ranked detections.
        writer: report adapter.
        output_dir: directory receiving every artifact.
        selection_dataset: validation repository used to pick the F1-optimal
            threshold; required unless ``operating_threshold`` is explicit.
        ood_dataset: optional out-of-domain labeled set (for example Norway).
        split: split label recorded in the report (``test``/``val``/``all``).
        score_threshold: detection cutoff recorded as provenance.
        operating_threshold: explicit operating threshold; ``None`` selects it
            on ``selection_dataset``.
        max_images: cap on evaluated images, applied after id sorting.
        progress: optional observer of per-image progress.

    Returns:
        The metrics result, the chosen threshold and the written paths.

    Raises:
        ConfigError: automatic threshold selection has no validation split, or
            the evaluation split is empty.
    """
    if operating_threshold is not None and not 0.0 <= operating_threshold <= 1.0:
        raise ConfigError("operating threshold must be within [0, 1]")

    selected_images: tuple[AnnotatedImage, ...] | None = None
    selected_predictions: dict[str, tuple[Detection, ...]] | None = None
    selection: ThresholdSelection | None = None
    if operating_threshold is None:
        if selection_dataset is None:
            raise ConfigError("Automatic threshold selection requires the validation split")
        selection_images, selection_predictions = _run_detector(
            selection_dataset,
            detector,
            stage="threshold",
            progress=progress,
        )
        selection = select_operating_threshold(selection_images, selection_predictions)
        if selection_dataset is dataset and split == "val":
            selected_images = selection_images
            selected_predictions = selection_predictions

    if selected_images is None or selected_predictions is None:
        selected_images, selected_predictions = _run_detector(
            dataset,
            detector,
            stage="evaluate",
            max_images=max_images,
            progress=progress,
        )

    if operating_threshold is None:
        assert selection is not None
        threshold = selection.threshold
    else:
        threshold = float(operating_threshold)
    operating = (
        selection
        if selection is not None
        else operating_point(selected_images, selected_predictions, threshold)
    )
    result = evaluate(selected_images, selected_predictions)
    confusion = confusion_matrix(
        selected_images, selected_predictions, iou=0.5, score_threshold=threshold
    )
    errors = error_breakdown(
        selected_images, selected_predictions, iou=0.5, score_threshold=threshold
    )
    image_index = {image.image_id: image for image in selected_images}
    thresholded = {
        image_id: tuple(detection for detection in detections if detection.score >= threshold)
        for image_id, detections in selected_predictions.items()
    }
    gallery = select_gallery_cases(errors.cases, image_index, thresholded)
    gallery_by_category = Counter(case.category for case in gallery)

    ood_result: EvaluationResult | None = None
    if ood_dataset is not None:
        ood_images, ood_predictions = _run_detector(
            ood_dataset,
            detector,
            stage="ood",
            progress=progress,
        )
        ood_result = evaluate(ood_images, ood_predictions)

    info = detector.info
    operating_source = "val" if selection is not None else "explicit"
    document = _build_document(
        images=selected_images,
        result=result,
        predictions=selected_predictions,
        model_id=info.id,
        model_sha256=info.sha256,
        model_input_size=info.input_size,
        device=info.device,
        split=split,
        score_threshold=score_threshold,
        operating=operating,
        operating_source=operating_source,
        max_images=max_images,
        confusion=confusion,
        errors=errors,
        ood=ood_result,
        gallery_count=len(gallery),
        gallery_by_category=gallery_by_category,
    )
    error_document = _build_error_document(
        errors=errors,
        split=split,
        score_threshold=score_threshold,
        operating_threshold=threshold,
    )
    report = EvaluationReport(
        document=document,
        error_document=error_document,
        pr_curves={
            damage_class: metrics.pr_curve
            for damage_class in _CLASS_ORDER
            if (metrics := result.per_class[damage_class]).pr_curve is not None
        },
        confusion=confusion,
        gallery=gallery,
    )
    artifacts = writer.write_report(Path(output_dir), report, load_image=dataset.load_image)
    return EvaluationOutcome(
        output_dir=Path(output_dir),
        metrics=result,
        operating_threshold=threshold,
        operating_source=operating_source,
        images_evaluated=len(selected_images),
        artifacts=artifacts,
    )
