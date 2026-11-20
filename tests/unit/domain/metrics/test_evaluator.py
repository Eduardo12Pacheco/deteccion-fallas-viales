"""Evaluation orchestration, per-class metrics and image grouping (D10)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.metrics.curves import PRCurve
from rdv.domain.metrics.evaluator import (
    DEFAULT_IOU_THRESHOLDS,
    EvaluationResult,
    by_prefix,
    evaluate,
)

Box = tuple[float, float, float, float]


def _image(
    image_id: str,
    *annotations: tuple[DamageClass, Box],
    width: int = 100,
    height: int = 100,
) -> AnnotatedImage:
    return AnnotatedImage(
        image_id=image_id,
        width=width,
        height=height,
        annotations=tuple(
            Annotation(damage_class, BBox(*box)) for damage_class, box in annotations
        ),
    )


def _detection(damage_class: DamageClass, box: Box, score: float = 0.9) -> Detection:
    return Detection(damage_class, score, BBox(*box))


def test_perfect_predictions_on_every_class_score_one() -> None:
    gt = [
        _image("img_1", (DamageClass.D00, (0, 0, 10, 10))),
        _image("img_2", (DamageClass.D10, (10, 10, 20, 20))),
        _image("img_3", (DamageClass.D20, (20, 20, 30, 30))),
        _image("img_4", (DamageClass.D40, (30, 30, 40, 40))),
    ]
    preds = {
        "img_1": [_detection(DamageClass.D00, (0, 0, 10, 10))],
        "img_2": [_detection(DamageClass.D10, (10, 10, 20, 20))],
        "img_3": [_detection(DamageClass.D20, (20, 20, 30, 30))],
        "img_4": [_detection(DamageClass.D40, (30, 30, 40, 40))],
    }

    result = evaluate(gt, preds)

    assert isinstance(result, EvaluationResult)
    assert result.mAP50 == pytest.approx(1.0, abs=1e-12)
    assert result.mAP50_95 == pytest.approx(1.0, abs=1e-12)
    assert result.num_images == 4
    assert result.num_gt == 4
    for damage_class in DamageClass:
        metrics = result.per_class[damage_class]
        assert metrics.num_gt == 1
        assert metrics.ap50 == pytest.approx(1.0, abs=1e-12)
        assert isinstance(metrics.pr_curve, PRCurve)
        assert metrics.pr_curve.iou_threshold == pytest.approx(0.5)


def test_no_predictions_scores_zero() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]

    result = evaluate(gt, {})

    assert result.mAP50 == 0.0
    assert result.mAP50_95 == 0.0
    assert result.per_class[DamageClass.D00].ap50 == 0.0


def test_classes_without_ground_truth_are_excluded_from_the_mean() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 10, 10))]}

    result = evaluate(gt, preds)

    absent = result.per_class[DamageClass.D10]
    assert absent.num_gt == 0
    assert absent.ap is None
    assert absent.ap50 is None
    assert absent.ap_per_iou == ()
    assert absent.pr_curve is None
    assert result.mAP50 == pytest.approx(1.0, abs=1e-12)


def test_mixing_perfect_and_missed_classes_averages_the_ap() -> None:
    gt = [
        _image("img_1", (DamageClass.D00, (0, 0, 10, 10))),
        _image("img_2", (DamageClass.D40, (20, 20, 30, 30))),
    ]
    preds = {"img_2": [_detection(DamageClass.D40, (20, 20, 30, 30))]}

    result = evaluate(gt, preds)

    assert result.mAP50 == pytest.approx(0.5)
    assert result.mAP50_95 == pytest.approx(0.5)


def test_custom_thresholds_still_report_a_pr_curve_at_iou_half() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 10, 10))]}

    result = evaluate(gt, preds, (0.75,))

    metrics = result.per_class[DamageClass.D00]
    assert metrics.pr_curve is not None
    assert metrics.pr_curve.iou_threshold == pytest.approx(0.5)
    assert metrics.ap_per_iou == (pytest.approx(1.0, abs=1e-12),)
    assert result.mAP50 == pytest.approx(1.0, abs=1e-12)
    assert result.mAP50_95 == pytest.approx(1.0, abs=1e-12)


def test_max_detections_can_drop_a_later_true_positive() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {
        "img_1": [
            _detection(DamageClass.D00, (50, 50, 60, 60), score=0.9),
            _detection(DamageClass.D00, (0, 0, 10, 10), score=0.8),
        ]
    }

    unlimited = evaluate(gt, preds)
    capped = evaluate(gt, preds, max_dets=1)

    assert unlimited.mAP50 == pytest.approx(0.5)
    assert capped.mAP50 == 0.0


def test_evaluation_is_deterministic() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {
        "img_1": [
            _detection(DamageClass.D00, (0, 0, 10, 10), score=0.7),
            _detection(DamageClass.D00, (50, 50, 60, 60), score=0.9),
        ]
    }

    first = evaluate(gt, preds)
    second = evaluate(gt, preds)

    assert first.mAP50 == second.mAP50
    assert first.per_class[DamageClass.D00].pr_curve == second.per_class[DamageClass.D00].pr_curve


def test_empty_ground_truth_scores_zero() -> None:
    result = evaluate([], {})

    assert result.mAP50 == 0.0
    assert result.mAP50_95 == 0.0
    assert result.num_images == 0
    assert result.num_gt == 0


def test_default_thresholds_match_coco() -> None:
    assert DEFAULT_IOU_THRESHOLDS[0] == pytest.approx(0.5)
    assert DEFAULT_IOU_THRESHOLDS[-1] == pytest.approx(0.95)
    assert len(DEFAULT_IOU_THRESHOLDS) == 10


def test_predictions_for_unknown_images_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown image"):
        evaluate([], {"nope": [_detection(DamageClass.D00, (0, 0, 1, 1))]})


def test_duplicate_ground_truth_ids_are_rejected() -> None:
    gt = [_image("img_1"), _image("img_1")]

    with pytest.raises(ValueError, match="duplicate"):
        evaluate(gt, {})


@pytest.mark.parametrize("max_dets", [0, -1])
def test_rejects_non_positive_max_detections(max_dets: int) -> None:
    with pytest.raises(ValueError, match="max_dets"):
        evaluate([], {}, max_dets=max_dets)


@pytest.mark.parametrize("thresholds", [(), (0.0,), (1.1,), (float("nan"),)])
def test_rejects_invalid_thresholds(thresholds: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        evaluate([], {}, thresholds)


def test_per_class_mapping_is_read_only() -> None:
    result = evaluate([], {})

    with pytest.raises(TypeError):
        result.per_class[DamageClass.D00] = result.per_class[DamageClass.D00]  # type: ignore[index]


def test_by_prefix_groups_image_ids_by_leading_token() -> None:
    groups = by_prefix(["Japan_000002", "India_000001", "Japan_000001"])

    assert groups == {
        "India": ("India_000001",),
        "Japan": ("Japan_000001", "Japan_000002"),
    }


def test_by_prefix_uses_the_file_name_of_relative_paths() -> None:
    groups = by_prefix(["Japan/train/images/Japan_000001.jpg", "Czech/train/images/Czech_1.jpg"])

    assert groups == {
        "Czech": ("Czech/train/images/Czech_1.jpg",),
        "Japan": ("Japan/train/images/Japan_000001.jpg",),
    }


def test_by_prefix_supports_a_custom_separator() -> None:
    groups = by_prefix(["Japan-1", "India-2"], separator="-")

    assert groups == {"India": ("India-2",), "Japan": ("Japan-1",)}


def test_by_prefix_rejects_an_empty_separator() -> None:
    with pytest.raises(ValueError, match="separator"):
        by_prefix(["Japan_1"], separator="")


def test_by_prefix_keeps_ids_without_the_separator() -> None:
    assert by_prefix(["Norway"]) == {"Norway": ("Norway",)}


def test_grouped_evaluation_matches_the_per_country_numbers() -> None:
    gt = [
        _image("Japan_000001", (DamageClass.D00, (0, 0, 10, 10))),
        _image("India_000001", (DamageClass.D00, (0, 0, 10, 10))),
    ]
    preds = {"Japan_000001": [_detection(DamageClass.D00, (0, 0, 10, 10))]}
    groups = by_prefix([image.image_id for image in gt])

    by_country = {
        name: evaluate(
            [image for image in gt if image.image_id in ids],
            {key: value for key, value in preds.items() if key in ids},
        )
        for name, ids in groups.items()
    }

    assert by_country["Japan"].mAP50 == pytest.approx(1.0, abs=1e-12)
    assert by_country["India"].mAP50 == 0.0


def _synthetic_case(seed: int) -> tuple[list[AnnotatedImage], dict[str, list[Detection]]]:
    rng = np.random.default_rng(seed)
    images: list[AnnotatedImage] = []
    predictions: dict[str, list[Detection]] = {}
    num_images = int(rng.integers(1, 6))
    for image_index in range(num_images):
        image_id = f"img_{image_index:04d}"
        num_gt = int(rng.integers(0, 5))
        boxes = []
        for _ in range(num_gt):
            x1 = float(rng.uniform(0, 100))
            y1 = float(rng.uniform(0, 100))
            boxes.append((x1, y1, x1 + float(rng.uniform(4, 40)), y1 + float(rng.uniform(4, 40))))
        annotations = tuple(
            Annotation(DamageClass(int(rng.integers(0, 4))), BBox(*box)) for box in boxes
        )
        images.append(
            AnnotatedImage(image_id=image_id, width=160, height=160, annotations=annotations)
        )
        num_pred = int(rng.integers(0, 5))
        scores = list(rng.permutation(np.linspace(0.05, 0.99, num_pred))) if num_pred else []
        predictions[image_id] = []
        for index in range(num_pred):
            x1 = float(rng.uniform(0, 120))
            y1 = float(rng.uniform(0, 120))
            predictions[image_id].append(
                Detection(
                    DamageClass(int(rng.integers(0, 4))),
                    float(scores[index]),
                    BBox(x1, y1, x1 + float(rng.uniform(4, 40)), y1 + float(rng.uniform(4, 40))),
                )
            )
    return images, predictions


@given(st.integers(min_value=0, max_value=100_000))
def test_ap_values_are_bounded(seed: int) -> None:
    images, predictions = _synthetic_case(seed)

    result = evaluate(images, predictions)

    assert 0.0 <= result.mAP50 <= 1.0
    assert 0.0 <= result.mAP50_95 <= 1.0
    for metrics in result.per_class.values():
        if metrics.ap is not None:
            assert 0.0 <= metrics.ap <= 1.0
        if metrics.ap50 is not None:
            assert 0.0 <= metrics.ap50 <= 1.0
        for value in metrics.ap_per_iou:
            assert 0.0 <= value <= 1.0


@given(st.integers(min_value=0, max_value=100_000))
def test_ap_is_invariant_to_detection_input_order(seed: int) -> None:
    images, predictions = _synthetic_case(seed)
    reversed_predictions = {
        image_id: list(reversed(detections)) for image_id, detections in predictions.items()
    }

    first = evaluate(images, predictions)
    second = evaluate(images, reversed_predictions)

    assert first.mAP50 == pytest.approx(second.mAP50, abs=1e-12)
    assert first.mAP50_95 == pytest.approx(second.mAP50_95, abs=1e-12)


@given(st.integers(min_value=0, max_value=100_000))
def test_added_low_score_false_positive_never_raises_ap(seed: int) -> None:
    images, predictions = _synthetic_case(seed)
    annotated = [image for image in images if image.annotations]
    assume(bool(annotated))
    target = annotated[0]
    damage_class = target.annotations[0].damage_class
    baseline = evaluate(images, predictions)

    augmented = {image_id: list(dets) for image_id, dets in predictions.items()}
    # Coordinates in the bottom-right corner cannot overlap the boxes produced
    # by the synthetic generator (which stay in the upper-left 140 px), and the
    # score is below every generated score.
    augmented[target.image_id].append(Detection(damage_class, 1e-6, BBox(150, 150, 155, 155)))
    updated = evaluate(images, augmented)

    assert updated.mAP50 <= baseline.mAP50 + 1e-12
    assert updated.mAP50_95 <= baseline.mAP50_95 + 1e-12


def test_predictions_mapping_may_omit_images() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds: Mapping[str, list[Detection]] = {}

    result = evaluate(gt, preds)

    assert result.per_class[DamageClass.D00].ap50 == 0.0
