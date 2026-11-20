"""5x5 confusion matrix with background (decision D10)."""

from __future__ import annotations

import numpy as np
import pytest

from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.metrics.confusion import (
    BACKGROUND_LABEL,
    CONFUSION_LABELS,
    ConfusionMatrix,
    confusion_matrix,
)


def _image(image_id: str, *annotations: tuple[DamageClass, tuple[float, float, float, float]]):
    return AnnotatedImage(
        image_id=image_id,
        width=100,
        height=100,
        annotations=tuple(
            Annotation(damage_class, BBox(*box)) for damage_class, box in annotations
        ),
    )


def _detection(damage_class: DamageClass, box: tuple[float, float, float, float], score=0.9):
    return Detection(damage_class, score, BBox(*box))


def test_labels_have_four_classes_and_background() -> None:
    assert CONFUSION_LABELS == ("D00", "D10", "D20", "D40", BACKGROUND_LABEL)


def test_perfect_predictions_fill_the_diagonal() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)), (DamageClass.D40, (20, 20, 40, 40)))]
    preds = {
        "img_1": [
            _detection(DamageClass.D00, (0, 0, 10, 10)),
            _detection(DamageClass.D40, (20, 20, 40, 40)),
        ]
    }

    result = confusion_matrix(gt, preds)

    expected = np.zeros((5, 5), dtype=np.int64)
    expected[0, 0] = 1
    expected[3, 3] = 1
    assert isinstance(result, ConfusionMatrix)
    assert np.array_equal(result.matrix, expected)
    assert result.total == 2


def test_missed_ground_truth_goes_to_the_background_column() -> None:
    gt = [_image("img_1", (DamageClass.D10, (0, 0, 10, 10)))]

    result = confusion_matrix(gt, {})

    assert result.matrix[1, 4] == 1
    assert result.total == 1


def test_false_positive_goes_to_the_background_row() -> None:
    gt = [_image("img_1")]
    preds = {"img_1": [_detection(DamageClass.D20, (50, 50, 60, 60))]}

    result = confusion_matrix(gt, preds)

    assert result.matrix[4, 2] == 1


def test_cross_class_match_is_off_diagonal() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D40, (0, 0, 10, 10))]}

    result = confusion_matrix(gt, preds)

    assert result.matrix[0, 3] == 1
    assert result.matrix[0, 4] == 0


def test_class_agnostic_matching_uses_the_best_overlap() -> None:
    gt = [
        _image(
            "img_1",
            (DamageClass.D00, (0, 0, 10, 10)),
            (DamageClass.D10, (20, 0, 30, 10)),
        )
    ]
    preds = {"img_1": [_detection(DamageClass.D00, (20, 0, 30, 10))]}

    result = confusion_matrix(gt, preds)

    assert result.matrix[1, 0] == 1
    assert result.matrix[0, 1] == 0


def test_low_score_predictions_are_filtered_by_the_operating_threshold() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 10, 10), score=0.2)]}

    result = confusion_matrix(gt, preds, score_threshold=0.5)

    assert result.matrix[0, 4] == 1
    assert result.matrix[4, 0] == 0
    assert result.total == 1


def test_boxes_below_the_iou_threshold_do_not_match() -> None:
    gt = [_image("img_1", (DamageClass.D00, (0, 0, 10, 10)))]
    preds = {"img_1": [_detection(DamageClass.D00, (0, 0, 3, 10))]}

    result = confusion_matrix(gt, preds, iou=0.5)

    assert result.matrix[0, 4] == 1
    assert result.matrix[4, 0] == 1


def test_empty_input_gives_a_zero_matrix() -> None:
    result = confusion_matrix([], {})

    assert result.matrix.shape == (5, 5)
    assert result.matrix.sum() == 0


def test_predictions_for_unknown_images_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown image"):
        confusion_matrix([], {"nope": [_detection(DamageClass.D00, (0, 0, 1, 1))]})


def test_duplicate_image_ids_are_rejected() -> None:
    gt = [_image("img_1"), _image("img_1")]

    with pytest.raises(ValueError, match="duplicate"):
        confusion_matrix(gt, {})


@pytest.mark.parametrize("iou", [-0.1, 0.0, 1.1, float("nan")])
def test_rejects_invalid_iou(iou: float) -> None:
    with pytest.raises(ValueError, match="iou"):
        confusion_matrix([], {}, iou=iou)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_score_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        confusion_matrix([], {}, score_threshold=threshold)
