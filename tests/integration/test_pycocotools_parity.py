"""Parity of the numpy metrics engine with pycocotools COCOeval (D10).

The parity suite is the safety net for the decision to own the metrics engine:
if matching, sorting, interpolation or category handling drift from COCO
semantics, this test fails before any number reaches a report. It runs on
seeded synthetic datasets (500 boxes, four classes, noisy predictions) and on
a compact 16-image fixture shaped like ``tests/fixtures/rdd2022-mini``.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.metrics.confusion import confusion_matrix
from rdv.domain.metrics.evaluator import evaluate

pytestmark = pytest.mark.parity

TOLERANCE = 1e-4


@dataclass(frozen=True)
class Dataset:
    """Images and predictions in both representations."""

    images: list[AnnotatedImage]
    predictions: dict[str, list[Detection]]


def _make_synthetic_dataset(seed: int, *, num_boxes: int = 500) -> Dataset:
    rng = np.random.default_rng(seed)
    num_images = 40
    boxes_per_image: list[list[Annotation]] = [[] for _ in range(num_images)]
    for _ in range(num_boxes):
        image_index = int(rng.integers(0, num_images))
        x1 = float(rng.uniform(0, 560))
        y1 = float(rng.uniform(0, 560))
        width = float(rng.uniform(8, 40))
        height = float(rng.uniform(8, 40))
        boxes_per_image[image_index].append(
            Annotation(
                DamageClass(int(rng.integers(0, 4))),
                BBox(x1, y1, x1 + width, y1 + height),
            )
        )

    images: list[AnnotatedImage] = []
    predictions: dict[str, list[Detection]] = {}
    for image_index, image_annotations in enumerate(boxes_per_image):
        image_id = f"img_{image_index:04d}"
        images.append(
            AnnotatedImage(
                image_id=image_id,
                width=600,
                height=600,
                annotations=tuple(image_annotations),
            )
        )
        image_predictions: list[Detection] = []
        for annotation in image_annotations:
            if rng.random() < 0.65:
                box = annotation.bbox
                jitter = rng.normal(0.0, 3.0, size=4)
                left = max(0.0, box.x1 + jitter[0])
                top = max(0.0, box.y1 + jitter[1])
                right = min(600.0, box.x2 + abs(jitter[2]))
                bottom = min(600.0, box.y2 + abs(jitter[3]))
                if right <= left or bottom <= top:
                    continue
                predicted_box = BBox(left, top, right, bottom)
                damage_class = (
                    annotation.damage_class
                    if rng.random() < 0.85
                    else DamageClass(int(rng.integers(0, 4)))
                )
                image_predictions.append(
                    Detection(damage_class, float(rng.uniform(0.05, 0.99)), predicted_box)
                )
        for _ in range(int(rng.integers(0, 4))):
            x1 = float(rng.uniform(0, 560))
            y1 = float(rng.uniform(0, 560))
            x2 = x1 + float(rng.uniform(8, 40))
            y2 = y1 + float(rng.uniform(8, 40))
            image_predictions.append(
                Detection(
                    DamageClass(int(rng.integers(0, 4))),
                    float(rng.uniform(0.05, 0.99)),
                    BBox(x1, y1, x2, y2),
                )
            )
        predictions[image_id] = image_predictions

    return Dataset(images=images, predictions=predictions)


def _make_mini_dataset(*, noisy: bool, seed: int = 5) -> Dataset:
    rng = np.random.default_rng(seed)
    images: list[AnnotatedImage] = []
    predictions: dict[str, list[Detection]] = {}
    image_index = 0
    for class_index, damage_class in enumerate(DamageClass):
        for offset in range(3):
            image_id = f"mini_{image_index:04d}"
            x1 = 10.0 + offset * 60.0
            y1 = 10.0 + class_index * 40.0
            annotation = Annotation(damage_class, BBox(x1, y1, x1 + 50.0, y1 + 35.0))
            images.append(
                AnnotatedImage(
                    image_id=image_id,
                    width=200,
                    height=200,
                    annotations=(annotation,),
                )
            )
            if noisy:
                jitter = rng.normal(0.0, 4.0, size=4)
                left = max(0.0, x1 + jitter[0])
                top = max(0.0, y1 + jitter[1])
                right = min(200.0, x1 + 50.0 + abs(jitter[2]))
                bottom = min(200.0, y1 + 35.0 + abs(jitter[3]))
                if right <= left or bottom <= top:
                    continue
                predicted_box = BBox(left, top, right, bottom)
                predictions[image_id] = [
                    Detection(
                        damage_class,
                        float(rng.uniform(0.3, 0.99)),
                        predicted_box,
                    )
                ]
            else:
                predictions[image_id] = [Detection(damage_class, 0.95, annotation.bbox)]
            image_index += 1
    for _ in range(4):
        image_id = f"mini_{image_index:04d}"
        images.append(AnnotatedImage(image_id=image_id, width=200, height=200))
        predictions[image_id] = []
        image_index += 1
    return Dataset(images=images, predictions=predictions)


def _coco_evaluate(dataset: Dataset) -> COCOeval:
    category_ids = {damage_class: index + 1 for index, damage_class in enumerate(DamageClass)}
    image_ids = {
        image.image_id: index + 1
        for index, image in enumerate(sorted(dataset.images, key=lambda image: image.image_id))
    }
    images = [
        {
            "id": image_ids[image.image_id],
            "width": image.width,
            "height": image.height,
            "file_name": f"{image.image_id}.jpg",
        }
        for image in dataset.images
    ]
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image in dataset.images:
        for annotation in image.annotations:
            box = annotation.bbox
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_ids[image.image_id],
                    "category_id": category_ids[annotation.damage_class],
                    "bbox": [box.x1, box.y1, box.width, box.height],
                    "area": box.area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    results = [
        {
            "image_id": image_ids[image_id],
            "category_id": category_ids[detection.damage_class],
            "bbox": [
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.width,
                detection.bbox.height,
            ],
            "score": detection.score,
        }
        for image_id, detections in dataset.predictions.items()
        for detection in detections
    ]

    coco_ground_truth = COCO()
    coco_ground_truth.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": damage_class.code}
            for damage_class, category_id in category_ids.items()
        ],
    }
    coco_ground_truth.createIndex()
    coco_detections = coco_ground_truth.loadRes(results)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(coco_ground_truth, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval


def _assert_parity(dataset: Dataset) -> tuple[COCOeval, object]:
    coco_eval = _coco_evaluate(dataset)
    result = evaluate(dataset.images, dataset.predictions)

    assert abs(result.mAP50_95 - float(coco_eval.stats[0])) <= TOLERANCE
    assert abs(result.mAP50 - float(coco_eval.stats[1])) <= TOLERANCE

    for class_index, damage_class in enumerate(DamageClass):
        precision = coco_eval.eval["precision"][0, :, class_index, 0, 2]
        valid = precision > -1
        if not np.any(valid):
            assert result.per_class[damage_class].ap50 is None
            continue
        expected = float(np.mean(precision[valid]))
        assert result.per_class[damage_class].ap50 == pytest.approx(expected, abs=TOLERANCE), (
            damage_class
        )
    return coco_eval, result


@pytest.mark.parametrize("seed", [7, 21, 42])
def test_parity_on_seeded_synthetic_datasets(seed: int) -> None:
    _assert_parity(_make_synthetic_dataset(seed))


def test_parity_on_the_mini_fixture() -> None:
    dataset = _make_mini_dataset(noisy=True)

    _coco_eval, result = _assert_parity(dataset)

    # The jitter is small enough that every prediction still matches its box,
    # so the class-agnostic confusion matrix must stay the identity.
    matrix = confusion_matrix(dataset.images, dataset.predictions)
    expected = np.zeros((5, 5), dtype=np.int64)
    for index in range(len(DamageClass)):
        expected[index, index] = 3
    assert np.array_equal(matrix.matrix, expected)
    assert matrix.total == result.num_gt


def test_confusion_matrix_of_perfect_mini_predictions_is_the_identity() -> None:
    dataset = _make_mini_dataset(noisy=False)

    matrix = confusion_matrix(dataset.images, dataset.predictions)

    expected = np.zeros((5, 5), dtype=np.int64)
    for index in range(len(DamageClass)):
        expected[index, index] = 3
    assert np.array_equal(matrix.matrix, expected)
    assert matrix.total == 12
