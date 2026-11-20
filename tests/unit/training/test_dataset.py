"""Torch dataset over the VOC repository: labels, negatives and collation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from rdv.adapters.training.augment import build_eval_transform, make_sample
from rdv.adapters.training.config import AugmentConfig
from rdv.adapters.training.dataset import (
    VocTorchDataset,
    collate_samples,
    seed_worker,
)
from rdv.application.ports import DatasetStats
from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.geometry import BBox

pytestmark = pytest.mark.train


def _annotations() -> list[AnnotatedImage]:
    return [
        AnnotatedImage(
            image_id="Japan/train/images/a.jpg",
            width=100,
            height=80,
            annotations=(
                Annotation(DamageClass.D00, BBox(10.0, 20.0, 30.0, 40.0)),
                Annotation(DamageClass.D40, BBox(60.0, 50.0, 90.0, 70.0)),
            ),
        ),
        AnnotatedImage(
            image_id="Japan/train/images/b.jpg",
            width=100,
            height=80,
            annotations=(),
        ),
    ]


def _loader(image_id: str) -> np.ndarray:
    return np.full((80, 100, 3), 128, dtype=np.uint8)


def test_samples_normalize_boxes_to_the_resized_canvas() -> None:
    dataset = VocTorchDataset(
        _annotations(),
        _loader,
        augment=AugmentConfig(resize=(640, 640)),
        training=False,
    )

    sample = dataset[0]

    assert sample.pixel_values.shape == (3, 640, 640)
    assert sample.pixel_values.dtype == torch.float32
    assert sample.pixel_values.max() <= 1.0
    assert sample.class_labels.tolist() == [0, 3]
    expected = torch.tensor(
        [
            [0.2, 0.375, 0.2, 0.25],
            [0.75, 0.75, 0.3, 0.25],
        ]
    )
    assert torch.allclose(sample.boxes, expected)


def test_negative_images_are_kept_with_empty_labels() -> None:
    dataset = VocTorchDataset(
        _annotations(),
        _loader,
        augment=AugmentConfig(resize=(320, 320)),
        training=False,
    )

    sample = dataset[1]

    assert sample.boxes.shape == (0, 4)
    assert sample.class_labels.shape == (0,)


def test_collate_builds_hf_labels_and_stacks_pixels() -> None:
    dataset = VocTorchDataset(
        _annotations(),
        _loader,
        augment=AugmentConfig(resize=(320, 320)),
        training=False,
    )
    samples = [dataset[0], dataset[1]]

    batch = collate_samples(samples)

    assert batch["pixel_values"].shape == (2, 3, 320, 320)
    assert len(batch["labels"]) == 2
    assert batch["labels"][1]["class_labels"].numel() == 0
    assert batch["labels"][0]["boxes"].shape == (2, 4)
    with pytest.raises(ValueError, match="empty batch"):
        collate_samples([])


def test_max_images_caps_the_dataset_deterministically() -> None:
    dataset = VocTorchDataset.from_repository(
        _StubRepository(_annotations()),
        augment=AugmentConfig(resize=(320, 320)),
        training=False,
        max_images=1,
    )

    assert len(dataset) == 1
    assert dataset.annotation[0].image_id.endswith("a.jpg")


def test_seed_worker_sets_reproducible_rngs() -> None:
    seed_worker(3, seed=42)
    first = (np.random.rand(), torch.rand(1).item())
    seed_worker(3, seed=42)
    second = (np.random.rand(), torch.rand(1).item())

    assert first == second


def test_eval_sample_and_transform_are_consistent() -> None:
    sample = make_sample(np.zeros((20, 10, 3), dtype=np.uint8), [])
    output = build_eval_transform(AugmentConfig(resize=(320, 320)))(sample)

    assert output["image"].shape == (3, 320, 320)
    assert output["boxes"].shape == (0, 4)


class _StubRepository:
    def __init__(self, images: list[AnnotatedImage]) -> None:
        self._images = images

    def __iter__(self):
        return iter(self._images)

    def __len__(self) -> int:
        return len(self._images)

    def load_image(self, image_id: str) -> np.ndarray:
        return _loader(image_id)

    def stats(self) -> DatasetStats:
        return DatasetStats(images={}, countries={}, classes={}, dropped_labels={})
