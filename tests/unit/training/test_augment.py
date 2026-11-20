"""Augmentation invariants: boxes stay inside and degenerate boxes disappear."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torchvision import tv_tensors

from rdv.adapters.training.augment import (
    build_eval_transform,
    build_train_transform,
    make_sample,
)
from rdv.adapters.training.config import AugmentConfig

pytestmark = pytest.mark.train


def _sample() -> dict[str, object]:
    image = np.random.default_rng(7).integers(0, 255, size=(80, 100, 3), dtype=np.uint8)
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[10.0, 10.0, 40.0, 30.0], [60.0, 50.0, 99.0, 79.0]]),
        format="XYXY",
        canvas_size=(80, 100),
    )
    return {
        "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
        "boxes": boxes,
        "labels": torch.tensor([0, 3], dtype=torch.int64),
    }


def test_train_transform_keeps_boxes_inside_for_many_seeds() -> None:
    transform = build_train_transform(AugmentConfig())
    height, width = 640, 640

    for seed in range(25):
        torch.manual_seed(seed)
        output = transform(_sample())
        boxes = output["boxes"]
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        if boxes.numel() == 0:
            continue
        assert float(boxes[:, 0].min()) >= 0.0
        assert float(boxes[:, 1].min()) >= 0.0
        assert float(boxes[:, 2].max()) <= width
        assert float(boxes[:, 3].max()) <= height
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        assert bool((widths >= 2.0).all())
        assert bool((heights >= 2.0).all())


def test_empty_boxes_survive_augmentation() -> None:
    transform = build_train_transform(AugmentConfig())
    sample = _sample()
    sample["boxes"] = tv_tensors.BoundingBoxes(
        torch.zeros((0, 4)), format="XYXY", canvas_size=(80, 100)
    )
    sample["labels"] = torch.zeros((0,), dtype=torch.int64)

    output = transform(sample)

    assert output["boxes"].shape == (0, 4)


def test_eval_transform_only_resizes_and_is_deterministic() -> None:
    transform = build_eval_transform(AugmentConfig(resize=(240, 320)))
    first = transform(_sample())
    second = transform(_sample())

    assert first["image"].shape == (3, 240, 320)
    assert torch.equal(first["image"], second["image"])
    boxes = first["boxes"]
    assert boxes.canvas_size == (240, 320)
    assert torch.allclose(boxes[0], torch.tensor([32.0, 30.0, 128.0, 90.0]))


def test_make_sample_normalizes_arrays_and_rejects_bad_images() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    sample = make_sample(image, [])

    assert sample["image"].shape == (3, 10, 20)
    assert sample["image"].dtype == torch.float32
    assert sample["boxes"].shape == (0, 4)
    assert sample["labels"].dtype == torch.int64

    with pytest.raises(ValueError, match="HxWx3"):
        make_sample(np.zeros((10, 20), dtype=np.uint8), [])
