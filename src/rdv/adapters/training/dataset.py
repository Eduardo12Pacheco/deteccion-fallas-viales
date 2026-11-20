"""Torch ``Dataset`` over the VOC repository with HF detection labels (D16).

Each sample is resized to the network canvas, augmented with the configured
torchvision v2 chain, and returned as ``{"pixel_values", "labels"}`` where
``labels`` uses the Hugging Face RT-DETR contract: integer ``class_labels`` and
``boxes`` normalized ``cxcywh`` in the resized image. Images without boxes are
kept as negatives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from rdv.adapters.training.augment import make_sample
from rdv.adapters.training.config import AugmentConfig
from rdv.application.ports import DatasetRepository, RgbImage
from rdv.domain.annotation import AnnotatedImage


@dataclass(frozen=True, slots=True)
class TorchSample:
    """One collated-ready training sample."""

    pixel_values: torch.Tensor
    boxes: torch.Tensor
    class_labels: torch.Tensor


def _normalized_cxcywh(boxes: tv_tensors.BoundingBoxes) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    height, width = boxes.canvas_size
    xyxy = boxes.as_subclass(torch.Tensor).to(torch.float32)
    x1, y1, x2, y2 = xyxy.unbind(-1)
    cx = (x1 + x2) / 2.0 / float(width)
    cy = (y1 + y2) / 2.0 / float(height)
    box_width = (x2 - x1) / float(width)
    box_height = (y2 - y1) / float(height)
    return torch.stack([cx, cy, box_width, box_height], dim=-1)


class VocTorchDataset(Dataset[TorchSample]):
    """Training/validation dataset over a ``DatasetRepository``.

    Args:
        annotations: metadata-only images selected for this split.
        load_image: lazy pixel decoder, usually ``repository.load_image``.
        augment: augmentation configuration (``resize`` defines the canvas).
        training: use the stochastic chain when true, deterministic resize
            otherwise.
        max_images: cap the number of images, useful for CPU smoke runs.
    """

    def __init__(
        self,
        annotations: Sequence[AnnotatedImage],
        load_image: Callable[[str], RgbImage],
        *,
        augment: AugmentConfig | None = None,
        training: bool = True,
        max_images: int | None = None,
    ) -> None:
        config = augment if augment is not None else AugmentConfig()
        if max_images is not None and max_images < 1:
            raise ValueError("max_images must be positive or None")
        selected = list(annotations)[:max_images] if max_images is not None else list(annotations)
        self._images = tuple(selected)
        self._load_image = load_image
        self._config = config
        self.training = training
        if training:
            from rdv.adapters.training.augment import build_train_transform

            self._transform = build_train_transform(config)
        else:
            from rdv.adapters.training.augment import build_eval_transform

            self._transform = build_eval_transform(config)

    @classmethod
    def from_repository(
        cls,
        repository: DatasetRepository,
        *,
        augment: AugmentConfig | None = None,
        training: bool = True,
        max_images: int | None = None,
    ) -> VocTorchDataset:
        """Build the dataset from a repository, preserving its deterministic order."""
        return cls(
            list(repository),
            repository.load_image,
            augment=augment,
            training=training,
            max_images=max_images,
        )

    def __len__(self) -> int:
        return len(self._images)

    @property
    def annotation(self) -> tuple[AnnotatedImage, ...]:
        """The selected labels in dataset order (used for validation metrics)."""
        return self._images

    def __getitem__(self, index: int) -> TorchSample:
        image = self._images[index]
        sample = make_sample(self._load_image(image.image_id), image.annotations)
        transformed = self._transform(sample)
        boxes = transformed["boxes"]
        labels = transformed["labels"]
        return TorchSample(
            pixel_values=transformed["image"].to(torch.float32),
            boxes=_normalized_cxcywh(boxes),
            class_labels=labels.to(torch.int64),
        )


def collate_samples(samples: Sequence[TorchSample]) -> dict[str, Any]:
    """Collate samples into the HF RT-DETR ``pixel_values``/``labels`` contract."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {
        "pixel_values": torch.stack([sample.pixel_values for sample in samples], dim=0),
        "labels": [
            {
                "class_labels": sample.class_labels,
                "boxes": sample.boxes,
            }
            for sample in samples
        ],
    }


def seed_worker(worker_id: int, *, seed: int) -> None:
    """Seed a DataLoader worker deterministically (NumPy + Python + torch)."""
    import random

    worker_seed = int(seed) + int(worker_id)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
