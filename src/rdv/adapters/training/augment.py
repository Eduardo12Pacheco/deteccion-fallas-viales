"""torchvision v2 augmentation pipeline (decision D16).

Samples flow through transforms as ``{"image", "boxes", "labels"}`` dictionaries
so torchvision keeps geometry and labels in sync. ``SanitizeBoundingBoxes`` is
always last: it clips boxes to the canvas, drops degenerate ones and removes
the matching labels, which is the invariant the trainer relies on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors

from rdv.adapters.training.config import AugmentConfig
from rdv.domain.annotation import Annotation


def make_sample(
    image: np.ndarray,
    annotations: Sequence[Annotation],
) -> dict[str, Any]:
    """Build a torchvision v2 sample from an RGB array and annotations.

    Raises:
        ValueError: the array is not an ``HxWx3`` ``uint8`` image.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {array.shape}")
    height, width = int(array.shape[0]), int(array.shape[1])
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float() / 255.0
    if annotations:
        boxes = torch.tensor(
            [
                [annotation.bbox.x1, annotation.bbox.y1, annotation.bbox.x2, annotation.bbox.y2]
                for annotation in annotations
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [int(annotation.damage_class) for annotation in annotations],
            dtype=torch.int64,
        )
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    return {
        "image": tensor,
        "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
        "labels": labels,
    }


def build_train_transform(config: AugmentConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the configured stochastic augmentation chain."""
    operations: list[Callable[[dict[str, Any]], dict[str, Any]]] = [
        transforms.Resize(config.resize, antialias=True)
    ]
    if config.hflip > 0.0:
        operations.append(transforms.RandomHorizontalFlip(p=config.hflip))
    if config.vflip:
        operations.append(transforms.RandomVerticalFlip(p=0.5))
    if config.rot90:
        operations.append(transforms.RandomRotation((90.0, 90.0)))
    if config.photometric_distort > 0.0:
        operations.append(transforms.RandomPhotometricDistort(p=config.photometric_distort))
    if config.affine.p > 0.0:
        operations.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=0.0,
                        translate=(config.affine.translate, config.affine.translate),
                        scale=config.affine.scale,
                    )
                ],
                p=config.affine.p,
            )
        )
    if config.zoom_out:
        operations.append(transforms.RandomZoomOut(p=0.5))
    operations.append(transforms.SanitizeBoundingBoxes(min_size=2.0))
    return cast(Callable[[dict[str, Any]], dict[str, Any]], transforms.Compose(operations))


def build_eval_transform(config: AugmentConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Deterministic resize used for validation and inference."""
    return cast(
        Callable[[dict[str, Any]], dict[str, Any]],
        transforms.Compose([transforms.Resize(config.resize, antialias=True)]),
    )
