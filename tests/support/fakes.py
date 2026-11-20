"""Deterministic fakes shared by evaluation tests.

``GroundTruthEchoDetector`` returns each image's ground truth as a detection
(optionally with seeded pixel jitter), which lets the evaluation use case be
tested without a real model while still exercising the metrics engine.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rdv.application.ports import (
    DEFAULT_CLASS_CODES,
    DatasetRepository,
    DatasetStats,
    ModelInfo,
    RgbImage,
)
from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox


def image_digest(image: RgbImage) -> str:
    """Stable content digest used to identify an image inside a fake detector."""
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Ground truth and shape known for one image digest."""

    image_id: str
    width: int
    height: int
    annotations: tuple[Annotation, ...]


class GroundTruthEchoDetector:
    """Emit the ground-truth boxes as high-score detections.

    Args:
        ground_truth: labeled images that the detector will see.
        image_loader: decodes each image id so its content digest can be
            registered; usually ``repository.load_image``.
        jitter: maximum absolute pixel shift per box corner.
        seed: seed for the deterministic jitter.
        score: confidence assigned to every echoed box.
    """

    def __init__(
        self,
        ground_truth: Sequence[AnnotatedImage],
        *,
        image_loader: Callable[[str], RgbImage],
        jitter: float = 0.0,
        seed: int = 42,
        score: float = 0.95,
        model_id: str = "ground-truth-echo",
    ) -> None:
        if jitter < 0.0:
            raise ValueError("jitter must be non-negative")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be within [0, 1]")
        self._jitter = float(jitter)
        self._seed = seed
        self._score = float(score)
        self._records: dict[str, ImageRecord] = {}
        for image in ground_truth:
            digest = image_digest(image_loader(image.image_id))
            self._records[digest] = ImageRecord(
                image_id=image.image_id,
                width=image.width,
                height=image.height,
                annotations=image.annotations,
            )
        self._info = ModelInfo(
            id=model_id,
            input_size=640,
            score_threshold=0.0,
            device="cpu",
            classes=DEFAULT_CLASS_CODES,
        )

    @classmethod
    def from_repository(
        cls,
        repository: DatasetRepository,
        *,
        jitter: float = 0.0,
        seed: int = 42,
        score: float = 0.95,
        model_id: str = "ground-truth-echo",
    ) -> GroundTruthEchoDetector:
        """Build a detector bound to a repository's lazy image loader."""
        return cls(
            list(repository),
            image_loader=repository.load_image,
            jitter=jitter,
            seed=seed,
            score=score,
            model_id=model_id,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        for image in images:
            record = self._records.get(image_digest(image))
            if record is None:
                results.append([])
                continue
            results.append(self._detections(record))
        return results

    def _detections(self, record: ImageRecord) -> list[Detection]:
        detections: list[Detection] = []
        for index, annotation in enumerate(record.annotations):
            box = annotation.bbox
            if self._jitter:
                rng = random.Random(f"{self._seed}:{record.image_id}:{index}")
                box = BBox(
                    box.x1 + rng.uniform(-self._jitter, self._jitter),
                    box.y1 + rng.uniform(-self._jitter, self._jitter),
                    box.x2 + rng.uniform(-self._jitter, self._jitter),
                    box.y2 + rng.uniform(-self._jitter, self._jitter),
                )
            clipped = box.clipped(record.width, record.height)
            if clipped.is_degenerate:
                continue
            detections.append(
                Detection(
                    damage_class=annotation.damage_class,
                    score=self._score,
                    bbox=clipped,
                )
            )
        return detections


class EmptyDetector:
    """Detector that never finds anything (mAP 0 baseline)."""

    def __init__(self) -> None:
        self._info = ModelInfo(
            id="empty",
            input_size=640,
            score_threshold=0.0,
            device="cpu",
            classes=DEFAULT_CLASS_CODES,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
        return [[] for _ in images]


class InMemoryDatasetRepository:
    """Minimal ``DatasetRepository`` over in-memory images and annotations."""

    def __init__(
        self,
        images: Sequence[AnnotatedImage],
        pixels: Mapping[str, RgbImage],
        *,
        name: str = "memory",
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self._images = tuple(images)
        self._pixels = dict(pixels)
        self._name = name
        self._on_event = on_event
        missing = {image.image_id for image in self._images} - set(self._pixels)
        if missing:
            raise ValueError(f"Missing pixels for: {sorted(missing)!r}")

    def __iter__(self):
        self._event("iter")
        return iter(self._images)

    def __len__(self) -> int:
        return len(self._images)

    def load_image(self, image_id: str) -> RgbImage:
        self._event(f"load:{image_id}")
        return self._pixels[image_id]

    def stats(self) -> DatasetStats:
        images: dict[str, int] = {}
        countries: dict[str, int] = {}
        classes = dict.fromkeys((code for code in DEFAULT_CLASS_CODES), 0)
        for image in self._images:
            countries[image.image_id.split("/", 1)[0]] = (
                countries.get(image.image_id.split("/", 1)[0], 0) + 1
            )
            for annotation in image.annotations:
                classes[annotation.damage_class.code] += 1
        images["memory"] = len(self._images)
        return DatasetStats(
            images=images,
            countries=countries,
            classes=classes,
            dropped_labels={},
        )

    def _event(self, value: str) -> None:
        if self._on_event is not None:
            self._on_event(self._name, value)


def solid_image(width: int, height: int, value: int) -> RgbImage:
    """Small unique image so fake detectors can identify it by digest."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def annotation(damage_class: DamageClass, box: tuple[float, float, float, float]) -> Annotation:
    """Shorthand annotation builder for tests."""
    return Annotation(damage_class=damage_class, bbox=BBox(*box))
