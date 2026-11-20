"""Ports implemented by driven adapters (hexagonal architecture, decision D9).

Ports are ``typing.Protocol`` classes: the application layer depends on these
shapes, never on a concrete adapter. The first ports introduced here are the
dataset repository and the progress observer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rdv.application.events import ProgressEvent
from rdv.domain.annotation import AnnotatedImage
from rdv.domain.detection import Detection

RgbImage = NDArray[np.uint8]
"""HxWx3 ``uint8`` array in RGB order (kept as a portable alias)."""

DEFAULT_CLASS_CODES: tuple[str, ...] = ("D00", "D10", "D20", "D40")
"""Class codes emitted by every detector, in model-id order (decision D4)."""

DEFAULT_SCORE_THRESHOLD = 0.25
"""Fallback score cutoff when no model manifest recommends a value."""


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Aggregate counts produced while scanning a dataset's annotations.

    Attributes:
        images: image count per split.
        countries: image count per country.
        classes: ground-truth box count per damage-class code.
        dropped_labels: raw label text -> occurrences dropped as unsupported.
    """

    images: Mapping[str, int]
    countries: Mapping[str, int]
    classes: Mapping[str, int]
    dropped_labels: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation for ``rdv data stats --json``."""
        return {
            "images": dict(self.images),
            "countries": dict(self.countries),
            "classes": dict(self.classes),
            "dropped_labels": dict(self.dropped_labels),
        }


class DatasetRepository(Protocol):
    """Iterate labeled images by split without eagerly reading pixels.

    Iteration yields metadata-only :class:`AnnotatedImage` values (XML parse
    plus an existence check). Pixel decoding happens only in
    :meth:`load_image`, so scanning a split stays cheap and testable.
    """

    def __iter__(self) -> Iterator[AnnotatedImage]:
        """Yield annotations in deterministic order."""
        ...

    def __len__(self) -> int:
        """Number of images selected by the repository configuration."""
        ...

    def load_image(self, image_id: str) -> RgbImage:
        """Decode one image lazily; the id must have been yielded before."""
        ...

    def stats(self) -> DatasetStats:
        """Scan annotations and return aggregate counts (cached)."""
        ...


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Identity and inference parameters of a loaded detector.

    Attributes:
        id: manifest id or file stem of a local model.
        input_size: square network input side in pixels.
        score_threshold: score cutoff applied by the detector.
        device: execution device actually selected (``cpu`` or ``cuda``).
        sha256: digest of the model file when available.
        classes: class codes in model-id order.
        path: local path of the model file, for diagnostics.
        version: manifest version, when the model came from the manifest.
        license: license identifier recorded in the manifest.
    """

    id: str
    input_size: int
    score_threshold: float
    device: str
    sha256: str | None = None
    classes: tuple[str, ...] = DEFAULT_CLASS_CODES
    path: str | None = None
    version: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ModelInfo id must not be empty")
        if self.input_size <= 0:
            raise ValueError("ModelInfo input_size must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("ModelInfo score_threshold must be within [0, 1]")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported ModelInfo device: {self.device!r}")
        if tuple(self.classes) != DEFAULT_CLASS_CODES:
            raise ValueError(f"ModelInfo classes must be {DEFAULT_CLASS_CODES!r}")
        object.__setattr__(self, "classes", tuple(self.classes))


class Detector(Protocol):
    """Run object detection on RGB images (implemented by ONNX/torch adapters)."""

    def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
        """Return detections per input image, in the same order."""
        ...

    @property
    def info(self) -> ModelInfo:
        """Identity, threshold and device of the loaded model."""
        ...


class ProgressSink(Protocol):
    """Observer of :class:`ProgressEvent` values."""

    def emit(self, event: ProgressEvent) -> None:
        """Handle one progress event."""
        ...
