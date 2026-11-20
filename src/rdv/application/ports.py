"""Ports implemented by driven adapters (hexagonal architecture, decision D9).

Ports are ``typing.Protocol`` classes: the application layer depends on these
shapes, never on a concrete adapter. The first ports introduced here are the
dataset repository and the progress observer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model published in the packaged manifest (decision D13).

    Attributes:
        id: stable identifier used by ``rdv models pull`` and ``RDV_MODEL``.
        version: artifact version (``v1`` for the first fine-tune).
        url: pinned download URL of the single-file ONNX artifact.
        sha256: lowercase hex digest the downloaded file must match.
        size: exact byte size of the artifact.
        input_size: square network input side expected by the graph.
        score_threshold: recommended detection threshold for this model.
        classes: class codes in model-id order (always D4's four classes).
        license: license identifier of the weights (``Apache-2.0``).
        metrics_ref: relative path of the committed metrics, if any.
    """

    id: str
    version: str
    url: str
    sha256: str
    size: int
    input_size: int
    score_threshold: float
    classes: tuple[str, ...] = DEFAULT_CLASS_CODES
    license: str = "Apache-2.0"
    metrics_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ModelEntry id must not be empty")
        if not self.url.startswith(("https://", "http://")):
            raise ValueError(f"ModelEntry url must be HTTP(S), got {self.url!r}")
        if len(self.sha256) != 64:
            raise ValueError(f"ModelEntry sha256 must be 64 hex characters, got {self.sha256!r}")
        int(self.sha256, 16)
        if self.size <= 0:
            raise ValueError("ModelEntry size must be positive")
        if self.input_size <= 0:
            raise ValueError("ModelEntry input_size must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("ModelEntry score_threshold must be within [0, 1]")
        if tuple(self.classes) != DEFAULT_CLASS_CODES:
            raise ValueError(f"ModelEntry classes must be {DEFAULT_CLASS_CODES!r}")
        object.__setattr__(self, "classes", tuple(self.classes))


class ModelStore(Protocol):
    """Resolve, download, and verify model artifacts (decision D13)."""

    def entries(self) -> tuple[ModelEntry, ...]:
        """Models published in the manifest, in file order."""
        ...

    def resolve(self, model: str) -> Path:
        """Return a local path for a manifest id or an explicit ``.onnx`` path."""
        ...

    def pull(
        self,
        model_id: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download, verify and atomically install a manifest model."""
        ...

    def verify(self, path: Path, model_id: str | None = None) -> ModelEntry:
        """Check a file against its manifest entry and return that entry."""
        ...


class MediaSource(Protocol):
    """Decode input files into RGB arrays and encode rendered PNGs."""

    def load_image(self, path: Path) -> RgbImage:
        """Decode one image, applying EXIF orientation."""
        ...

    def save_png(self, path: Path, image: RgbImage) -> None:
        """Write an RGB array as a PNG file."""
        ...


class VideoSource(Protocol):
    """Decode a video file as an ordered stream of RGB frames (S6, D20).

    The source is single-use and must be closed by the caller (or by the
    generator's finalizer) so the underlying capture handle is released even
    when the pipeline stops early because of ``--max-frames``.
    """

    @property
    def path(self) -> Path:
        """Input file path, recorded as-is in the detections document."""
        ...

    @property
    def fps(self) -> float:
        """Source frame rate read from the container."""
        ...

    @property
    def frame_count(self) -> int | None:
        """Declared frame count, or ``None`` when the container omits it."""
        ...

    @property
    def width(self) -> int:
        """Display width in pixels after container rotation is applied."""
        ...

    @property
    def height(self) -> int:
        """Display height in pixels after container rotation is applied."""
        ...

    def frames(self) -> Iterator[RgbImage]:
        """Yield every decodable frame in display order (RGB ``uint8``)."""
        ...

    def close(self) -> None:
        """Release the capture handle; safe to call more than once."""
        ...


class VideoSink(Protocol):
    """Encode annotated RGB frames as an MP4, atomically per output path."""

    def write_video(
        self,
        path: Path,
        frames: Iterable[RgbImage],
        *,
        fps: float,
    ) -> None:
        """Consume ``frames`` into ``path``; a failed stream leaves no file."""
        ...


class Renderer(Protocol):
    """Draw detections over an input image without mutating it."""

    def render(self, image: RgbImage, detections: Sequence[Detection]) -> RgbImage:
        """Return a new RGB array with boxes and score labels."""
        ...


class DetectionsWriter(Protocol):
    """Persist one ``rdv.detections/v1`` document."""

    def write_detections(self, path: Path, document: Mapping[str, Any]) -> None:
        """Serialize ``document`` deterministically to ``path``."""
        ...


class SummaryWriter(Protocol):
    """Persist the aggregate run summary in JSON and Markdown."""

    def write_summary_json(self, path: Path, document: Mapping[str, Any]) -> None:
        """Serialize the aggregate summary as stable JSON."""
        ...

    def write_summary_markdown(self, path: Path, document: Mapping[str, Any]) -> None:
        """Render a human-readable Markdown summary."""
        ...


class ReportWriter(DetectionsWriter, SummaryWriter, Protocol):
    """Composite of both report writers (split into two adapters in practice).

    The use case takes the two halves explicitly so each adapter has one
    responsibility; this Protocol names the complete report-writing port.
    """


class ProgressSink(Protocol):
    """Observer of :class:`ProgressEvent` values."""

    def emit(self, event: ProgressEvent) -> None:
        """Handle one progress event."""
        ...
