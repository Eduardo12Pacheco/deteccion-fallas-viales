"""Ports implemented by driven adapters (hexagonal architecture, decision D9).

Ports are ``typing.Protocol`` classes: the application layer depends on these
shapes, never on a concrete adapter. The first ports introduced here are the
dataset repository and the progress observer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rdv.application.events import ProgressEvent
from rdv.domain.annotation import AnnotatedImage

RgbImage = NDArray[np.uint8]
"""HxWx3 ``uint8`` array in RGB order (kept as a portable alias)."""


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


class ProgressSink(Protocol):
    """Observer of :class:`ProgressEvent` values."""

    def emit(self, event: ProgressEvent) -> None:
        """Handle one progress event."""
        ...
