"""Ground-truth annotations and labeled images.

The repository adapter parses VOC XML into these value objects; the metrics
engine consumes them. Constructing an ``AnnotatedImage`` validates that every
ground-truth box lies inside its image, so a parser bug fails loudly here
instead of skewing an evaluation later.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdv.domain.classes import DamageClass
from rdv.domain.geometry import BBox


@dataclass(frozen=True, slots=True)
class Annotation:
    """One ground-truth box."""

    damage_class: DamageClass
    bbox: BBox


@dataclass(frozen=True, slots=True)
class AnnotatedImage:
    """A labeled image identified by a dataset-relative path."""

    image_id: str
    width: int
    height: int
    annotations: tuple[Annotation, ...] = ()

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        annotations = tuple(self.annotations)
        for annotation in annotations:
            box = annotation.bbox
            if box.x1 < 0.0 or box.y1 < 0.0 or box.x2 > self.width or box.y2 > self.height:
                raise ValueError(
                    f"Annotation box {box.to_xyxy()} is outside image "
                    f"{self.image_id!r} ({self.width}x{self.height})"
                )
        object.__setattr__(self, "annotations", annotations)

    @property
    def is_negative(self) -> bool:
        """True for labeled images that contain no damage."""
        return not self.annotations
