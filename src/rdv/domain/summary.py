"""Damage coverage ratio and per-image summaries (decision D21).

Coverage is the exact union area of the detection boxes divided by the image
area. It is descriptive only: a large pothole-free crack can cover more area
than a dangerous pothole, so the number is never presented as severity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox, union_area


def damage_coverage(boxes: Sequence[BBox], image_size: tuple[int, int]) -> float:
    """Return the covered image fraction in ``[0, 1]``.

    Boxes are clipped to the image before measuring, so out-of-image
    coordinates cannot inflate the result above one.

    Raises:
        ValueError: the image dimensions are not positive.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_size!r}")
    return union_area(boxes, bounds=(width, height)) / (width * height)


@dataclass(frozen=True, slots=True)
class DamageSummary:
    """Per-class counts, the best score and the coverage of one image/frame."""

    counts: Mapping[DamageClass, int]
    max_score: float
    coverage_ratio: float = 0.0

    def __post_init__(self) -> None:
        unknown = set(self.counts) - set(DamageClass)
        if unknown:
            raise ValueError(f"counts contains an unknown class: {sorted(unknown)!r}")
        normalized: dict[DamageClass, int] = {}
        for damage_class in DamageClass:
            count = self.counts.get(damage_class, 0)
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"counts must be non-negative integers, got {count!r}")
            normalized[damage_class] = count
        if not math.isfinite(self.max_score) or not 0.0 <= self.max_score <= 1.0:
            raise ValueError(f"max_score must be within [0, 1], got {self.max_score!r}")
        if not math.isfinite(self.coverage_ratio) or not 0.0 <= self.coverage_ratio <= 1.0:
            raise ValueError(f"coverage_ratio must be within [0, 1], got {self.coverage_ratio!r}")
        object.__setattr__(self, "counts", MappingProxyType(normalized))
        object.__setattr__(self, "max_score", float(self.max_score))
        object.__setattr__(self, "coverage_ratio", float(self.coverage_ratio))

    @property
    def total(self) -> int:
        """Total number of detections."""
        return sum(self.counts.values())

    @classmethod
    def from_detections(
        cls, detections: Sequence[Detection], image_size: tuple[int, int]
    ) -> DamageSummary:
        """Summarize detections for an image of ``image_size`` pixels."""
        counts = dict.fromkeys(DamageClass, 0)
        for detection in detections:
            counts[detection.damage_class] += 1
        max_score = max((detection.score for detection in detections), default=0.0)
        return cls(
            counts=counts,
            max_score=max_score,
            coverage_ratio=damage_coverage(
                [detection.bbox for detection in detections], image_size
            ),
        )
