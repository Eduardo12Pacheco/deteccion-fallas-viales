"""Detections and per-frame results."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rdv.domain.classes import DamageClass
from rdv.domain.geometry import BBox


@dataclass(frozen=True, slots=True)
class Detection:
    """One predicted box with its class and calibrated score."""

    damage_class: DamageClass
    score: float
    bbox: BBox

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("Detection score must be finite")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection score must be within [0, 1]")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Detections for a single analyzed frame of a video."""

    frame_index: int
    detections: tuple[Detection, ...]
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.timestamp is not None:
            if not math.isfinite(self.timestamp):
                raise ValueError("timestamp must be finite")
            if self.timestamp < 0.0:
                raise ValueError("timestamp must be non-negative")
            object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "detections", tuple(self.detections))
