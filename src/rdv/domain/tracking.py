"""Greedy IoU tracking and unique-instance counts (decision D20).

Tracking runs on analyzed frames only. Detections are matched per class by
descending IoU; unmatched detections start new tracks and unmatched tracks age
out after ``max_age`` analyzed frames. A "unique instance" is a track with at
least ``min_hits`` matched frames. Without ego-motion compensation the counts
are approximate, which the documentation states.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox, iou_matrix


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection linked to its track and the track's current hit count."""

    track_id: int
    detection: Detection
    hits: int

    def __post_init__(self) -> None:
        if self.track_id < 1:
            raise ValueError("track_id must be positive")
        if self.hits < 1:
            raise ValueError("hits must be positive")


@dataclass(slots=True)
class _Track:
    track_id: int
    damage_class: DamageClass
    bbox: BBox
    score: float
    hits: int
    age: int = 0


class IouTracker:
    """Greedy IoU tracker with per-class matching and track retirement."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 2,
    ) -> None:
        if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be within [0, 1], got {iou_threshold!r}")
        if max_age < 0:
            raise ValueError(f"max_age must be non-negative, got {max_age!r}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be positive, got {min_hits!r}")
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._min_hits = min_hits
        self._next_track_id = 1
        self._tracks: list[_Track] = []
        self._retired_counts: dict[DamageClass, int] = dict.fromkeys(DamageClass, 0)

    def update(self, frame_detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Match one analyzed frame and return detections in input order."""
        detections = tuple(frame_detections)
        tracked: list[TrackedDetection | None] = [None] * len(detections)
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self._tracks)))

        for damage_class in DamageClass:
            detection_indices = [
                index
                for index, detection in enumerate(detections)
                if detection.damage_class is damage_class
            ]
            track_indices = [
                index
                for index, track in enumerate(self._tracks)
                if track.damage_class is damage_class and index in unmatched_tracks
            ]
            if not detection_indices or not track_indices:
                continue

            overlaps = iou_matrix(
                [detections[index].bbox for index in detection_indices],
                [self._tracks[index].bbox for index in track_indices],
            )
            candidates = [
                (float(overlaps[row, column]), detection_index, track_index)
                for row, detection_index in enumerate(detection_indices)
                for column, track_index in enumerate(track_indices)
                if overlaps[row, column] >= self._iou_threshold
            ]
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, detection_index, track_index in candidates:
                if (
                    detection_index not in unmatched_detections
                    or track_index not in unmatched_tracks
                ):
                    continue
                track = self._tracks[track_index]
                detection = detections[detection_index]
                track.bbox = detection.bbox
                track.score = detection.score
                track.hits += 1
                track.age = 0
                unmatched_detections.remove(detection_index)
                unmatched_tracks.remove(track_index)
                tracked[detection_index] = TrackedDetection(
                    track_id=track.track_id,
                    detection=detection,
                    hits=track.hits,
                )

        for detection_index in sorted(unmatched_detections):
            detection = detections[detection_index]
            track = _Track(
                track_id=self._next_track_id,
                damage_class=detection.damage_class,
                bbox=detection.bbox,
                score=detection.score,
                hits=1,
            )
            self._next_track_id += 1
            self._tracks.append(track)
            tracked[detection_index] = TrackedDetection(
                track_id=track.track_id,
                detection=detection,
                hits=track.hits,
            )

        surviving: list[_Track] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in unmatched_tracks:
                track.age += 1
            if track.age > self._max_age:
                if track.hits >= self._min_hits:
                    self._retired_counts[track.damage_class] += 1
            else:
                surviving.append(track)
        self._tracks = surviving

        return [item for item in tracked if item is not None]

    def unique_counts(self) -> dict[DamageClass, int]:
        """Count unique instances per class, including retired tracks."""
        counts = dict(self._retired_counts)
        for track in self._tracks:
            if track.hits >= self._min_hits:
                counts[track.damage_class] += 1
        return counts
