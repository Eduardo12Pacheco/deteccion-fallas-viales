"""Composable frame pipeline used by video detection (S6, decision D20).

Stages are plain callables that transform an iterable lazily, so a 40-frame
clip and a 40-minute clip share the same memory profile. The stages mirror the
plan's ``FrameSource -> StrideSelector -> DetectStage -> TrackStage -> Sink``
chain; the sink is a consumer so it can own atomic output files.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rdv.application.events import ProgressEvent
from rdv.application.ports import Detector, ProgressSink, RgbImage
from rdv.domain.detection import Detection
from rdv.domain.tracking import IouTracker
from rdv.errors import MediaError


class Stage[T](Protocol):
    """One lazy transformation over a stream of items."""

    def __call__(self, items: Iterable[T]) -> Iterator[T]:
        """Transform ``items`` lazily."""
        ...


class Pipeline[T]:
    """Chain of :class:`Stage` instances applied left to right."""

    def __init__(self, stages: Iterable[Stage[T]]) -> None:
        self._stages = tuple(stages)

    def run(self, items: Iterable[T]) -> Iterator[T]:
        """Return the lazily transformed stream."""
        stream: Iterable[T] = items
        for stage in self._stages:
            stream = stage(stream)
        return iter(stream)


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One source frame plus everything the downstream stages derived.

    Attributes:
        index: zero-based source frame index.
        timestamp: seconds from the start, ``index / fps``.
        image: decoded RGB frame.
        analyzed: whether this frame runs the detector and the tracker.
        detections: current detections (latest analyzed ones on skipped frames).
        track_ids: tracker ids aligned one-to-one with ``detections``.
    """

    index: int
    timestamp: float
    image: RgbImage
    analyzed: bool = True
    detections: tuple[Detection, ...] = ()
    track_ids: tuple[int, ...] = ()


def analysis_stride(fps: float, analysis_fps: float) -> int:
    """Source frames per analyzed frame, ``max(1, round(fps / analysis_fps))``.

    Raises:
        ValueError: either rate is not a positive finite number.
    """
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps!r}")
    if not math.isfinite(analysis_fps) or analysis_fps <= 0.0:
        raise ValueError(f"analysis_fps must be positive and finite, got {analysis_fps!r}")
    return max(1, round(fps / analysis_fps))


def frame_packets(
    frames: Iterable[RgbImage],
    *,
    fps: float,
    max_frames: int | None = None,
    expected_frames: int | None = None,
) -> Iterator[FramePacket]:
    """Wrap decoded frames as packets and close the stream when stopped early.

    Args:
        frames: lazily decoded frames.
        fps: source frame rate used for timestamps.
        max_frames: stop after this many source frames.
        expected_frames: frames the container promised; a short stream raises
            :class:`MediaError` so a truncated file never looks complete.

    Yields:
        One :class:`FramePacket` per decoded source frame.
    """
    iterator = iter(frames)
    processed = 0
    try:
        while max_frames is None or processed < max_frames:
            try:
                image = next(iterator)
            except StopIteration:
                break
            yield FramePacket(
                index=processed,
                timestamp=processed / fps,
                image=np.asarray(image),
            )
            processed += 1
        if expected_frames is not None and processed < expected_frames:
            raise MediaError(f"Video ended after {processed} of {expected_frames} promised frames")
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


class StrideSelector:
    """Mark every ``stride``-th frame for detection (D20 analysis rate)."""

    def __init__(self, *, fps: float, analysis_fps: float) -> None:
        self.stride = analysis_stride(fps, analysis_fps)

    def __call__(self, frames: Iterable[FramePacket]) -> Iterator[FramePacket]:
        """Set ``analyzed`` on the packets that survive the stride."""
        for packet in frames:
            yield replace(packet, analyzed=packet.index % self.stride == 0)


class DetectStage:
    """Run the detector on analyzed frames and emit progress events."""

    def __init__(
        self,
        detector: Detector,
        *,
        progress: ProgressSink | None = None,
        total: int | None = None,
        stage: str = "video",
    ) -> None:
        self._detector = detector
        self._progress = progress
        self._total = total
        self._stage = stage

    def __call__(self, frames: Iterable[FramePacket]) -> Iterator[FramePacket]:
        """Attach detections to analyzed frames; pass skipped frames through."""
        analyzed = 0
        for packet in frames:
            if not packet.analyzed:
                yield packet
                continue
            detections = tuple(self._detector.detect([packet.image])[0])
            analyzed += 1
            if self._progress is not None:
                self._progress.emit(
                    ProgressEvent(
                        stage=self._stage,
                        current=analyzed,
                        total=self._total,
                        message=f"frame {packet.index}: {len(detections)} detection(s)",
                    )
                )
            yield replace(packet, detections=detections)


class TrackStage:
    """Track analyzed detections and repeat the latest ones on skipped frames."""

    def __init__(self, tracker: IouTracker | None = None) -> None:
        self.tracker = tracker if tracker is not None else IouTracker()

    def __call__(self, frames: Iterable[FramePacket]) -> Iterator[FramePacket]:
        """Attach stable track ids; skipped frames reuse the latest detections."""
        latest_detections: tuple[Detection, ...] = ()
        latest_ids: tuple[int, ...] = ()
        for packet in frames:
            if packet.analyzed:
                tracked = self.tracker.update(packet.detections)
                latest_detections = tuple(item.detection for item in tracked)
                latest_ids = tuple(item.track_id for item in tracked)
            yield replace(
                packet,
                detections=latest_detections,
                track_ids=latest_ids,
            )


def ensure_rgb(frame: NDArray[np.uint8]) -> RgbImage:
    """Validate a decoded frame and return it unchanged (defensive helper)."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise MediaError(f"Expected an HxWx3 uint8 RGB frame, got {array.shape}/{array.dtype}")
    return array
