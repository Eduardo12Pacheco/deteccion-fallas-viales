"""Detect damage in a video and write the S6 artifact set.

The use case composes the frame pipeline (stride -> detect -> track), renders
every source frame through the ``Renderer`` port and hands the stream to an
atomic ``VideoSink``. Documents are plain JSON-ready dictionaries shared with
the CLI and, later, the web API.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdv import __version__
from rdv.application.detect_image import TOOL_NAME, detection_payload, round_value
from rdv.application.pipeline import (
    DetectStage,
    FramePacket,
    Pipeline,
    StrideSelector,
    TrackStage,
    analysis_stride,
    frame_packets,
)
from rdv.application.ports import (
    DetectionsWriter,
    Detector,
    ModelInfo,
    ProgressSink,
    Renderer,
    RgbImage,
    SummaryWriter,
    VideoSink,
    VideoSource,
)
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.summary import DamageSummary
from rdv.domain.tracking import IouTracker
from rdv.errors import MediaError

SCHEMA = "rdv.detections/v1"
SUMMARY_SCHEMA = "rdv.summary/v1"
DEFAULT_ANALYSIS_FPS = 5.0
_CLASS_ORDER = tuple(DamageClass)


@dataclass(frozen=True, slots=True)
class VideoOutcome:
    """Paths and aggregate statistics of one processed video."""

    source: Path
    width: int
    height: int
    fps: float
    stride: int
    analysis_fps: float
    max_frames: int | None
    frames_processed: int
    frames_analyzed: int
    frames_with_damage: int
    detections_total: int
    unique_instances: Mapping[DamageClass, int]
    coverage_ratio: float
    max_score: float
    truncated: bool
    annotated: Path
    detections_json: Path
    summary_json: Path
    summary_markdown: Path


def _counts(
    records: Sequence[tuple[int, tuple[Detection, ...], tuple[int, ...]]],
) -> dict[str, int]:
    counts = dict.fromkeys((damage_class.code for damage_class in _CLASS_ORDER), 0)
    for _, detections, _ in records:
        for detection in detections:
            counts[detection.damage_class.code] += 1
    return counts


def _unique_payload(unique: Mapping[DamageClass, int]) -> dict[str, int]:
    return {damage_class.code: int(unique.get(damage_class, 0)) for damage_class in _CLASS_ORDER}


def build_video_detections_document(
    *,
    source: Path,
    width: int,
    height: int,
    fps: float,
    frames: Sequence[tuple[int, float, Sequence[Detection], Sequence[int]]],
    model: ModelInfo,
    score_threshold: float,
    analysis_fps: float,
    max_frames: int | None,
    frames_processed: int,
    unique_instances: Mapping[DamageClass, int],
    coverage_ratio: float,
) -> dict[str, Any]:
    """Build the ``rdv.detections/v1`` document for one video (S6)."""
    records = [(index, tuple(detections), tuple(ids)) for index, _, detections, ids in frames]
    counts = _counts(records)
    scores = [detection.score for _, detections, _ in records for detection in detections]
    frames_with_damage = sum(1 for _, detections, _ in records if detections)
    return {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": __version__},
        "model": {
            "id": model.id,
            "sha256": model.sha256,
            "input_size": model.input_size,
        },
        "source": {
            "path": str(source),
            "kind": "video",
            "width": width,
            "height": height,
            "fps": round_value(fps),
            "frames": frames_processed,
        },
        "params": {
            "score_threshold": round_value(score_threshold),
            "device": model.device,
            "analysis_fps": round_value(analysis_fps),
            "max_frames": max_frames,
        },
        "frames": [
            {
                "index": index,
                "t": round_value(timestamp),
                "detections": [
                    {**detection_payload(detection), "track_id": track_id}
                    for detection, track_id in zip(detections, ids, strict=True)
                ],
            }
            for index, timestamp, detections, ids in frames
        ],
        "summary": {
            "counts": counts,
            "unique_instances": _unique_payload(unique_instances),
            "frames_analyzed": len(frames),
            "frames_with_damage": frames_with_damage,
            "coverage_ratio": round_value(coverage_ratio),
            "max_score": round_value(max(scores, default=0.0)),
        },
    }


def build_video_summary_document(outcome: VideoOutcome, *, model: ModelInfo) -> dict[str, Any]:
    """Build the aggregate ``rdv.summary/v1`` view-model for one video."""
    return {
        "schema": SUMMARY_SCHEMA,
        "tool": {"name": TOOL_NAME, "version": __version__},
        "model": {
            "id": model.id,
            "sha256": model.sha256,
            "input_size": model.input_size,
        },
        "params": {
            "score_threshold": round_value(model.score_threshold),
            "device": model.device,
            "analysis_fps": round_value(outcome.analysis_fps),
            "max_frames": outcome.max_frames,
        },
        "totals": {
            "videos": 1,
            "frames_processed": outcome.frames_processed,
            "frames_analyzed": outcome.frames_analyzed,
            "frames_with_damage": outcome.frames_with_damage,
            "detections": outcome.detections_total,
            "max_score": round_value(outcome.max_score),
            "mean_coverage_ratio": round_value(outcome.coverage_ratio),
            "truncated": outcome.truncated,
        },
        "classes": {
            damage_class.code: {
                "name": damage_class.display_name,
                "count": 0,
                "unique_instances": int(outcome.unique_instances.get(damage_class, 0)),
            }
            for damage_class in _CLASS_ORDER
        },
        "videos": [
            {
                "path": str(outcome.source),
                "width": outcome.width,
                "height": outcome.height,
                "fps": round_value(outcome.fps),
                "frames_processed": outcome.frames_processed,
                "frames_analyzed": outcome.frames_analyzed,
                "frames_with_damage": outcome.frames_with_damage,
                "detections": outcome.detections_total,
                "max_score": round_value(outcome.max_score),
                "coverage_ratio": round_value(outcome.coverage_ratio),
                "unique_instances": _unique_payload(outcome.unique_instances),
                "annotated": outcome.annotated.name,
                "detections_json": outcome.detections_json.name,
                "truncated": outcome.truncated,
            }
        ],
    }


def detect_video(
    source: VideoSource,
    *,
    detector: Detector,
    renderer: Renderer,
    sink: VideoSink,
    detections_writer: DetectionsWriter,
    summary_writer: SummaryWriter,
    output_dir: Path,
    analysis_fps: float = DEFAULT_ANALYSIS_FPS,
    max_frames: int | None = None,
    progress: ProgressSink | None = None,
) -> VideoOutcome:
    """Analyze one video and persist the S6 artifacts.

    Args:
        source: open video decoded frame by frame.
        detector: inference port.
        renderer: overlay renderer for the annotated MP4.
        sink: atomic MP4 encoder.
        detections_writer: per-video JSON writer.
        summary_writer: aggregate JSON/Markdown writer.
        output_dir: directory receiving every artifact (created if needed).
        analysis_fps: analyzed frames per second (D20 stride).
        max_frames: stop after this many source frames.
        progress: optional observer of per-analyzed-frame progress.

    Returns:
        Paths and aggregate statistics of the run.

    Raises:
        MediaError: the video is invalid, ends before its declared frame count,
            contains no frames, or an artifact cannot be written.
    """
    if not math.isfinite(analysis_fps) or analysis_fps <= 0.0:
        raise MediaError(f"analysis_fps must be positive and finite, got {analysis_fps!r}")
    if max_frames is not None and max_frames < 1:
        raise MediaError(f"max_frames must be positive, got {max_frames!r}")
    if source.frame_count is not None and source.frame_count < 1:
        raise MediaError(f"Video contains no frames: {source.path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stride = analysis_stride(source.fps, analysis_fps)
    declared = source.frame_count
    expected_frames = (
        min(declared, max_frames)
        if declared is not None and max_frames is not None
        else max_frames
        if max_frames is not None
        else declared
    )
    expected_analyzed = math.ceil(expected_frames / stride) if expected_frames is not None else None

    tracker = IouTracker()
    stages: list[Any] = [
        StrideSelector(fps=source.fps, analysis_fps=analysis_fps),
        DetectStage(detector, progress=progress, total=expected_analyzed),
        TrackStage(tracker),
    ]
    pipeline: Pipeline[FramePacket] = Pipeline(stages)

    records: list[tuple[int, float, tuple[Detection, ...], tuple[int, ...]]] = []
    processed = 0
    frames_with_damage = 0
    coverage_total = 0.0
    max_score = 0.0

    def rendered() -> Iterator[RgbImage]:
        nonlocal processed, frames_with_damage, coverage_total, max_score
        for packet in pipeline.run(
            frame_packets(
                source.frames(),
                fps=source.fps,
                max_frames=max_frames,
                expected_frames=expected_frames,
            )
        ):
            processed += 1
            if packet.analyzed:
                summary = DamageSummary.from_detections(
                    packet.detections, (source.width, source.height)
                )
                frames_with_damage += 1 if packet.detections else 0
                coverage_total += summary.coverage_ratio
                max_score = max(max_score, summary.max_score)
                records.append(
                    (packet.index, packet.timestamp, packet.detections, packet.track_ids)
                )
            yield renderer.render(packet.image, packet.detections)

    stem = Path(source.path).stem
    annotated = output_dir / f"{stem}.annotated.mp4"
    detections_json = output_dir / f"{stem}.detections.json"
    summary_json = output_dir / f"{stem}.summary.json"
    summary_markdown = output_dir / f"{stem}.summary.md"
    try:
        sink.write_video(annotated, rendered(), fps=source.fps)
    finally:
        source.close()

    if not records:
        raise MediaError(f"Video produced no analyzed frames: {source.path}")
    if declared is not None and processed < declared and max_frames is None:
        raise MediaError(f"Video ended after {processed} of {declared} promised frames")
    frames_analyzed = len(records)
    coverage_ratio = coverage_total / frames_analyzed
    unique = tracker.unique_counts()
    truncated = max_frames is not None and (declared is None or processed < declared)

    document = build_video_detections_document(
        source=source.path,
        width=source.width,
        height=source.height,
        fps=source.fps,
        frames=records,
        model=detector.info,
        score_threshold=detector.info.score_threshold,
        analysis_fps=analysis_fps,
        max_frames=max_frames,
        frames_processed=processed,
        unique_instances=unique,
        coverage_ratio=coverage_ratio,
    )
    detections_writer.write_detections(detections_json, document)

    detections_total = sum(len(detections) for _, _, detections, _ in records)
    outcome = VideoOutcome(
        source=source.path,
        width=source.width,
        height=source.height,
        fps=source.fps,
        stride=stride,
        analysis_fps=analysis_fps,
        max_frames=max_frames,
        frames_processed=processed,
        frames_analyzed=frames_analyzed,
        frames_with_damage=frames_with_damage,
        detections_total=detections_total,
        unique_instances=unique,
        coverage_ratio=coverage_ratio,
        max_score=max_score,
        truncated=truncated,
        annotated=annotated,
        detections_json=detections_json,
        summary_json=summary_json,
        summary_markdown=summary_markdown,
    )
    summary = build_video_summary_document(outcome, model=detector.info)
    for code, payload in summary["classes"].items():
        payload["count"] = document["summary"]["counts"][code]
    summary_writer.write_summary_json(summary_json, summary)
    summary_writer.write_summary_markdown(summary_markdown, summary)
    return outcome
