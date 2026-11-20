"""Video detection use case over fake ports (S6, D20/D21)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rdv.application.detect_video import (
    SCHEMA,
    SUMMARY_SCHEMA,
    build_video_detections_document,
    detect_video,
)
from rdv.application.events import ProgressEvent
from rdv.application.ports import ModelInfo, RgbImage
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.errors import MediaError

INFO = ModelInfo(
    id="fake-model",
    input_size=640,
    score_threshold=0.25,
    device="cpu",
    sha256="b" * 64,
)


@dataclass
class FakeSource:
    frame_count: int | None
    fps: float = 10.0
    width: int = 32
    height: int = 24
    path: Path = Path("clip.mp4")
    drop_frames: int | None = None
    produced: int | None = None

    def frames(self) -> Iterator[RgbImage]:
        produced = self.drop_frames if self.drop_frames is not None else self.produced
        if produced is None:
            produced = self.frame_count or 0
        for _ in range(produced):
            yield np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class FixedDetector:
    def __init__(self, detections: Sequence[Detection]) -> None:
        self._detections = tuple(detections)
        self.calls = 0

    def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
        self.calls += len(images)
        return [list(self._detections) for _ in images]

    @property
    def info(self) -> ModelInfo:
        return INFO


class FakeRenderer:
    def render(self, image: RgbImage, detections: Sequence[Detection]) -> RgbImage:
        return image.copy()


class FakeSink:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.frames: list[RgbImage] = []
        self.fps: float | None = None

    def write_video(self, path: Path, frames, *, fps: float) -> None:  # type: ignore[no-untyped-def]
        self.path = path
        self.fps = fps
        self.frames = list(frames)


class RecordingWriter:
    def __init__(self) -> None:
        self.documents: list[tuple[Path, dict]] = []

    def write_detections(self, path: Path, document):  # type: ignore[no-untyped-def]
        self.documents.append((path, dict(document)))


class RecordingSummaryWriter(RecordingWriter):
    def write_summary_json(self, path: Path, document):  # type: ignore[no-untyped-def]
        self.documents.append((path, dict(document)))

    def write_summary_markdown(self, path: Path, document):  # type: ignore[no-untyped-def]
        self.documents.append((path, dict(document)))


class EventSink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _detection(code: DamageClass, x: float) -> Detection:
    return Detection(code, 0.9, BBox(x, 1.0, x + 4.0, 7.0))


def _run(tmp_path: Path, source: FakeSource, **kwargs: object):
    sink = FakeSink()
    detections = RecordingWriter()
    summaries = RecordingSummaryWriter()
    result = detect_video(
        source,  # type: ignore[arg-type]
        detector=FixedDetector([_detection(DamageClass.D00, 1.0)]),
        renderer=FakeRenderer(),
        sink=sink,
        detections_writer=detections,
        summary_writer=summaries,
        output_dir=tmp_path,
        **kwargs,  # type: ignore[arg-type]
    )
    return result, sink, detections, summaries


def test_detect_video_analyzes_every_stride_frame_and_repeats_detections(tmp_path: Path) -> None:
    source = FakeSource(frame_count=9)

    result, sink, documents, _ = _run(tmp_path, source, analysis_fps=5.0)

    assert result.frames_processed == 9
    assert result.frames_analyzed == 5
    assert len(sink.frames) == 9
    assert sink.fps == pytest.approx(10.0)
    document = documents.documents[0][1]
    assert document["schema"] == SCHEMA
    assert document["source"]["kind"] == "video"
    assert document["params"]["analysis_fps"] == 5.0
    assert [frame["index"] for frame in document["frames"]] == [0, 2, 4, 6, 8]
    assert all(frame["detections"][0]["track_id"] == 1 for frame in document["frames"])


def test_detect_video_summary_reports_counts_unique_instances_and_coverage(tmp_path: Path) -> None:
    source = FakeSource(frame_count=6)

    result, _, _, summaries = _run(tmp_path, source, analysis_fps=10.0)

    assert result.unique_instances == {
        DamageClass.D00: 1,
        DamageClass.D10: 0,
        DamageClass.D20: 0,
        DamageClass.D40: 0,
    }
    summary = summaries.documents[0][1]
    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["totals"]["videos"] == 1
    assert summary["totals"]["frames_analyzed"] == 6
    assert summary["totals"]["detections"] == 6
    assert summary["classes"]["D00"]["unique_instances"] == 1
    assert summary["videos"][0]["annotated"] == "clip.annotated.mp4"
    assert summary["videos"][0]["truncated"] is False


def test_max_frames_stops_early_and_is_recorded_in_the_summary(tmp_path: Path) -> None:
    source = FakeSource(frame_count=40)

    result, sink, documents, summaries = _run(tmp_path, source, analysis_fps=5.0, max_frames=10)

    assert result.frames_processed == 10
    assert result.frames_analyzed == 5
    assert result.truncated is True
    assert len(sink.frames) == 10
    assert documents.documents[0][1]["params"]["max_frames"] == 10
    assert summaries.documents[0][1]["totals"]["truncated"] is True
    assert summaries.documents[0][1]["videos"][0]["frames_processed"] == 10


def test_detect_video_requires_at_least_one_analyzed_frame(tmp_path: Path) -> None:
    source = FakeSource(frame_count=0)

    with pytest.raises(MediaError, match="no frames"):
        _run(tmp_path, source)


def test_detect_video_rejects_non_positive_analysis_fps(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="analysis_fps"):
        _run(tmp_path, FakeSource(frame_count=4), analysis_fps=0.0)


def test_build_document_serializes_timestamps_and_track_ids() -> None:
    detection = _detection(DamageClass.D40, 2.0)

    document = build_video_detections_document(
        source=Path("clip.mp4"),
        width=32,
        height=24,
        fps=10.0,
        frames=[(0, 0.0, (detection,), (7,))],
        model=INFO,
        score_threshold=0.25,
        analysis_fps=5.0,
        max_frames=None,
        frames_processed=1,
        unique_instances={DamageClass.D40: 0},
        coverage_ratio=0.01,
    )

    frame = document["frames"][0]
    assert frame["t"] == 0.0
    assert frame["detections"][0]["track_id"] == 7
    assert document["source"]["fps"] == 10.0
    assert document["summary"]["unique_instances"]["D40"] == 0


@settings(deadline=None, max_examples=40)
@given(
    frame_count=st.integers(min_value=1, max_value=40),
    source_fps=st.floats(min_value=1.0, max_value=60.0, allow_nan=False),
    analysis_fps=st.floats(min_value=1.0, max_value=30.0, allow_nan=False),
    max_frames=st.one_of(st.none(), st.integers(min_value=1, max_value=40)),
)
def test_progress_events_are_monotone_and_bounded(
    tmp_path_factory: pytest.TempPathFactory,
    frame_count: int,
    source_fps: float,
    analysis_fps: float,
    max_frames: int | None,
) -> None:
    root = tmp_path_factory.mktemp("video-progress")
    events = EventSink()

    result, _, _, _ = _run(
        root,
        FakeSource(frame_count=frame_count, fps=source_fps),
        analysis_fps=analysis_fps,
        max_frames=max_frames,
        progress=events,
    )

    assert events.events, "expected at least one progress event"
    currents = [event.current for event in events.events]
    assert currents == sorted(currents)
    assert currents[-1] == result.frames_analyzed
    assert all(event.total == result.frames_analyzed for event in events.events)


def test_detect_video_supports_containers_without_a_frame_count(tmp_path: Path) -> None:
    source = FakeSource(frame_count=None, produced=9)

    result, sink, _, _ = _run(tmp_path, source, analysis_fps=5.0)

    assert result.frames_processed == 9
    assert result.frames_analyzed == 5
    assert len(sink.frames) == 9
    assert result.truncated is False


def test_detect_video_supports_max_frames_without_a_frame_count(tmp_path: Path) -> None:
    source = FakeSource(frame_count=None, produced=9)

    result, _, _, _ = _run(tmp_path, source, analysis_fps=10.0, max_frames=3)

    assert result.frames_processed == 3
    assert result.truncated is True


def test_detect_video_rejects_zero_max_frames(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="max_frames"):
        _run(tmp_path, FakeSource(frame_count=4), max_frames=0)


def test_detect_video_rejects_a_stream_shorter_than_declared(tmp_path: Path) -> None:
    source = FakeSource(frame_count=5, produced=2)

    with pytest.raises(MediaError, match="ended after 2 of 5"):
        _run(tmp_path, source)
