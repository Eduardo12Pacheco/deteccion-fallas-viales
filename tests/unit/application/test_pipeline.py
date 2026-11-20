"""Staged video pipeline primitives (T7: source -> stride -> detect -> track)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
import pytest

from rdv.application.pipeline import (
    DetectStage,
    FramePacket,
    Pipeline,
    StrideSelector,
    TrackStage,
    analysis_stride,
    frame_packets,
)
from rdv.application.ports import ModelInfo
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox

INFO = ModelInfo(id="fake", input_size=640, score_threshold=0.25, device="cpu")


def _packet(index: int, *, analyzed: bool = True) -> FramePacket:
    return FramePacket(
        index=index,
        timestamp=index / 10.0,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        analyzed=analyzed,
    )


def _detection(code: DamageClass, x: float, score: float = 0.9) -> Detection:
    return Detection(code, score, BBox(x, 0.0, x + 2.0, 2.0))


class CountingDetector:
    def __init__(self, detections: Sequence[Detection]) -> None:
        self._detections = list(detections)
        self.calls: list[int] = []

    def detect(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        self.calls.append(len(images))
        return [list(self._detections) for _ in images]

    @property
    def info(self) -> ModelInfo:
        return INFO


class UpperStage:
    def __init__(self, seen: list[int]) -> None:
        self._seen = seen

    def __call__(self, items: Iterable[FramePacket]) -> Iterator[FramePacket]:
        for packet in items:
            self._seen.append(packet.index)
            yield packet


def test_analysis_stride_rounds_and_never_goes_below_one() -> None:
    assert analysis_stride(10.0, 5.0) == 2
    assert analysis_stride(29.97, 5.0) == 6
    assert analysis_stride(3.0, 5.0) == 1


@pytest.mark.parametrize(("fps", "analysis_fps"), [(0.0, 5.0), (10.0, 0.0), (-1.0, 5.0)])
def test_analysis_stride_rejects_invalid_rates(fps: float, analysis_fps: float) -> None:
    with pytest.raises(ValueError):
        analysis_stride(fps, analysis_fps)


def test_stride_selector_marks_every_nth_frame() -> None:
    selected = list(StrideSelector(fps=10.0, analysis_fps=5.0)(_packet(i) for i in range(7)))

    assert [packet.analyzed for packet in selected] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert [packet.index for packet in selected] == list(range(7))


def test_pipeline_applies_stages_in_order() -> None:
    seen: list[int] = []
    pipeline: Pipeline[FramePacket] = Pipeline(
        [UpperStage(seen), DetectStage(CountingDetector([]))]
    )

    output = list(pipeline.run([_packet(0), _packet(1)]))

    assert seen == [0, 1]
    assert [packet.index for packet in output] == [0, 1]


def test_pipeline_without_stages_is_the_identity() -> None:
    packets = [_packet(0), _packet(1)]

    assert list(Pipeline[FramePacket]([]).run(packets)) == packets


def test_detect_stage_only_runs_on_analyzed_frames() -> None:
    detector = CountingDetector([_detection(DamageClass.D00, 1.0)])
    packets = [_packet(0), _packet(1, analyzed=False), _packet(2)]

    output = list(DetectStage(detector)(packets))

    assert detector.calls == [1, 1]
    assert [len(packet.detections) for packet in output] == [1, 0, 1]


def test_detect_stage_emits_monotone_progress_for_analyzed_frames() -> None:
    events: list[tuple[int, int | None]] = []

    class Sink:
        def emit(self, event: object) -> None:
            events.append((event.current, event.total))  # type: ignore[attr-defined]

    packets = [_packet(i, analyzed=i % 2 == 0) for i in range(6)]

    list(
        DetectStage(
            CountingDetector([]),
            progress=Sink(),  # type: ignore[arg-type]
            total=3,
        )(packets)
    )

    assert events == [(1, 3), (2, 3), (3, 3)]


def test_track_stage_keeps_ids_and_repeats_the_latest_detections() -> None:
    detector = CountingDetector(
        [_detection(DamageClass.D40, 1.0, 0.8), _detection(DamageClass.D00, 4.0, 0.7)]
    )
    packets = [
        _packet(0),
        _packet(1, analyzed=False),
        _packet(2),
        _packet(3, analyzed=False),
    ]

    output = list(TrackStage()(DetectStage(detector)(packets)))

    assert output[0].track_ids == (1, 2)
    assert output[1].track_ids == (1, 2)
    assert len(output[1].detections) == 2
    assert output[2].track_ids == (1, 2)
    assert [packet.index for packet in output] == [0, 1, 2, 3]
    assert [packet.analyzed for packet in output] == [True, False, True, False]


def test_frame_packets_stop_at_max_frames_and_close_the_stream() -> None:
    closed = False

    def stream() -> Iterator[np.ndarray]:
        nonlocal closed
        try:
            for _ in range(10):
                yield np.zeros((4, 4, 3), dtype=np.uint8)
        finally:
            closed = True

    packets = list(frame_packets(stream(), fps=10.0, max_frames=4))

    assert [packet.index for packet in packets] == [0, 1, 2, 3]
    assert packets[0].timestamp == pytest.approx(0.0)
    assert packets[3].timestamp == pytest.approx(0.3)
    assert closed is True
