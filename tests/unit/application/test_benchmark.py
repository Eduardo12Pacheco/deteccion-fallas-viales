"""Latency/throughput benchmark use case (S11)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from rdv.application.benchmark import benchmark_model
from rdv.application.ports import ModelInfo, RgbImage
from rdv.domain.detection import Detection

INFO = ModelInfo(id="fake", input_size=640, score_threshold=0.25, device="cpu")


class CountingDetector:
    def __init__(self, info: ModelInfo = INFO) -> None:
        self.calls = 0
        self._info = info

    def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
        self.calls += len(images)
        return [[] for _ in images]

    @property
    def info(self) -> ModelInfo:
        return self._info


def test_benchmark_records_latency_percentiles_runs_and_video_throughput() -> None:
    detector = CountingDetector()

    result = benchmark_model(
        detector,
        runs=12,
        warmup=3,
        video_frames=7,
        gpu_detector=None,
        hardware_gpu=None,
    )

    document = result.as_dict()
    assert document["schema"] == "rdv.benchmark/v1"
    assert set(document["cpu"]) >= {"p50_ms", "p95_ms", "mean_ms"}
    assert document["cpu"]["p50_ms"] <= document["cpu"]["p95_ms"] >= 0.0
    assert document["cpu"]["runs"] == 12
    assert document["cpu"]["warmup"] == 3
    assert document["video"]["frames_per_s"] > 0.0
    assert document["video"]["frames"] == 7
    assert document["hardware"]["cpu"]
    assert document["hardware"]["gpu"] is None
    assert "gpu" not in document
    # 3 warmups + 12 timed + 3 video warmups + 7 video frames.
    assert detector.calls == 25


def test_benchmark_records_a_gpu_row_when_a_detector_is_available() -> None:
    gpu_info = ModelInfo(id="fake", input_size=640, score_threshold=0.25, device="cuda")

    result = benchmark_model(
        CountingDetector(),
        runs=4,
        warmup=1,
        video_frames=3,
        gpu_detector=CountingDetector(gpu_info),
        hardware_gpu="Fake GPU",
    )

    document = result.as_dict()
    assert document["gpu"]["device"] == "cuda"
    assert document["gpu"]["runs"] == 4
    assert document["hardware"]["gpu"] == "Fake GPU"


def test_benchmark_honours_the_sanity_gates() -> None:
    result = benchmark_model(
        CountingDetector(),
        runs=3,
        warmup=1,
        video_frames=2,
        gpu_detector=None,
        hardware_gpu=None,
    )

    document = result.as_dict()
    assert document["gates"]["cpu_p50_ok"] is True
    assert document["gates"]["gpu_p50_ok"] is None


def test_benchmark_input_image_is_640x640() -> None:
    class RecordingDetector(CountingDetector):
        shapes: ClassVar[list[tuple[int, ...]]] = []

        def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
            self.shapes.extend(tuple(image.shape) for image in images)
            return super().detect(images)

    detector = RecordingDetector()

    benchmark_model(
        detector,
        runs=1,
        warmup=0,
        video_frames=1,
        gpu_detector=None,
        hardware_gpu=None,
    )

    assert (640, 640, 3) in detector.shapes
    assert all(image.shape[2] == 3 for image in [np.zeros((1, 1, 3), dtype=np.uint8)])


def test_benchmark_rejects_invalid_parameters() -> None:
    detector = CountingDetector()

    import pytest

    for kwargs in (
        {"runs": 0},
        {"warmup": -1},
        {"video_frames": 0},
    ):
        with pytest.raises(ValueError):
            benchmark_model(detector, gpu_detector=None, hardware_gpu=None, **kwargs)


def test_cpu_description_falls_back_when_proc_is_unavailable(monkeypatch) -> None:
    import platform

    from rdv.application import benchmark as benchmark_module

    def broken_open(*args: object, **kwargs: object):
        raise OSError("no /proc")

    monkeypatch.setattr("builtins.open", broken_open)
    monkeypatch.setattr(platform, "processor", lambda: "Example CPU")
    assert benchmark_module.cpu_description() == "Example CPU"

    monkeypatch.setattr(platform, "processor", lambda: "")
    description = benchmark_module.cpu_description()
    assert description
    assert "/" not in description
