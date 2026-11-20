"""Latency and throughput benchmark use case (spec S11).

The measurement protocol is fixed by the plan: batch 1, 640x640 inputs, 20
warm-up runs and 100 timed runs for ONNX CPU (plus a torch GPU row when a CUDA
detector is available), and analyzed-frames-per-second on synthetic 320x240
frames. Only sanity gates are evaluated; the numbers are never tuned.
"""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rdv.application.ports import Detector, RgbImage

SCHEMA = "rdv.benchmark/v1"
INPUT_SIZE = 640
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
CPU_P50_LIMIT_MS = 1500.0
GPU_P50_LIMIT_MS = 100.0

Clock = Callable[[], float]


def cpu_description() -> str:
    """Best-effort CPU model name without reading outside the process.

    ``/proc/cpuinfo`` is used on Linux; ``platform.processor()`` and the machine
    architecture are the portable fallbacks for Windows and macOS.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                key, separator, value = line.partition(":")
                if separator and key.strip() == "model name" and value.strip():
                    return value.strip()
    except OSError:
        pass
    processor = platform.processor().strip()
    if processor:
        return processor
    return f"{platform.system()} {platform.machine()}".strip() or "unknown"


def _timings_ms(
    detect: Callable[[RgbImage], object],
    image: RgbImage,
    *,
    warmup: int,
    runs: int,
    clock: Clock,
) -> list[float]:
    for _ in range(warmup):
        detect(image)
    timings: list[float] = []
    for _ in range(runs):
        started = clock()
        detect(image)
        timings.append((clock() - started) * 1000.0)
    return timings


def _row(timings: list[float], *, device: str, warmup: int) -> dict[str, Any]:
    ordered = sorted(timings)
    p50 = float(np.percentile(ordered, 50))
    p95 = float(np.percentile(ordered, 95))
    mean = float(statistics.fmean(ordered))
    return {
        "device": device,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "mean_ms": round(mean, 3),
        "min_ms": round(min(ordered), 3),
        "max_ms": round(max(ordered), 3),
        "runs": len(ordered),
        "warmup": warmup,
        "batch": 1,
        "input_size": INPUT_SIZE,
        "fps": round(1000.0 / mean, 3) if mean > 0.0 else None,
    }


def _video_throughput(
    detector: Detector,
    *,
    frames: int,
    clock: Clock,
    warmup: int = 3,
) -> tuple[dict[str, Any], int]:
    video_frames = [
        np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), index % 255, dtype=np.uint8)
        for index in range(max(1, frames))
    ]
    for frame in video_frames[:warmup]:
        detector.detect([frame])
    started = clock()
    for frame in video_frames:
        detector.detect([frame])
    elapsed = clock() - started
    fps = len(video_frames) / elapsed if elapsed > 0.0 else math.inf
    return (
        {
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "frames": len(video_frames),
            "frames_per_s": round(fps, 3),
        },
        len(video_frames) + min(warmup, len(video_frames)),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """JSON-ready benchmark outcome (``rdv.benchmark/v1``)."""

    model: dict[str, Any]
    cpu: dict[str, Any]
    gpu: dict[str, Any] | None
    video: dict[str, Any]
    hardware: dict[str, Any]
    config: dict[str, Any]
    gates: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result with the documented envelope."""
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "model": self.model,
            "config": self.config,
            "cpu": self.cpu,
            "video": self.video,
            "hardware": self.hardware,
            "gates": self.gates,
        }
        if self.gpu is not None:
            document["gpu"] = self.gpu
        return document


def benchmark_model(
    detector: Detector,
    *,
    runs: int = 100,
    warmup: int = 20,
    video_frames: int = 50,
    gpu_detector: Detector | None = None,
    hardware_gpu: str | None = None,
    clock: Clock = time.perf_counter,
) -> BenchmarkResult:
    """Measure latency, video throughput and the S11 sanity gates.

    Args:
        detector: CPU ONNX detector under test.
        runs: timed single-image runs.
        warmup: untimed runs before timing.
        video_frames: synthetic frames for the throughput measurement.
        gpu_detector: optional CUDA detector (torch backend, T9).
        hardware_gpu: GPU name when a CUDA detector exists.
        clock: monotonic clock, injectable for deterministic tests.

    Raises:
        ValueError: ``runs`` or ``video_frames`` is not positive, or
            ``warmup`` is negative.
    """
    if runs < 1:
        raise ValueError(f"runs must be positive, got {runs!r}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup!r}")
    if video_frames < 1:
        raise ValueError(f"video_frames must be positive, got {video_frames!r}")

    image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    cpu_timings = _timings_ms(
        lambda frame: detector.detect([frame]),
        image,
        warmup=warmup,
        runs=runs,
        clock=clock,
    )
    cpu = _row(cpu_timings, device=detector.info.device, warmup=warmup)
    video, _ = _video_throughput(detector, frames=video_frames, clock=clock)

    gpu: dict[str, Any] | None = None
    if gpu_detector is not None:
        gpu_timings = _timings_ms(
            lambda frame: gpu_detector.detect([frame]),
            image,
            warmup=warmup,
            runs=runs,
            clock=clock,
        )
        gpu = _row(gpu_timings, device=gpu_detector.info.device, warmup=warmup)

    info = detector.info
    return BenchmarkResult(
        model={
            "id": info.id,
            "sha256": info.sha256,
            "input_size": info.input_size,
            "score_threshold": info.score_threshold,
        },
        cpu=cpu,
        gpu=gpu,
        video=video,
        hardware={
            "cpu": cpu_description(),
            "cores": os.cpu_count() or 1,
            "platform": platform.platform(),
            "gpu": hardware_gpu,
        },
        config={
            "batch": 1,
            "input_size": INPUT_SIZE,
            "warmup": warmup,
            "runs": runs,
            "video_frames": video_frames,
            "video_size": [VIDEO_WIDTH, VIDEO_HEIGHT],
        },
        gates={
            "cpu_p50_ok": cpu["p50_ms"] <= CPU_P50_LIMIT_MS,
            "cpu_p50_limit_ms": CPU_P50_LIMIT_MS,
            "gpu_p50_ok": (gpu["p50_ms"] <= GPU_P50_LIMIT_MS) if gpu is not None else None,
            "gpu_p50_limit_ms": GPU_P50_LIMIT_MS,
        },
    )
