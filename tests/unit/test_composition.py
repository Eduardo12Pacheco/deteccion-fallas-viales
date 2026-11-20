"""Composition-root factories for the video and benchmark commands (T7)."""

from __future__ import annotations

from pathlib import Path

from rdv import composition
from rdv.adapters.media.video import OpenCvVideoSink, OpenCvVideoSource
from rdv.config.settings import Settings


def test_video_factories_build_the_opencv_adapters(tmp_path: Path) -> None:
    from tests.support.synthetic_video import write_moving_rectangle_video

    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=3)
    source = composition.build_video_source(clip)
    try:
        assert isinstance(source, OpenCvVideoSource)
        assert source.frame_count == 3
    finally:
        source.close()
    assert isinstance(composition.build_video_sink(), OpenCvVideoSink)


def test_gpu_helpers_are_safe_without_torch(monkeypatch) -> None:
    monkeypatch.setattr(composition, "_torch_cuda_name", lambda: None)
    settings = Settings(model_dir=Path("/tmp/rdv-composition-models"))

    assert composition.describe_gpu() is None
    assert composition.build_benchmark_gpu_detector(settings) is None
