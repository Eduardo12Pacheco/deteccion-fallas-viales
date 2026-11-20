"""Composition-root factories for the video and benchmark commands (T7)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

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


def test_gpu_factory_uses_a_torch_checkpoint_when_cuda_is_available(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeDetector:
        def __init__(self, path: Path, **kwargs: object) -> None:
            captured["path"] = Path(path)
            captured.update(kwargs)
            self.info = None

    class FakeModule:
        TorchDetector = FakeDetector

    module = ModuleType("rdv.adapters.inference.torch_detector")
    module.TorchDetector = FakeDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(composition, "_torch_cuda_name", lambda: "Fake GPU")
    settings = Settings(model_dir=tmp_path / "models")

    detector = composition.build_benchmark_gpu_detector(settings, checkpoint=tmp_path, model=None)

    assert isinstance(detector, FakeDetector)
    assert captured["path"] == tmp_path
    assert captured["device"] == "cuda"
    assert captured["model_id"] == tmp_path.name


def test_gpu_factory_returns_none_for_a_non_checkpoint_model(monkeypatch, tmp_path: Path) -> None:
    class FakeModule:
        TorchDetector = type("TorchDetector", (), {})

    module = ModuleType("rdv.adapters.inference.torch_detector")
    module.TorchDetector = FakeModule.TorchDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(composition, "_torch_cuda_name", lambda: "Fake GPU")
    settings = Settings(model="not-a-published-model", model_dir=tmp_path / "models")

    assert composition.build_benchmark_gpu_detector(settings) is None
