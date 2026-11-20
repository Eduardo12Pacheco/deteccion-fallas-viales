"""Composition root.

Concrete adapters are built here from typed settings so entrypoints never
import them directly. The data CLI uses these factories; later tasks add the
inference, model-store and reporting wiring.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from rdv.adapters.datasets.rdd2022 import DatasetTreeScan, scan_dataset_tree
from rdv.adapters.datasets.remote_zip import (
    DEFAULT_BLOCK_SIZE,
    DatasetLock,
    RemoteZipArchive,
)
from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.adapters.inference.preprocess import INPUT_SIZE
from rdv.adapters.inference.providers import available_providers
from rdv.adapters.media.images import PillowImageSource
from rdv.adapters.media.renderer import PillowRenderer
from rdv.adapters.media.video import OpenCvVideoSink, OpenCvVideoSource
from rdv.adapters.models.manifest import ModelManifest, default_manifest_path
from rdv.adapters.models.store import CachedModelStore
from rdv.adapters.reporting.detections_json import DetectionsJsonWriter
from rdv.adapters.reporting.summary_md import SummaryWriter
from rdv.application.ports import DEFAULT_SCORE_THRESHOLD, Detector
from rdv.config.settings import Settings
from rdv.errors import ModelError


def load_dataset_lock(path: Path) -> DatasetLock:
    """Read and validate the pinned RDD2022 archive metadata."""
    return DatasetLock.load(path)


def open_dataset_archive(
    lock: DatasetLock,
    *,
    offline: bool,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> RemoteZipArchive:
    """Build the ranged archive reader for the pinned Figshare file."""
    return RemoteZipArchive(lock.url, size=lock.outer_size, offline=offline, block_size=block_size)


def scan_dataset(root: Path, countries: Sequence[str] | None = None) -> DatasetTreeScan:
    """Count a downloaded dataset tree with the VOC adapter."""
    return scan_dataset_tree(root, countries)


def model_manifest_path() -> Path:
    """Default location of the packaged model manifest (decision D13)."""
    return default_manifest_path()


def load_model_manifest(path: Path | None = None) -> ModelManifest:
    """Read the packaged or explicitly provided model manifest."""
    return ModelManifest.load(path)


def build_model_store(
    settings: Settings,
    *,
    manifest: ModelManifest | None = None,
    client: httpx.Client | None = None,
) -> CachedModelStore:
    """Build the cached model store from settings and the manifest."""
    resolved = manifest if manifest is not None else load_model_manifest()
    return CachedModelStore(
        resolved,
        settings.model_dir,
        offline=settings.offline,
        client=client,
    )


def build_detector(
    settings: Settings,
    *,
    model: str | None = None,
    score_threshold: float | None = None,
    manifest: ModelManifest | None = None,
) -> OnnxDetector:
    """Build the default ONNX detector for a manifest id or local file (D8)."""
    store = build_model_store(settings, manifest=manifest)
    selected = model or settings.model
    path = store.resolve(selected)
    entry = store.entry(selected)
    threshold = score_threshold if score_threshold is not None else settings.score_threshold
    if threshold is None:
        threshold = entry.score_threshold if entry is not None else DEFAULT_SCORE_THRESHOLD
    return OnnxDetector(
        path,
        score_threshold=threshold,
        device=settings.device,
        input_size=entry.input_size if entry is not None else INPUT_SIZE,
        model_id=entry.id if entry is not None else path.stem,
        model_version=entry.version if entry is not None else None,
        model_license=entry.license if entry is not None else None,
    )


def build_video_source(path: Path) -> OpenCvVideoSource:
    """Open a video through the OpenCV adapter (S6)."""
    return OpenCvVideoSource(path)


def build_video_sink() -> OpenCvVideoSink:
    """Atomic ``mp4v`` encoder for annotated videos (S6)."""
    return OpenCvVideoSink()


def _torch_cuda_name() -> str | None:
    """Return the first CUDA device name, or ``None`` without torch/CUDA.

    ``importlib`` is intentional: the torch stack is optional and must never be
    imported by the base install (S2), and the architecture contract forbids a
    static import outside ``rdv.adapters.training``.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    try:
        if not bool(torch.cuda.is_available()):
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:  # pragma: no cover - driver/runtime failures are not fatal
        return None


def describe_gpu() -> str | None:
    """Human-readable GPU name for benchmark reports, or ``None``."""
    return _torch_cuda_name()


def build_benchmark_gpu_detector(
    settings: Settings,
    *,
    model: str | None = None,
    manifest: ModelManifest | None = None,
) -> Detector | None:
    """Build the optional torch CUDA detector used by ``rdv benchmark`` (S11).

    The ``TorchDetector`` adapter arrives with T9; until then, or on a CPU-only
    host, this returns ``None`` and the benchmark reports the CPU row only.
    """
    if _torch_cuda_name() is None:
        return None
    try:
        module = importlib.import_module("rdv.adapters.inference.torch_detector")
    except ImportError:
        return None
    factory = getattr(module, "TorchDetector", None)
    if factory is None:
        return None
    store = build_model_store(settings, manifest=manifest)
    selected = model or settings.model
    path = store.resolve(selected)
    entry = store.entry(selected)
    threshold = settings.score_threshold
    if threshold is None:
        threshold = entry.score_threshold if entry is not None else DEFAULT_SCORE_THRESHOLD
    try:
        detector: Detector = factory(
            path,
            device="cuda",
            score_threshold=threshold,
            model_id=entry.id if entry is not None else path.stem,
        )
    except TypeError:  # pragma: no cover - factory signature drift
        return None
    return detector


def build_media_source() -> PillowImageSource:
    """Still-image decoder and PNG encoder (decision D8)."""
    return PillowImageSource()


def build_renderer() -> PillowRenderer:
    """Overlay renderer using the Okabe-Ito palette (decision D11)."""
    return PillowRenderer()


def build_detections_writer() -> DetectionsJsonWriter:
    """Deterministic ``rdv.detections/v1`` writer."""
    return DetectionsJsonWriter()


def build_summary_writer() -> SummaryWriter:
    """Aggregate summary writer (JSON and Markdown from one view-model)."""
    return SummaryWriter()


def doctor_report(settings: Settings) -> dict[str, Any]:
    """Collect environment, provider and model status for ``rdv doctor``."""
    import platform

    import onnxruntime

    from rdv import __version__

    store = build_model_store(settings)
    model: dict[str, Any] = {"requested": settings.model}
    try:
        path = store.resolve(settings.model)
    except ModelError as exc:
        entry = store.entry(settings.model)
        model.update(
            {
                "status": "missing",
                "id": settings.model,
                "path": None,
                "error": exc.message,
                "manifest_known": entry is not None,
            }
        )
    else:
        entry = store.entry(settings.model)
        model.update(
            {
                "status": "ready",
                "id": entry.id if entry is not None else path.stem,
                "path": str(path),
                "sha256": entry.sha256 if entry is not None else None,
                "error": None,
                "manifest_known": entry is not None,
            }
        )
    return {
        "schema": "rdv.doctor/v1",
        "rdv_version": __version__,
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "onnxruntime": onnxruntime.__version__,
        },
        "providers": list(available_providers()),
        "device": settings.device,
        "offline": settings.offline,
        "data_dir": str(settings.data_dir),
        "model_dir": str(settings.model_dir),
        "model": model,
    }
