"""Composition root.

Concrete adapters are built here from typed settings so entrypoints never
import them directly. The data CLI uses these factories; later tasks add the
inference, model-store and reporting wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import httpx

from rdv.adapters.datasets.rdd2022 import DatasetTreeScan, scan_dataset_tree
from rdv.adapters.datasets.remote_zip import (
    DEFAULT_BLOCK_SIZE,
    DatasetLock,
    RemoteZipArchive,
)
from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.adapters.inference.preprocess import INPUT_SIZE
from rdv.adapters.media.images import PillowImageSource
from rdv.adapters.media.renderer import PillowRenderer
from rdv.adapters.models.manifest import ModelManifest, default_manifest_path
from rdv.adapters.models.store import CachedModelStore
from rdv.adapters.reporting.detections_json import DetectionsJsonWriter
from rdv.adapters.reporting.summary_md import SummaryWriter
from rdv.application.ports import DEFAULT_SCORE_THRESHOLD
from rdv.config.settings import Settings


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
