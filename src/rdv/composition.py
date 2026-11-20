"""Composition root.

Concrete adapters are built here from typed settings so entrypoints never
import them directly. The data CLI uses these factories; later tasks add the
inference, model-store and reporting wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rdv.adapters.datasets.rdd2022 import DatasetTreeScan, scan_dataset_tree
from rdv.adapters.datasets.remote_zip import (
    DEFAULT_BLOCK_SIZE,
    DatasetLock,
    RemoteZipArchive,
)


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
