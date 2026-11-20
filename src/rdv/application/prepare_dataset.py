"""Build the deterministic RDD2022 split manifest and verify its lock (D3/S8).

The split is computed from the downloaded ``<Country>/train/images`` folders,
never from a previously written manifest, so running the command twice always
reconstructs the same bytes. Its sha256 is pinned in ``data/splits.lock.json``;
the first run writes the lock and later runs refuse a different digest unless
the lock is regenerated intentionally.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdv.application.events import NullProgressSink, ProgressEvent
from rdv.application.ports import ProgressSink
from rdv.domain.splits import SplitAssignment, block_split
from rdv.errors import DatasetError

SPLITS_SCHEMA = "rdv.splits/v1"
LOCK_SCHEMA = "rdv.splits-lock/v1"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True, slots=True)
class PrepareResult:
    """Outcome of :func:`prepare_dataset`."""

    manifest_path: Path
    lock_path: Path
    sha256: str
    counts: Mapping[str, int]
    lock_written: bool


def _country_ids(root: Path, country: str) -> list[str]:
    images = root / country / "train" / "images"
    if not images.is_dir():
        raise DatasetError(f"Missing downloaded image folder for {country}: {images}")
    ids = [f"{country}/train/images/{path.name}" for path in images.glob("*.jpg") if path.is_file()]
    if not ids:
        raise DatasetError(f"No JPEG images found for {country}: {images}")
    return sorted(ids)


def _manifest_bytes(
    *,
    seed: int,
    block_size: int,
    ratios: Sequence[float],
    assignment: SplitAssignment,
) -> bytes:
    payload = {
        "schema": SPLITS_SCHEMA,
        "seed": seed,
        "block_size": block_size,
        "ratios": list(ratios),
        "splits": assignment.as_dict(),
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_lock(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read split lock {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != LOCK_SCHEMA:
        raise DatasetError(f"Unsupported split lock {path}: expected {LOCK_SCHEMA!r}")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DatasetError(f"Malformed split lock {path}: missing sha256")
    return digest


def _write_lock(path: Path, *, sha256: str, counts: Mapping[str, int], seed: int) -> None:
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "sha256": sha256,
        "seed": seed,
        "counts": dict(counts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(
    root: Path,
    *,
    countries: Sequence[str],
    lock_path: Path,
    manifest_path: Path | None = None,
    seed: int = 42,
    block_size: int = 100,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    progress: ProgressSink | None = None,
) -> PrepareResult:
    """Build ``splits.json`` from downloaded images and verify its sha256 lock.

    Args:
        root: dataset root containing ``<Country>/train/images``.
        countries: country folders to include, for example
            ``("Japan", "India", "Czech")``.
        lock_path: ``data/splits.lock.json``; written when absent.
        manifest_path: output manifest; defaults to ``<root>/splits.json``.
        seed: block-shuffle seed (project default 42).
        block_size: consecutive ids per block (project default 100).
        ratios: per-country train/val/test ratios.
        progress: optional observer for per-country progress.

    Returns:
        Paths, manifest sha256, split counts and whether the lock was written.

    Raises:
        DatasetError: a country folder is missing/empty, the existing lock
            carries a different digest, or writing the manifest fails.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset root not found: {root}")
    if not countries:
        raise ValueError("countries must not be empty")
    sink = progress or NullProgressSink()

    combined: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for index, country in enumerate(countries, start=1):
        ids = _country_ids(root, country)
        assignment = block_split(ids, seed=seed, block_size=block_size, ratios=ratios)
        for name in SPLIT_NAMES:
            combined[name].extend(assignment.as_dict()[name])
        sink.emit(
            ProgressEvent(
                stage="prepare",
                current=index,
                total=len(countries),
                message=f"{country}: {len(ids)} images",
            )
        )

    assignment = SplitAssignment(
        train=tuple(sorted(combined["train"])),
        val=tuple(sorted(combined["val"])),
        test=tuple(sorted(combined["test"])),
    )
    payload = _manifest_bytes(
        seed=seed, block_size=block_size, ratios=ratios, assignment=assignment
    )
    destination = Path(manifest_path) if manifest_path is not None else root / "splits.json"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    except OSError as exc:
        raise DatasetError(f"Cannot write split manifest {destination}: {exc}") from exc

    digest = hashlib.sha256(payload).hexdigest()
    counts = {name: len(assignment.as_dict()[name]) for name in SPLIT_NAMES}
    lock_path = Path(lock_path)
    existing = _read_lock(lock_path)
    lock_written = existing is None
    if existing is not None and existing != digest:
        raise DatasetError(
            f"Split manifest sha256 {digest} does not match {lock_path} ({existing}); "
            "delete the lock deliberately to regenerate it"
        )
    if lock_written:
        _write_lock(lock_path, sha256=digest, counts=counts, seed=seed)

    return PrepareResult(
        manifest_path=destination,
        lock_path=lock_path,
        sha256=digest,
        counts=counts,
        lock_written=lock_written,
    )
