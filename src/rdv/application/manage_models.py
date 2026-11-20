"""Use cases around model artifacts: list, resolve, pull and verify (D13).

The entrypoints translate these results to text/JSON; they never touch HTTP or
the filesystem themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rdv.application.ports import ModelEntry, ModelStore


def list_models(store: ModelStore) -> tuple[ModelEntry, ...]:
    """Return every published model entry in manifest order."""
    return store.entries()


def resolve_model(store: ModelStore, model: str) -> Path:
    """Return the local file for a manifest id or an explicit ONNX path."""
    return store.resolve(model)


def pull_model(
    store: ModelStore,
    model_id: str,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download, verify and install a manifest model.

    Raises:
        rdv.errors.ModelError: the id is unknown, the download fails or the
            artifact does not match the pinned sha256.
    """
    return store.pull(model_id, progress)


def verify_model(store: ModelStore, path: Path, model_id: str | None = None) -> ModelEntry:
    """Verify a local artifact against its manifest entry."""
    return store.verify(path, model_id)
