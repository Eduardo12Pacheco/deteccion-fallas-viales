"""Packaged model manifest: the pinned contract for downloadable weights (D13).

The manifest is the only place that knows a model URL and its sha256, so a
download can be verified independently of the server, and the CLI can explain
what is available without touching the network. Until the first fine-tune runs
the list is intentionally empty; local models are passed with ``--model``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from rdv.application.ports import DEFAULT_CLASS_CODES, ModelEntry
from rdv.errors import ModelError

SCHEMA = "rdv.models/v1"


def default_manifest_path() -> Path:
    """Location of the manifest shipped inside the installed package."""
    return Path(str(resources.files("rdv").joinpath("assets/models.json")))


class ModelManifest:
    """Validated list of publishable model artifacts."""

    def __init__(self, models: tuple[ModelEntry, ...], *, source: Path | None = None) -> None:
        seen: set[str] = set()
        for entry in models:
            if entry.id in seen:
                raise ModelError(f"Duplicate model id in manifest: {entry.id!r}")
            seen.add(entry.id)
        self._models = models
        self._by_id = {entry.id: entry for entry in models}
        self.source = source

    @property
    def models(self) -> tuple[ModelEntry, ...]:
        """Entries in manifest order."""
        return self._models

    def get(self, model_id: str) -> ModelEntry:
        """Return one entry by id.

        Raises:
            ModelError: the id is not published in the manifest.
        """
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._by_id)) or "none"
            raise ModelError(
                f"Model {model_id!r} is not in the manifest (available: {available}). "
                "Pass --model PATH to use a local ONNX file."
            ) from exc

    def find(self, model_id: str) -> ModelEntry | None:
        """Return one entry by id, or ``None`` when it is not published."""
        return self._by_id.get(model_id)

    @classmethod
    def load(cls, path: Path | None = None) -> ModelManifest:
        """Read and validate a manifest file.

        Args:
            path: manifest location; defaults to the packaged ``models.json``.

        Raises:
            ModelError: the file is missing, is not JSON, or violates the
                installed schema. Exit code 4 because it is a model problem.
        """
        source = Path(path) if path is not None else default_manifest_path()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelError(f"Model manifest not found: {source}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"Cannot read model manifest {source}: {exc}") from exc
        return cls.from_dict(payload, source=source)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, source: Path | None = None) -> ModelManifest:
        """Build a manifest from a decoded JSON mapping."""
        if not isinstance(payload, Mapping):
            raise ModelError("Model manifest must be a JSON object")
        if payload.get("schema") != SCHEMA:
            raise ModelError(f"Model manifest schema must be {SCHEMA!r}")
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise ModelError("Model manifest 'models' must be a list")
        entries: list[ModelEntry] = []
        for index, raw in enumerate(raw_models):
            if not isinstance(raw, Mapping):
                raise ModelError(f"Model manifest entry {index} must be an object")
            entries.append(_parse_entry(raw, index))
        return cls(tuple(entries), source=source)


def _parse_entry(raw: Mapping[str, Any], index: int) -> ModelEntry:
    required = (
        "id",
        "version",
        "url",
        "sha256",
        "size",
        "input_size",
        "recommended_score_threshold",
        "license",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ModelError(f"Model manifest entry {index} is missing {missing!r}")
    classes = raw.get("classes", list(DEFAULT_CLASS_CODES))
    if not isinstance(classes, list) or not all(isinstance(code, str) for code in classes):
        raise ModelError(f"Model manifest entry {index} 'classes' must be a list of strings")
    try:
        return ModelEntry(
            id=str(raw["id"]),
            version=str(raw["version"]),
            url=str(raw["url"]),
            sha256=str(raw["sha256"]),
            size=int(raw["size"]),
            input_size=int(raw["input_size"]),
            score_threshold=float(raw["recommended_score_threshold"]),
            classes=tuple(classes),
            license=str(raw["license"]),
            metrics_ref=str(raw["metrics_ref"]) if raw.get("metrics_ref") else None,
        )
    except (TypeError, ValueError) as exc:
        raise ModelError(f"Invalid model manifest entry {index}: {exc}") from exc
