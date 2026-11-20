"""Deterministic ``rdv.detections/v1`` JSON writer (S5 contract)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rdv.errors import MediaError


def dumps_document(document: Mapping[str, Any]) -> str:
    """Serialize a report document with stable key order and formatting.

    Sorted keys plus fixed indentation make repeated runs byte-identical,
    which spec S5 asserts for the same input.
    """
    try:
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise MediaError(f"Cannot serialize report document: {exc}") from exc


class DetectionsJsonWriter:
    """Write one detections document per input (``DetectionsWriter`` port)."""

    def write_detections(self, path: Path, document: Mapping[str, Any]) -> None:
        """Write ``document`` to ``path`` deterministically.

        Raises:
            MediaError: the document is not JSON-serializable or the file
                cannot be written.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps_document(document), encoding="utf-8")
        except OSError as exc:
            raise MediaError(f"Cannot write report {path}: {exc}") from exc
