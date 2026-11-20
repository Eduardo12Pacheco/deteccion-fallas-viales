"""Aggregate run summary as JSON and Markdown (S5).

Both formats render the same view-model so the CLI, the web app and the HTML
report (T8/T12) cannot disagree about counts or thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rdv.adapters.reporting.detections_json import dumps_document
from rdv.errors import MediaError

_CLASS_ORDER = ("D00", "D10", "D20", "D40")


def _cell(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|")


def render_markdown(document: Mapping[str, Any]) -> str:
    """Render the aggregate summary document as Markdown."""
    model = document.get("model", {})
    params = document.get("params", {})
    totals = document.get("totals", {})
    lines = [
        "# Detection summary",
        "",
        f"- Model: `{_cell(model.get('id', 'unknown'))}` "
        f"(device `{_cell(params.get('device', '?'))}`)",
        f"- Score threshold: {_cell(params.get('score_threshold', '?'))}",
        f"- Images: {_cell(totals.get('images', 0))}",
        f"- Detections: {_cell(totals.get('detections', 0))}",
        f"- Images with damage: {_cell(totals.get('images_with_damage', 0))}",
        f"- Max score: {_cell(totals.get('max_score', 0.0))}",
        "",
        "## Detections per class",
        "",
        "| Class | Name | Count |",
        "|-------|------|-------|",
    ]
    classes = document.get("classes", {})
    for code in _CLASS_ORDER:
        entry = classes.get(code, {})
        lines.append(
            f"| {code} | {_cell(entry.get('name', ''))} | {_cell(entry.get('count', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Images",
            "",
            "| File | Width | Height | Detections | Max score | Coverage | Annotated |",
            "|------|-------|--------|------------|-----------|----------|-----------|",
        ]
    )
    for image in document.get("images", []):
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    image.get("path", ""),
                    image.get("width", ""),
                    image.get("height", ""),
                    image.get("detections", 0),
                    image.get("max_score", 0.0),
                    image.get("coverage_ratio", 0.0),
                    image.get("annotated", ""),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


class SummaryWriter:
    """Persist the aggregate summary as JSON and Markdown (``SummaryWriter``)."""

    def write_summary_json(self, path: Path, document: Mapping[str, Any]) -> None:
        """Write the aggregate summary document as stable JSON."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps_document(document), encoding="utf-8")
        except OSError as exc:
            raise MediaError(f"Cannot write report {path}: {exc}") from exc

    def write_summary_markdown(self, path: Path, document: Mapping[str, Any]) -> None:
        """Render the aggregate summary as Markdown."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_markdown(document), encoding="utf-8")
        except OSError as exc:
            raise MediaError(f"Cannot write report {path}: {exc}") from exc
