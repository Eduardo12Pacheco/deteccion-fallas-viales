"""Detections JSON and aggregate summary writers (S5 contract)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdv.adapters.reporting.detections_json import DetectionsJsonWriter, dumps_document
from rdv.adapters.reporting.summary_md import SummaryWriter, render_markdown
from rdv.errors import MediaError


def _summary_document() -> dict:
    return {
        "schema": "rdv.summary/v1",
        "model": {"id": "tiny-detector", "sha256": "a" * 64, "input_size": 640},
        "params": {"score_threshold": 0.25, "device": "cpu"},
        "totals": {
            "images": 1,
            "detections": 2,
            "images_with_damage": 1,
            "max_score": 0.99,
            "mean_coverage_ratio": 0.05,
        },
        "classes": {
            "D00": {"name": "Longitudinal crack", "count": 1},
            "D10": {"name": "Transverse crack", "count": 0},
            "D20": {"name": "Alligator crack", "count": 0},
            "D40": {"name": "Pothole", "count": 1},
        },
        "images": [
            {
                "path": "a|b.jpg",
                "width": 600,
                "height": 600,
                "detections": 2,
                "max_score": 0.99,
                "coverage_ratio": 0.05,
                "annotated": "a.annotated.png",
                "detections_json": "a.detections.json",
            }
        ],
    }


def test_dumps_document_sorts_keys_and_ends_with_a_newline() -> None:
    text = dumps_document({"b": 2, "a": 1})

    assert text == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"

    DetectionsJsonWriter().write_detections(path, {"schema": "x"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"schema": "x"}


def test_non_serializable_document_is_a_media_error() -> None:
    with pytest.raises(MediaError, match="Cannot serialize"):
        DetectionsJsonWriter().write_detections(Path("unused.json"), {"bad": object()})


def test_markdown_renders_the_aggregate_sections_and_escapes_pipes() -> None:
    markdown = render_markdown(_summary_document())

    assert "# Detection summary" in markdown
    assert "`tiny-detector`" in markdown
    assert "| D00 | Longitudinal crack | 1 |" in markdown
    assert "| D40 | Pothole | 1 |" in markdown
    assert "a\\|b.jpg" in markdown


def test_summary_writer_writes_both_formats(tmp_path: Path) -> None:
    document = _summary_document()
    writer = SummaryWriter()

    json_path = tmp_path / "out" / "summary.json"
    markdown_path = tmp_path / "out" / "summary.md"
    writer.write_summary_json(json_path, document)
    writer.write_summary_markdown(markdown_path, document)

    assert json.loads(json_path.read_text(encoding="utf-8"))["totals"]["detections"] == 2
    assert "Detection summary" in markdown_path.read_text(encoding="utf-8")


def test_summary_markdown_is_deterministic(tmp_path: Path) -> None:
    writer = SummaryWriter()
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    writer.write_summary_markdown(first, _summary_document())
    writer.write_summary_markdown(second, _summary_document())

    assert first.read_bytes() == second.read_bytes()
