"""Unit tests for the self-contained HTML report renderer (S12)."""

from __future__ import annotations

from pathlib import Path

from rdv.entrypoints.web.report import render_detections_report_html


def _document(kind: str = "image") -> dict[str, object]:
    return {
        "schema": "rdv.detections/v1",
        "model": {"id": "tiny-detector"},
        "source": {"path": '<script>alert("x")</script>.jpg', "kind": kind},
        "params": {"device": "cpu", "score_threshold": 0.25},
        "frames": [
            {
                "index": 0,
                "t": None,
                "detections": [
                    {
                        "class": "D00",
                        "name": "Longitudinal crack",
                        "score": 0.9,
                        "box": [1.0, 2.0, 3.0, 4.0],
                        "track_id": 7 if kind == "video" else None,
                    }
                ],
            }
        ],
        "summary": {
            "counts": {"D00": 1, "D10": 0, "D20": 0, "D40": 0},
            "unique_instances": {"D00": 1, "D10": 0, "D20": 0, "D40": 0}
            if kind == "video"
            else None,
            "frames_analyzed": 1,
            "frames_with_damage": 1,
            "coverage_ratio": 0.01,
            "max_score": 0.9,
        },
    }


def test_report_escapes_untrusted_names_and_lists_counts() -> None:
    markup = render_detections_report_html(_document())

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "D00" in markup
    assert "Longitudinal crack" in markup
    assert "0.900" in markup
    assert "<td>1.0, 2.0, 3.0, 4.0</td>" in markup
    assert "Unique instances" in markup


def test_report_embeds_the_annotated_png_as_a_data_uri(tmp_path: Path) -> None:
    annotated = tmp_path / "annotated.png"
    annotated.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    markup = render_detections_report_html(_document(), annotated_png=annotated)

    assert 'src="data:image/png;base64,' in markup
    assert "Annotated image" in markup


def test_report_without_detections_states_so() -> None:
    document = _document()
    document["frames"] = [{"index": 0, "t": None, "detections": []}]

    markup = render_detections_report_html(document)

    assert "No detections at the configured threshold." in markup
    assert "Annotated image" not in markup
