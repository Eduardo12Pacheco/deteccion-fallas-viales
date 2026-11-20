"""Self-contained HTML report for one web analysis (S12).

The report embeds the annotated still (for images) as a data URI and renders a
plain, dependency-free page so a user can download one file and keep the
result offline. Every value from the detections document is HTML-escaped.
"""

from __future__ import annotations

import base64
import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CLASS_NAMES = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}
_MAX_ROWS = 500
_STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0;
       background: #f7f7f7; color: #111; }
main { max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
p.meta { margin: 0 0 24px; color: #555; }
section { background: #fff; border: 1px solid #ddd; border-radius: 12px;
          padding: 20px; margin-bottom: 20px; }
h2 { font-size: 1.1rem; margin: 0 0 12px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0; }
dt { color: #555; }
dd { margin: 0; font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e5e5; }
th { font-weight: 600; color: #333; }
img { max-width: 100%; height: auto; border-radius: 8px; display: block; }
@media (prefers-color-scheme: dark) {
  body { background: #111; color: #eee; }
  section { background: #1b1b1b; border-color: #333; }
  dt, p.meta, th { color: #aaa; }
  th, td { border-color: #333; }
}
"""


def _value(value: Any) -> str:
    return html.escape(str(value))


def _class_rows(document: Mapping[str, Any]) -> str:
    summary = document.get("summary") or {}
    counts = summary.get("counts") or {}
    unique = summary.get("unique_instances") or {}
    rows: list[str] = []
    for code in ("D00", "D10", "D20", "D40"):
        instances = unique.get(code)
        rows.append(
            "<tr>"
            f"<td>{_value(code)}</td>"
            f"<td>{_value(_CLASS_NAMES[code])}</td>"
            f"<td>{_value(counts.get(code, 0))}</td>"
            f"<td>{_value(instances if instances is not None else '-')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _detection_rows(document: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for frame in document.get("frames") or []:
        for detection in frame.get("detections") or []:
            box = detection.get("box") or []
            coordinates = ", ".join(f"{float(value):.1f}" for value in box)
            score = detection.get("score")
            score_text = f"{float(score):.3f}" if score is not None else "-"
            track = detection.get("track_id")
            rows.append(
                "<tr>"
                f"<td>{_value(frame.get('index', 0))}</td>"
                f"<td>{_value(detection.get('class', ''))}</td>"
                f"<td>{_value(detection.get('name', ''))}</td>"
                f"<td>{_value(score_text)}</td>"
                f"<td>{_value(coordinates)}</td>"
                f"<td>{_value(track if track is not None else '-')}</td>"
                "</tr>"
            )
            if len(rows) >= _MAX_ROWS:
                break
        if len(rows) >= _MAX_ROWS:
            break
    if not rows:
        return '<tr><td colspan="6">No detections at the configured threshold.</td></tr>'
    return "\n".join(rows)


def render_detections_report_html(
    document: Mapping[str, Any],
    *,
    annotated_png: Path | None = None,
) -> str:
    """Render a standalone HTML report from a ``rdv.detections/v1`` document.

    Args:
        document: detections document produced by the image/video use cases.
        annotated_png: optional rendered PNG embedded as a data URI.
    """
    source = document.get("source") or {}
    params = document.get("params") or {}
    summary = document.get("summary") or {}
    kind = str(source.get("kind", "image")).title()
    title = f"Road damage report - {_value(source.get('path', 'upload'))}"

    image_markup = ""
    if annotated_png is not None and annotated_png.is_file():
        encoded = base64.b64encode(annotated_png.read_bytes()).decode("ascii")
        image_markup = (
            "<section><h2>Annotated image</h2>"
            f'<img alt="Annotated detection overlay" src="data:image/png;base64,{encoded}">'
            "</section>"
        )

    details = [
        ("Source", source.get("path", "upload")),
        ("Kind", kind),
        ("Model", (document.get("model") or {}).get("id", "-")),
        ("Device", params.get("device", "-")),
        ("Score threshold", params.get("score_threshold", "-")),
        ("Frames analyzed", summary.get("frames_analyzed", 1)),
        ("Frames with damage", summary.get("frames_with_damage", 0)),
        ("Coverage ratio", summary.get("coverage_ratio", 0.0)),
        ("Max score", summary.get("max_score", 0.0)),
    ]
    detail_rows = "\n".join(
        f"<dt>{_value(name)}</dt><dd>{_value(value)}</dd>" for name, value in details
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>Road damage report</h1>
<p class="meta">Generated by road-damage-vision from a local analysis.</p>
{image_markup}
<section>
<h2>Summary</h2>
<dl>
{detail_rows}
</dl>
</section>
<section>
<h2>Detections by class</h2>
<table>
<thead><tr><th>Class</th><th>Name</th><th>Detections</th><th>Unique instances</th></tr></thead>
<tbody>
{_class_rows(document)}
</tbody>
</table>
</section>
<section>
<h2>Detections</h2>
<table>
<thead><tr><th>Frame</th><th>Class</th><th>Name</th><th>Score</th>
<th>Box (x1, y1, x2, y2)</th><th>Track</th></tr></thead>
<tbody>
{_detection_rows(document)}
</tbody>
</table>
</section>
</main>
</body>
</html>
"""
