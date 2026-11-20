# ruff: noqa: E501
"""Self-contained HTML evaluation report (S7).

Jinja2 renders one file with inline styles and base64-encoded plot and gallery
thumbnails, so the report can be opened offline and moved between machines.
Autoescaping is always on.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape
from PIL import Image

from rdv.adapters.reporting.gallery import GalleryItem
from rdv.domain.classes import DamageClass
from rdv.errors import MediaError

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation report — road-damage-vision</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0 auto;
       max-width: 1024px; padding: 2rem 1rem 4rem; line-height: 1.5; color: #1d1d1f;
       background: #ffffff; }
@media (prefers-color-scheme: dark) {
  body { color: #f2f2f7; background: #161618; }
  table { border-color: #3a3a3c; }
  th, td { border-color: #3a3a3c; }
  .card { background: #1f1f22; }
  code { background: #2c2c2e; }
}
h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.2rem; margin-top: 2rem; }
p.lead { color: #666; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; }
th, td { border: 1px solid #d2d2d7; padding: 0.35rem 0.6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: rgba(127, 127, 127, 0.12); font-weight: 600; }
code { background: #f2f2f7; padding: 0.1rem 0.3rem; border-radius: 4px; }
.card { background: #f8f8fa; border-radius: 10px; padding: 1rem 1.25rem; margin: 1rem 0; }
.plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
.plots img { width: 100%; border-radius: 8px; background: #fff; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }
.gallery figure { margin: 0; }
.gallery img { width: 100%; border-radius: 8px; display: block; }
.gallery figcaption { font-size: 0.8rem; color: #666; margin-top: 0.35rem; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; }
dt { font-weight: 600; }
</style>
</head>
<body>
<h1>Evaluation report</h1>
<p class="lead">road-damage-vision — {{ model.id }} on split <code>{{ split }}</code></p>

<div class="card">
<dl>
<dt>Images</dt><dd>{{ dataset.images }}</dd>
<dt>Ground-truth boxes</dt><dd>{{ dataset.ground_truth_boxes }}</dd>
<dt>Operating threshold</dt><dd>{{ operating.threshold }} ({{ operating_source }})</dd>
<dt>Device</dt><dd>{{ params.device }}</dd>
</dl>
</div>

<h2>Metrics</h2>
<table>
<thead><tr><th>Class</th><th>Ground truth</th><th>AP@0.5</th><th>AP@0.5:0.95</th></tr></thead>
<tbody>
{% for row in class_rows %}
<tr><td>{{ row.code }} — {{ row.name }}</td><td>{{ row.num_gt }}</td><td>{{ row.ap50 }}</td><td>{{ row.ap }}</td></tr>
{% endfor %}
<tr><th>mAP</th><th>{{ dataset.ground_truth_boxes }}</th><th>{{ metrics.map_50 }}</th><th>{{ metrics.map_50_95 }}</th></tr>
</tbody>
</table>

<h2>Operating point</h2>
<table>
<thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th></tr></thead>
<tbody><tr>
<td>{{ operating.threshold }}</td><td>{{ operating.precision }}</td><td>{{ operating.recall }}</td>
<td>{{ operating.f1 }}</td><td>{{ operating.true_positives }}</td><td>{{ operating.false_positives }}</td>
</tr></tbody>
</table>

<h2>Per country</h2>
<table>
<thead><tr><th>Country</th><th>Images</th><th>Boxes</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th></tr></thead>
<tbody>
{% for row in country_rows %}
<tr><td>{{ row.country }}</td><td>{{ row.images }}</td><td>{{ row.boxes }}</td><td>{{ row.map50 }}</td><td>{{ row.map }}</td></tr>
{% endfor %}
</tbody>
</table>

{% if ood_row %}
<h2>Out of domain</h2>
<table>
<thead><tr><th>Set</th><th>Images</th><th>Boxes</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th></tr></thead>
<tbody><tr><td>OOD</td><td>{{ ood_row.images }}</td><td>{{ ood_row.boxes }}</td><td>{{ ood_row.map50 }}</td><td>{{ ood_row.map }}</td></tr></tbody>
</table>
{% endif %}

<h2>Error taxonomy</h2>
<table>
<thead><tr><th>Category</th><th>Count</th></tr></thead>
<tbody>
{% for row in error_rows %}<tr><td>{{ row.category }}</td><td>{{ row.count }}</td></tr>{% endfor %}
<tr><th>False positives</th><th>{{ errors.total_false_positives }}</th></tr>
<tr><th>Missed</th><th>{{ errors.total_missed }}</th></tr>
</tbody>
</table>

<h2>Confusion matrix</h2>
<table>
<thead><tr><th>Truth \\ prediction</th>{% for label in confusion.labels %}<th>{{ label }}</th>{% endfor %}</tr></thead>
<tbody>
{% for row in confusion_rows %}<tr><th>{{ row.label }}</th>{% for value in row.cells %}<td>{{ value }}</td>{% endfor %}</tr>{% endfor %}
</tbody>
</table>

<h2>Plots</h2>
<div class="plots">
{% if pr_curves %}<figure><img src="{{ pr_curves }}" alt="Precision-recall curves at IoU 0.5"><figcaption>PR curves at IoU 0.5.</figcaption></figure>{% endif %}
{% if confusion_plot %}<figure><img src="{{ confusion_plot }}" alt="Confusion matrix at IoU 0.5"><figcaption>Confusion matrix at IoU 0.5.</figcaption></figure>{% endif %}
</div>

<h2>Error gallery</h2>
{% if gallery_groups %}
{% for group in gallery_groups %}
<h3>{{ group.category }}</h3>
<div class="gallery">
{% for item in group.cases %}
<figure><img src="{{ item.data_uri }}" alt="{{ item.alt }}"><figcaption>{{ item.caption }}</figcaption></figure>
{% endfor %}
</div>
{% endfor %}
{% else %}
<p>No error cases were selected for this run.</p>
{% endif %}

<footer><p class="lead">Generated by {{ tool.name }} {{ tool.version }}. Metrics follow COCO semantics; see docs/EVALUATION.md.</p></footer>
</body>
</html>
"""

_CATEGORY_ORDER = (
    "classification",
    "localization",
    "duplicate",
    "background",
    "missed",
)


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _read_image_data_uri(path: Path, *, thumbnail: bool) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MediaError(f"Cannot read report image {path}: {exc}") from exc
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    if thumbnail:
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.thumbnail((480, 480))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=80)
            raw = buffer.getvalue()
            mime = "image/jpeg"
        except OSError as exc:
            raise MediaError(f"Cannot decode report image {path}: {exc}") from exc
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def render_report_html(
    document: Mapping[str, Any],
    *,
    pr_curves_png: Path | None,
    confusion_png: Path | None,
    gallery_items: Sequence[GalleryItem],
) -> str:
    """Render the self-contained HTML report with autoescaping enabled."""
    metrics = document["metrics"]
    class_rows = [
        {
            "code": damage_class.code,
            "name": damage_class.display_name,
            "num_gt": metrics["per_class"][damage_class.code]["num_gt"],
            "ap50": _format(metrics["per_class"][damage_class.code]["ap_50"]),
            "ap": _format(metrics["per_class"][damage_class.code]["ap_50_95"]),
        }
        for damage_class in DamageClass
    ]
    country_rows = [
        {
            "country": country,
            "images": values["images"],
            "boxes": values["ground_truth_boxes"],
            "map50": _format(values["map_50"]),
            "map": _format(values["map_50_95"]),
        }
        for country, values in document["per_country"].items()
    ]
    ood = document.get("ood")
    ood_row = (
        {
            "images": ood["images"],
            "boxes": ood["ground_truth_boxes"],
            "map50": _format(ood["map_50"]),
            "map": _format(ood["map_50_95"]),
        }
        if ood is not None
        else None
    )
    error_rows = [
        {"category": category, "count": document["errors"]["counts"].get(category, 0)}
        for category in _CATEGORY_ORDER
    ]
    confusion = document["confusion_matrix"]
    confusion_rows = [
        {"label": label, "cells": values}
        for label, values in zip(confusion["labels"], confusion["values"], strict=True)
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in gallery_items:
        caption_parts = [item.case.image_id, item.case.damage_class.code]
        if item.case.score is not None:
            caption_parts.append(f"score {item.case.score:.3f}")
        if item.case.iou is not None:
            caption_parts.append(f"IoU {item.case.iou:.3f}")
        grouped.setdefault(item.case.category.value, []).append(
            {
                "data_uri": _read_image_data_uri(item.path, thumbnail=True),
                "alt": f"{item.case.category.value} {item.case.damage_class.code}",
                "caption": " — ".join(caption_parts),
            }
        )
    gallery_groups = [
        {"category": category, "cases": grouped[category]}
        for category in _CATEGORY_ORDER
        if grouped.get(category)
    ]
    environment = Environment(autoescape=select_autoescape(["html", "xml"]), auto_reload=False)
    return environment.from_string(_TEMPLATE).render(
        tool=document["tool"],
        model=document["model"],
        split=document["params"]["split"],
        dataset=document["dataset"],
        params=document["params"],
        operating=document["operating_point"],
        operating_source=document["params"]["operating_source"],
        metrics=metrics,
        class_rows=class_rows,
        country_rows=country_rows,
        ood_row=ood_row,
        errors=document["errors"],
        error_rows=error_rows,
        confusion=confusion,
        confusion_rows=confusion_rows,
        pr_curves=(
            _read_image_data_uri(pr_curves_png, thumbnail=False)
            if pr_curves_png is not None and pr_curves_png.is_file()
            else None
        ),
        confusion_plot=(
            _read_image_data_uri(confusion_png, thumbnail=False)
            if confusion_png is not None and confusion_png.is_file()
            else None
        ),
        gallery_groups=gallery_groups,
    )
