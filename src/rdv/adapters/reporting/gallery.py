# ruff: noqa: E501
"""Error gallery images and index page (S7/D10).

Each selected error case becomes one deterministic JPEG showing the ground
truth and predictions, plus a self-contained index that links only to files
that were just written. The Okabe-Ito palette from D11 marks predictions; the
ground-truth box is drawn thinner with a ``GT`` label.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jinja2 import Environment, select_autoescape
from PIL import Image, ImageDraw

from rdv.adapters.reporting.plots import PALETTE
from rdv.application.ports import GalleryCase, RgbImage
from rdv.domain.geometry import BBox
from rdv.domain.metrics.error_analysis import ErrorCategory
from rdv.errors import MediaError

_CATEGORY_ORDER: tuple[ErrorCategory, ...] = (
    ErrorCategory.CLASSIFICATION,
    ErrorCategory.LOCALIZATION,
    ErrorCategory.DUPLICATE,
    ErrorCategory.BACKGROUND,
    ErrorCategory.MISSED,
)
_JPEG_QUALITY = 85
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class GalleryItem:
    """One written gallery image and the case it illustrates."""

    case: GalleryCase
    path: Path
    relative_path: str


def _label(case: GalleryCase) -> str:
    parts = [case.category.value.replace("_", " "), case.damage_class.code]
    if case.score is not None:
        parts.append(f"{case.score:.2f}")
    return " ".join(parts)


def _clipped(box: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    clipped = box.clipped(width, height)
    return (
        round(clipped.x1),
        round(clipped.y1),
        round(clipped.x2),
        round(clipped.y2),
    )


def render_gallery_image(image: RgbImage, case: GalleryCase) -> Image.Image:
    """Draw the ground truth and the case's detections over a copy of the image."""
    array = np.asarray(image, dtype=np.uint8)
    height, width = int(array.shape[0]), int(array.shape[1])
    canvas = Image.fromarray(array).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(0.004 * max(width, height)))
    color = PALETTE[case.damage_class]

    for annotation in case.ground_truth:
        if annotation.damage_class is not case.damage_class:
            continue
        draw.rectangle(
            _clipped(annotation.bbox, width, height),
            outline="#ffffff",
            width=max(1, line_width - 1),
        )
        gt_box = _clipped(annotation.bbox, width, height)
        draw.text(
            (gt_box[0] + 2, gt_box[1] + 2),
            f"GT {annotation.damage_class.code}",
            fill="#ffffff",
            stroke_width=1,
            stroke_fill="#000000",
        )

    for detection in case.detections:
        if detection.damage_class is not case.damage_class:
            continue
        box = _clipped(detection.bbox, width, height)
        draw.rectangle(box, outline=color, width=line_width)
        draw.text(
            (box[0] + 2, box[1] + 2),
            f"PRED {detection.damage_class.code} {detection.score:.2f}",
            fill=color,
            stroke_width=1,
            stroke_fill="#000000",
        )

    draw.rectangle((0, height - 18, width, height), fill="#000000")
    draw.text((4, height - 16), _label(case), fill="#ffffff")
    return canvas


def _safe_stem(image_id: str) -> str:
    stem = Path(image_id.replace("\\", "/")).stem
    return _SAFE_STEM.sub("_", stem) or "image"


def write_gallery(
    output_dir: Path,
    cases: Sequence[GalleryCase],
    load_image: Callable[[str], RgbImage],
) -> tuple[GalleryItem, ...]:
    """Write one JPEG per case under ``gallery/<category>/``.

    Raises:
        MediaError: an image cannot be decoded or a file cannot be written.
    """
    root = Path(output_dir) / "gallery"
    items: list[GalleryItem] = []
    counters = dict.fromkeys(_CATEGORY_ORDER, 0)
    for case in cases:
        counters[case.category] += 1
        directory = root / case.category.value
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MediaError(f"Cannot create gallery directory {directory}: {exc}") from exc
        file_name = f"{counters[case.category]:02d}_{_safe_stem(case.image_id)}.jpg"
        path = directory / file_name
        try:
            rendered = render_gallery_image(load_image(case.image_id), case)
            rendered.save(path, format="JPEG", quality=_JPEG_QUALITY, subsampling=0)
        except OSError as exc:
            raise MediaError(f"Cannot write gallery image {path}: {exc}") from exc
        items.append(
            GalleryItem(
                case=case,
                path=path,
                relative_path=path.relative_to(Path(output_dir)).as_posix(),
            )
        )
    return tuple(items)


_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Error gallery — road-damage-vision</title>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem; color: #222; background: #fff; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; text-transform: capitalize; }
table { border-collapse: collapse; width: 100%; }
td { vertical-align: top; padding: 0.5rem 0.75rem 1.25rem 0; }
img { max-width: 320px; width: 100%; border: 1px solid #ddd; border-radius: 6px; }
caption { text-align: left; font-size: 0.8rem; color: #555; }
.empty { color: #555; }
</style>
</head>
<body>
<h1>Error gallery</h1>
<p>Selected error cases ranked by score (false positives) and ground-truth area (misses), at most 6 per category and 24 in total.</p>
{% if groups %}
{% for category, items in groups %}
<h2>{{ category }}</h2>
<table>
{% for item in items %}
<tr>
<td>
<a href="{{ item.relative_path }}"><img src="{{ item.relative_path }}" alt="{{ item.alt }}"></a>
<caption>{{ item.caption }}</caption>
</td>
</tr>
{% endfor %}
</table>
{% endfor %}
{% else %}
<p class="empty">No error cases were selected for this run.</p>
{% endif %}
</body>
</html>
"""


def write_gallery_index(output_dir: Path, items: Sequence[GalleryItem]) -> Path:
    """Render ``gallery/index.html`` referencing only the written images."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in items:
        case = item.case
        caption_parts = [case.image_id, case.damage_class.code]
        if case.score is not None:
            caption_parts.append(f"score {case.score:.3f}")
        if case.iou is not None:
            caption_parts.append(f"IoU {case.iou:.3f}")
        grouped.setdefault(case.category.value, []).append(
            {
                "relative_path": Path(item.relative_path).relative_to("gallery").as_posix(),
                "alt": f"{case.category.value} {case.damage_class.code}",
                "caption": " — ".join(caption_parts),
            }
        )
    groups = [
        (category.value, grouped[category.value])
        for category in _CATEGORY_ORDER
        if grouped.get(category.value)
    ]
    environment = Environment(autoescape=select_autoescape(["html", "xml"]), auto_reload=False)
    rendered = environment.from_string(_INDEX_TEMPLATE).render(groups=groups)
    index_path = Path(output_dir) / "gallery" / "index.html"
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise MediaError(f"Cannot write gallery index {index_path}: {exc}") from exc
    return index_path
