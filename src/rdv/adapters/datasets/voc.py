"""PASCAL VOC annotation parser used by the RDD2022 dataset adapter.

The parser is deliberately strict about the file contract but forgiving about
the data: boxes crossing an image edge are clamped, boxes that collapse to zero
area are dropped, and labels outside D00/D10/D20/D40 are dropped and reported
so ``rdv data stats`` can show them. Structural problems (missing file,
malformed XML, missing ``<size>`` or a malformed box) raise
:class:`rdv.errors.DatasetError` naming the offending path (exit code 3).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from rdv.domain.annotation import Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.geometry import BBox
from rdv.errors import DatasetError


@dataclass(frozen=True, slots=True)
class VocAnnotation:
    """Parsed annotation content with dropped-label accounting."""

    width: int
    height: int
    annotations: tuple[Annotation, ...]
    dropped_labels: tuple[str, ...] = ()
    dropped_boxes: int = 0


def _text(element: ET.Element, tag: str, path: Path) -> str:
    child = element.find(tag)
    if child is None or child.text is None or not child.text.strip():
        raise DatasetError(f"Malformed VOC annotation {path}: missing <{tag}>")
    return child.text.strip()


def _dimension(element: ET.Element, tag: str, path: Path) -> int:
    raw = _text(element, tag, path)
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise DatasetError(
            f"Malformed VOC annotation {path}: <{tag}> is not a number: {raw!r}"
        ) from exc
    if value <= 0:
        raise DatasetError(f"Malformed VOC annotation {path}: <{tag}> must be positive")
    return value


def _coordinate(element: ET.Element, tag: str, path: Path) -> float:
    raw = _text(element, tag, path)
    try:
        return float(raw)
    except ValueError as exc:
        raise DatasetError(
            f"Malformed VOC annotation {path}: <{tag}> is not a number: {raw!r}"
        ) from exc


def parse_voc_file(path: Path) -> VocAnnotation:
    """Parse one VOC XML file.

    Args:
        path: annotation file to read.

    Returns:
        Image dimensions, supported annotations in document order, and counts
        of dropped labels/boxes.

    Raises:
        DatasetError: the file is missing, is not valid XML, or violates the
            required VOC structure.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise DatasetError(f"Malformed VOC annotation {path}: {exc}") from exc
    except OSError as exc:
        raise DatasetError(f"Cannot read VOC annotation {path}: {exc}") from exc

    root = tree.getroot()
    if root.tag != "annotation":
        raise DatasetError(
            f"Malformed VOC annotation {path}: root element is <{root.tag}>, expected <annotation>"
        )
    size = root.find("size")
    if size is None:
        raise DatasetError(f"Malformed VOC annotation {path}: missing <size>")
    width = _dimension(size, "width", path)
    height = _dimension(size, "height", path)

    annotations: list[Annotation] = []
    dropped_labels: list[str] = []
    dropped_boxes = 0
    for obj in root.findall("object"):
        label = _text(obj, "name", path)
        try:
            damage_class = DamageClass.from_code(label)
        except ValueError:
            dropped_labels.append(label)
            continue
        box = obj.find("bndbox")
        if box is None:
            raise DatasetError(f"Malformed VOC annotation {path}: <object> without <bndbox>")
        candidate = BBox(
            _coordinate(box, "xmin", path),
            _coordinate(box, "ymin", path),
            _coordinate(box, "xmax", path),
            _coordinate(box, "ymax", path),
        ).clipped(width, height)
        if candidate.is_degenerate:
            dropped_boxes += 1
            continue
        annotations.append(Annotation(damage_class=damage_class, bbox=candidate))

    return VocAnnotation(
        width=width,
        height=height,
        annotations=tuple(annotations),
        dropped_labels=tuple(dropped_labels),
        dropped_boxes=dropped_boxes,
    )
