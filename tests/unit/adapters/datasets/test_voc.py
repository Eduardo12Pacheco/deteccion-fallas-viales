"""PASCAL VOC parsing: labels, clamping, and malformed input."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from rdv.adapters.datasets.voc import parse_voc_file
from rdv.domain.classes import DamageClass
from rdv.errors import DatasetError


def _write_xml(
    path: Path,
    *,
    width: int = 100,
    height: int = 80,
    objects: list[tuple[str, tuple[int, int, int, int]]] | None = None,
    size: bool = True,
) -> Path:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = path.with_suffix(".jpg").name
    if size:
        size_node = ET.SubElement(root, "size")
        ET.SubElement(size_node, "width").text = str(width)
        ET.SubElement(size_node, "height").text = str(height)
        ET.SubElement(size_node, "depth").text = "3"
    for label, coordinates in objects or []:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        box = ET.SubElement(obj, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), coordinates, strict=True):
            ET.SubElement(box, tag).text = str(value)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_parses_size_and_boxes_in_document_order(tmp_path: Path) -> None:
    xml = _write_xml(
        tmp_path / "sample.xml",
        width=640,
        height=480,
        objects=[
            ("D00", (10, 20, 110, 220)),
            ("D40", (5, 6, 7, 8)),
        ],
    )

    parsed = parse_voc_file(xml)

    assert parsed.width == 640
    assert parsed.height == 480
    assert [annotation.damage_class for annotation in parsed.annotations] == [
        DamageClass.D00,
        DamageClass.D40,
    ]
    assert parsed.annotations[0].bbox.to_xyxy() == (10.0, 20.0, 110.0, 220.0)
    assert parsed.dropped_labels == ()
    assert parsed.dropped_boxes == 0


def test_boxes_are_clamped_to_the_image(tmp_path: Path) -> None:
    xml = _write_xml(
        tmp_path / "clamped.xml",
        width=100,
        height=80,
        objects=[("D10", (-10, -5, 50, 40))],
    )

    parsed = parse_voc_file(xml)

    assert parsed.annotations[0].bbox.to_xyxy() == (0.0, 0.0, 50.0, 40.0)


def test_degenerate_boxes_are_dropped_and_counted(tmp_path: Path) -> None:
    xml = _write_xml(
        tmp_path / "degenerate.xml",
        width=100,
        height=80,
        objects=[("D20", (10, 10, 10, 40)), ("D00", (0, 0, 10, 10))],
    )

    parsed = parse_voc_file(xml)

    assert [annotation.damage_class for annotation in parsed.annotations] == [DamageClass.D00]
    assert parsed.dropped_boxes == 1


def test_unknown_labels_are_dropped_and_counted(tmp_path: Path) -> None:
    xml = _write_xml(
        tmp_path / "unknown.xml",
        objects=[("D99", (1, 2, 30, 40)), ("D43", (1, 2, 30, 40)), ("D40", (5, 5, 20, 20))],
    )

    parsed = parse_voc_file(xml)

    assert [annotation.damage_class for annotation in parsed.annotations] == [DamageClass.D40]
    assert parsed.dropped_labels == ("D99", "D43")


def test_malformed_xml_names_the_file(tmp_path: Path) -> None:
    xml = tmp_path / "broken.xml"
    xml.write_text("<annotation><size>", encoding="utf-8")

    with pytest.raises(DatasetError) as raised:
        parse_voc_file(xml)

    assert raised.value.exit_code == 3
    assert str(xml) in raised.value.message


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    xml = tmp_path / "absent.xml"

    with pytest.raises(DatasetError) as raised:
        parse_voc_file(xml)

    assert str(xml) in raised.value.message


def test_missing_size_block_names_the_file(tmp_path: Path) -> None:
    xml = _write_xml(tmp_path / "no-size.xml", size=False, objects=[("D00", (1, 2, 3, 4))])

    with pytest.raises(DatasetError) as raised:
        parse_voc_file(xml)

    assert str(xml) in raised.value.message


def test_invalid_coordinates_name_the_file(tmp_path: Path) -> None:
    xml = _write_xml(tmp_path / "bad-box.xml", objects=[("D00", (1, 2, 3, 4))])
    tree = ET.parse(xml)
    tree.getroot().find("object/bndbox/xmax").text = "not-a-number"  # type: ignore[union-attr]
    tree.write(xml, encoding="utf-8", xml_declaration=True)

    with pytest.raises(DatasetError) as raised:
        parse_voc_file(xml)

    assert str(xml) in raised.value.message


def test_object_without_a_name_names_the_file(tmp_path: Path) -> None:
    xml = _write_xml(tmp_path / "nameless.xml", objects=[("D00", (1, 2, 3, 4))])
    tree = ET.parse(xml)
    name = tree.getroot().find("object/name")
    assert name is not None
    tree.getroot().find("object").remove(name)  # type: ignore[union-attr]
    tree.write(xml, encoding="utf-8", xml_declaration=True)

    with pytest.raises(DatasetError) as raised:
        parse_voc_file(xml)

    assert str(xml) in raised.value.message
