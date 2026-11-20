"""Seeded synthetic RDD2022-shaped datasets for tests (decision D15a).

The generator writes a realistic directory tree::

    <root>/<Country>/train/images/<Country>_train_000000.jpg
    <root>/<Country>/train/annotations/xmls/<Country>_train_000000.xml
    <root>/splits.json

Images are procedural asphalt with drawn cracks and potholes, and every XML
box matches the drawn region exactly, so tests can assert parsed coordinates
without guessing. Everything derives from the seed: regenerating with the same
arguments produces byte-identical files.
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rdv.domain.annotation import Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.geometry import BBox
from rdv.domain.splits import block_split

DEFAULT_RATIOS = (0.6, 0.2, 0.2)


@dataclass(frozen=True, slots=True)
class SyntheticImage:
    """One generated image and the ground truth written next to it."""

    image_id: str
    split: str
    country: str
    width: int
    height: int
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Paths and ground truth of a generated dataset."""

    root: Path
    images: tuple[SyntheticImage, ...]
    split_ids: Mapping[str, tuple[str, ...]]


def _xml_for(image: SyntheticImage) -> ET.ElementTree:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = image.country
    ET.SubElement(root, "filename").text = Path(image.image_id).name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image.width)
    ET.SubElement(size, "height").text = str(image.height)
    ET.SubElement(size, "depth").text = "3"
    for annotation in image.annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = annotation.damage_class.code
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(int(annotation.bbox.x1))
        ET.SubElement(box, "ymin").text = str(int(annotation.bbox.y1))
        ET.SubElement(box, "xmax").text = str(int(annotation.bbox.x2))
        ET.SubElement(box, "ymax").text = str(int(annotation.bbox.y2))
    return ET.ElementTree(root)


def _draw(
    path: Path,
    image: SyntheticImage,
    *,
    rng: np.random.Generator,
) -> None:
    height, width = image.height, image.width
    pixels = rng.integers(70, 150, size=(height, width, 3), dtype=np.uint8)
    canvas = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for annotation in image.annotations:
        box = annotation.bbox
        coordinates = [int(box.x1), int(box.y1), int(box.x2), int(box.y2)]
        if annotation.damage_class is DamageClass.D40:
            draw.ellipse(coordinates, fill=(45, 35, 30), outline=(20, 15, 12))
        elif annotation.damage_class is DamageClass.D20:
            for offset in (0.25, 0.5, 0.75):
                x = int(box.x1 + (box.width * offset))
                draw.line([(x, int(box.y1)), (x - 3, int(box.y2))], fill=(25, 25, 25), width=2)
        elif annotation.damage_class is DamageClass.D00:
            x = int((box.x1 + box.x2) / 2)
            draw.line([(x, int(box.y1)), (x, int(box.y2))], fill=(30, 30, 30), width=2)
        else:
            y = int((box.y1 + box.y2) / 2)
            draw.line([(int(box.x1), y), (int(box.x2), y)], fill=(30, 30, 30), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=90)


def write_synthetic_dataset(
    root: Path,
    *,
    countries: Sequence[str] = ("Japan", "India"),
    images_per_country: int = 6,
    seed: int = 42,
    width: int = 96,
    height: int = 64,
    block_size: int = 2,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> SyntheticDataset:
    """Write a seeded VOC-style dataset and its ``splits.json``.

    Args:
        root: destination directory; created if missing.
        countries: country folder names (first path component of each id).
        images_per_country: generated images per country.
        seed: seed for content, boxes and split assignment.
        width: image width in pixels.
        height: image height in pixels.
        block_size: block size of the seeded split.
        ratios: train/val/test ratios for the split.

    Returns:
        The generated images and the split manifest.
    """
    if images_per_country < 1:
        raise ValueError("images_per_country must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[SyntheticImage, np.random.Generator]] = []
    for country in countries:
        for index in range(images_per_country):
            image_id = f"{country}/train/images/{country}_train_{index:06d}.jpg"
            rng = random.Random(f"{seed}:{country}:{index}")
            count = rng.choice([0, 1, 1, 2])
            annotations: list[Annotation] = []
            for _ in range(count):
                damage_class = rng.choice(list(DamageClass))
                box_width = rng.randint(8, max(9, width // 3))
                box_height = rng.randint(8, max(9, height // 2))
                x1 = rng.randint(0, width - box_width)
                y1 = rng.randint(0, height - box_height)
                annotations.append(
                    Annotation(
                        damage_class=damage_class,
                        bbox=BBox(
                            float(x1), float(y1), float(x1 + box_width), float(y1 + box_height)
                        ),
                    )
                )
            pending.append(
                (
                    SyntheticImage(
                        image_id=image_id,
                        split="",
                        country=country,
                        width=width,
                        height=height,
                        annotations=tuple(annotations),
                    ),
                    np.random.default_rng([seed, countries.index(country), index]),
                )
            )

    assignment = block_split(
        [image.image_id for image, _ in pending],
        seed=seed,
        block_size=block_size,
        ratios=ratios,
    )
    split_of = {image_id: split for split, ids in assignment.as_dict().items() for image_id in ids}

    images: list[SyntheticImage] = []
    for image, rng in pending:
        split = split_of[image.image_id]
        image = SyntheticImage(
            image_id=image.image_id,
            split=split,
            country=image.country,
            width=image.width,
            height=image.height,
            annotations=image.annotations,
        )
        image_path = root / image.image_id
        annotation_path = _annotation_path(root, image.image_id)
        _draw(image_path, image, rng=rng)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        tree = _xml_for(image)
        ET.indent(tree)
        tree.write(annotation_path, encoding="utf-8", xml_declaration=True)
        images.append(image)

    manifest = {
        "schema": "rdv.splits/v1",
        "seed": seed,
        "block_size": block_size,
        "ratios": list(ratios),
        "splits": assignment.as_dict(),
    }
    (root / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return SyntheticDataset(root=root, images=tuple(images), split_ids=assignment.as_dict())


def _annotation_path(root: Path, image_id: str) -> Path:
    image_path = Path(image_id)
    relative = image_path.relative_to(image_path.parts[0])
    parts = list(relative.parts)
    images_index = parts.index("images")
    return (
        root
        / image_path.parts[0]
        / Path(*parts[:images_index])
        / "annotations"
        / "xmls"
        / f"{image_path.stem}.xml"
    )
