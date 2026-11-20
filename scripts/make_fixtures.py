#!/usr/bin/env python3
"""Generate the committed mini fixture and the packaged sample images.

The selection is deterministic (seed 42) and derived from the downloaded
RDD2022 tree, following plan decision D15:

* 12 positive images: three per damage class, one per country when possible;
* 4 negative images with no labelled object;
* each JPEG is resized so its long side is 640 px (quality 85) and the VOC XML
  is rewritten with the new size and scaled boxes;
* six samples are copied from the fixture (first of each class, second D40 and
  second D00) into ``src/rdv/assets/samples``;
* ``data/fixtures.lock.json`` records the sha256 of every produced file.

Re-running the script with the same dataset must reproduce byte-identical
fixtures; the acceptance check is ``git diff --stat`` being empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from rdv.adapters.datasets.voc import VocAnnotation, parse_voc_file
from rdv.domain.classes import DamageClass
from rdv.domain.splits import block_split

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
COUNTRIES = ("Japan", "India", "Czech")
LONG_SIDE = 640
JPEG_QUALITY = 85
PER_CLASS = 3
NEGATIVES = 4
SAMPLE_ORDER = (
    (DamageClass.D00, 0),
    (DamageClass.D10, 0),
    (DamageClass.D20, 0),
    (DamageClass.D40, 0),
    (DamageClass.D40, 1),
    (DamageClass.D00, 1),
)
LICENSE_TEXT = """\
The images and annotations in this directory are a derivative of the RDD2022
dataset (CRDDC'2022), published on Figshare under CC BY 4.0:
https://doi.org/10.6084/m9.figshare.21431547.v1

Some re-publications of the same dataset state CC BY-SA 4.0 (the license of the
accompanying paper). To satisfy both readings, these derivative images are
distributed under CC BY-SA 4.0:
https://creativecommons.org/licenses/by-sa/4.0/

The MIT license of this repository applies to the code only, not to these
images or annotations.
"""


@dataclass(frozen=True, slots=True)
class SourceImage:
    """One candidate image from the downloaded dataset."""

    image_id: str
    country: str
    path: Path
    annotation_path: Path
    parsed: VocAnnotation


def _load_sources(root: Path) -> list[SourceImage]:
    sources: list[SourceImage] = []
    for country in COUNTRIES:
        images = root / country / "train" / "images"
        if not images.is_dir():
            raise SystemExit(f"missing downloaded images for {country}: {images}")
        for path in sorted(images.glob("*.jpg")):
            annotation_path = root / country / "train" / "annotations" / "xmls" / f"{path.stem}.xml"
            if not annotation_path.is_file():
                raise SystemExit(f"missing annotation for {path}: {annotation_path}")
            sources.append(
                SourceImage(
                    image_id=f"{country}/train/images/{path.name}",
                    country=country,
                    path=path,
                    annotation_path=annotation_path,
                    parsed=parse_voc_file(annotation_path),
                )
            )
    return sources


def _pick(pool: list[SourceImage], key: str) -> SourceImage:
    ordered = sorted(pool, key=lambda source: source.image_id)
    random.Random(f"{SEED}:{key}").shuffle(ordered)
    return ordered[0]


def _select(
    sources: list[SourceImage],
) -> tuple[list[SourceImage], dict[DamageClass, list[SourceImage]]]:
    selected: list[SourceImage] = []
    selected_ids: set[str] = set()
    per_class: dict[DamageClass, list[SourceImage]] = {}

    for class_index, damage_class in enumerate(DamageClass):
        picks: list[SourceImage] = []
        for offset in range(len(COUNTRIES)):
            country = COUNTRIES[(class_index + offset) % len(COUNTRIES)]
            pool = [
                source
                for source in sources
                if source.image_id not in selected_ids
                and source.country == country
                and any(
                    annotation.damage_class is damage_class
                    for annotation in source.parsed.annotations
                )
            ]
            if not pool:
                continue
            single = [
                source
                for source in pool
                if {annotation.damage_class for annotation in source.parsed.annotations}
                == {damage_class}
            ]
            pick = _pick(single or pool, f"{damage_class.code}:{country}")
            picks.append(pick)
            selected_ids.add(pick.image_id)
            if len(picks) == PER_CLASS:
                break
        if len(picks) < PER_CLASS:
            remaining = [
                source
                for source in sources
                if source.image_id not in selected_ids
                and any(
                    annotation.damage_class is damage_class
                    for annotation in source.parsed.annotations
                )
            ]
            while len(picks) < PER_CLASS and remaining:
                pick = _pick(remaining, f"{damage_class.code}:fallback:{len(picks)}")
                picks.append(pick)
                selected_ids.add(pick.image_id)
                remaining = [source for source in remaining if source.image_id != pick.image_id]
        if len(picks) < PER_CLASS:
            raise SystemExit(f"not enough source images for class {damage_class.code}")
        per_class[damage_class] = picks
        selected.extend(picks)

    negatives = [
        source
        for source in sources
        if not source.parsed.annotations and source.image_id not in selected_ids
    ]
    if len(negatives) < NEGATIVES:
        raise SystemExit(f"not enough negative images: {len(negatives)} < {NEGATIVES}")
    ordered_negatives = sorted(negatives, key=lambda source: source.image_id)
    random.Random(f"{SEED}:negatives").shuffle(ordered_negatives)
    selected.extend(ordered_negatives[:NEGATIVES])

    # Keep the selected list ordered by role (class picks then negatives).
    return selected, per_class


def _write_resized(
    source: SourceImage, destination: Path
) -> tuple[int, int, list[dict[str, object]]]:
    with Image.open(source.path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb.size
        scale = min(1.0, LONG_SIDE / max(width, height))
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resized.save(destination, format="JPEG", quality=JPEG_QUALITY, optimize=False)

    objects: list[dict[str, object]] = []
    for annotation in source.parsed.annotations:
        box = annotation.bbox
        x1 = max(0, min(new_width - 1, round(box.x1 * scale)))
        y1 = max(0, min(new_height - 1, round(box.y1 * scale)))
        x2 = max(0, min(new_width - 1, round(box.x2 * scale)))
        y2 = max(0, min(new_height - 1, round(box.y2 * scale)))
        if x2 <= x1 or y2 <= y1:
            continue
        objects.append(
            {
                "code": annotation.damage_class.code,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return new_width, new_height, objects


def _write_xml(
    destination: Path,
    *,
    source: SourceImage,
    width: int,
    height: int,
    objects: list[dict[str, object]],
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = source.country
    ET.SubElement(root, "filename").text = Path(source.image_id).name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for obj in objects:
        element = ET.SubElement(root, "object")
        ET.SubElement(element, "name").text = str(obj["code"])
        ET.SubElement(element, "pose").text = "Unspecified"
        ET.SubElement(element, "truncated").text = "0"
        ET.SubElement(element, "difficult").text = "0"
        box = ET.SubElement(element, "bndbox")
        ET.SubElement(box, "xmin").text = str(obj["x1"])
        ET.SubElement(box, "ymin").text = str(obj["y1"])
        ET.SubElement(box, "xmax").text = str(obj["x2"])
        ET.SubElement(box, "ymax").text = str(obj["y2"])
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def _write_splits(fixture_root: Path, image_ids: list[str]) -> list[str]:
    assignment = block_split(image_ids, seed=SEED, block_size=4, ratios=(0.5, 0.25, 0.25))
    payload = {
        "schema": "rdv.splits/v1",
        "seed": SEED,
        "block_size": 4,
        "ratios": [0.5, 0.25, 0.25],
        "splits": assignment.as_dict(),
    }
    (fixture_root / "splits.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return sorted(image_ids)


def _write_attribution(
    path: Path,
    *,
    selections: list[SourceImage],
    samples: list[tuple[str, SourceImage]],
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        "Derived from the RDD2022 (CRDDC'2022) multi-national road damage dataset:",
        "",
        "- Dataset: RDD2022, released through CRDDC'2022",
        "- DOI: 10.6084/m9.figshare.21431547.v1",
        "- Figshare: https://figshare.com/articles/dataset/21431547",
        "- Paper: RDD2022: A multi-national image dataset for automatic road damage",
        "  detection, arXiv:2209.08538",
        "- Dataset license: CC BY 4.0 per the Figshare record; some mirrors state",
        "  CC BY-SA 4.0. These derivative images are distributed as CC BY-SA 4.0.",
        "",
        "Images were resized to a 640 px long side (JPEG quality 85) and their VOC",
        "boxes were scaled accordingly. They are a small illustrative subset, not the",
        "full dataset; use the Figshare source for training.",
        "",
    ]
    if samples:
        lines.extend(["## Packaged samples", "", "| Sample | Source image |", "|---|---|"])
        for name, source in samples:
            lines.append(f"| `{name}` | `{source.image_id}` |")
        lines.append("")
    lines.extend(
        ["## Fixture images", "", "| Fixture image id | Source image | Country |", "|---|---|---|"]
    )
    for source in selections:
        lines.append(f"| `{source.image_id}` | `{source.image_id}` | {source.country} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data" / "rdd2022", help="Downloaded tree."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini",
        help="Mini fixture destination.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=REPO_ROOT / "src" / "rdv" / "assets" / "samples",
        help="Packaged samples destination.",
    )
    parser.add_argument(
        "--fixtures-lock",
        type=Path,
        default=REPO_ROOT / "data" / "fixtures.lock.json",
        help="sha256 lock of every produced file.",
    )
    args = parser.parse_args(argv)

    sources = _load_sources(args.data_dir)
    selections, selection_by_class = _select(sources)

    for directory in (args.out, args.samples):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    for source in selections:
        destination = args.out / source.image_id
        width, height, objects = _write_resized(source, destination)
        annotation_destination = (
            args.out
            / source.country
            / "train"
            / "annotations"
            / "xmls"
            / f"{Path(source.image_id).stem}.xml"
        )
        _write_xml(
            annotation_destination,
            source=source,
            width=width,
            height=height,
            objects=objects,
        )

    image_ids = [source.image_id for source in selections]
    _write_splits(args.out, image_ids)
    (args.out / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    _write_attribution(
        args.out / "ATTRIBUTION.md",
        selections=selections,
        samples=[],
        title="RDD2022 mini fixture attribution",
    )

    samples: list[tuple[str, SourceImage]] = []
    for damage_class, index in SAMPLE_ORDER:
        candidates = selection_by_class.get(damage_class, [])
        if len(candidates) <= index:
            raise SystemExit(
                f"cannot pick sample {index} for {damage_class.code}: "
                f"only {len(candidates)} selection(s)"
            )
        source = candidates[index]
        name = f"{damage_class.code}-{index + 1}.jpg"
        shutil.copyfile(args.out / source.image_id, args.samples / name)
        samples.append((name, source))
    (args.samples / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    _write_attribution(
        args.samples / "ATTRIBUTION.md",
        selections=[],
        samples=samples,
        title="Packaged RDD2022 sample attribution",
    )

    produced: list[Path] = []
    produced.extend(sorted(path for path in args.out.rglob("*") if path.is_file()))
    produced.extend(sorted(path for path in args.samples.rglob("*") if path.is_file()))
    files = {_relative(path): _sha256(path) for path in produced}
    lock_payload = {
        "schema": "rdv.fixtures-lock/v1",
        "seed": SEED,
        "source": "RDD2022 (Figshare DOI 10.6084/m9.figshare.21431547.v1)",
        "selection": [
            {"fixture": source.image_id, "source": source.image_id, "country": source.country}
            for source in selections
        ],
        "files": dict(sorted(files.items())),
    }
    args.fixtures_lock.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures_lock.write_text(json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(selections)} fixture images, {len(samples)} samples, "
        f"and {len(files)} file hashes to {args.fixtures_lock}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
