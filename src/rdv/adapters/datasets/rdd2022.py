"""RDD2022 repository: VOC annotations plus a seeded split manifest.

Directory contract (decision D14 / spec S8)::

    <root>/splits.json
    <root>/<Country>/train/images/<image>.jpg
    <root>/<Country>/train/annotations/xmls/<image>.xml

Iteration parses XML metadata and checks that the image exists, but never opens
image pixels; :meth:`Rdd2022VocRepository.load_image` is the only decode path.
That keeps evaluation scans cheap and makes laziness directly testable.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from rdv.adapters.datasets.voc import parse_voc_file
from rdv.application.ports import DatasetStats, RgbImage
from rdv.domain.annotation import AnnotatedImage
from rdv.errors import DatasetError

SPLIT_NAMES = ("train", "val", "test")
SPLIT_SCHEMA = "rdv.splits/v1"


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Validated ``splits.json`` content."""

    seed: int
    block_size: int
    ratios: tuple[float, ...]
    splits: Mapping[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        """Read and validate a split manifest.

        Raises:
            DatasetError: the file is missing, malformed, or has the wrong
                schema/keys. The message always names the path.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetError(f"Cannot read split manifest {path}: {exc}") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Malformed split manifest {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise DatasetError(f"Malformed split manifest {path}: expected a JSON object")
        if payload.get("schema") != SPLIT_SCHEMA:
            raise DatasetError(
                f"Unsupported split manifest {path}: expected schema {SPLIT_SCHEMA!r}"
            )
        raw_splits = payload.get("splits")
        if not isinstance(raw_splits, dict):
            raise DatasetError(f"Malformed split manifest {path}: missing 'splits' object")

        splits: dict[str, tuple[str, ...]] = {}
        seen: set[str] = set()
        for name in SPLIT_NAMES:
            raw_ids = raw_splits.get(name)
            if not isinstance(raw_ids, list):
                raise DatasetError(
                    f"Malformed split manifest {path}: 'splits.{name}' is not a list"
                )
            ids: list[str] = []
            for raw_id in raw_ids:
                if not isinstance(raw_id, str) or not raw_id.strip():
                    raise DatasetError(
                        f"Malformed split manifest {path}: invalid id in 'splits.{name}'"
                    )
                if raw_id in seen:
                    raise DatasetError(f"Malformed split manifest {path}: duplicate id {raw_id!r}")
                seen.add(raw_id)
                ids.append(raw_id)
            splits[name] = tuple(ids)

        seed = _int_field(payload, "seed", path)
        block_size = _int_field(payload, "block_size", path)
        raw_ratios = payload.get("ratios")
        if not isinstance(raw_ratios, list) or len(raw_ratios) != 3:
            raise DatasetError(f"Malformed split manifest {path}: 'ratios' must be three numbers")
        try:
            ratios = tuple(float(value) for value in raw_ratios)
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Malformed split manifest {path}: invalid 'ratios'") from exc
        return cls(seed=seed, block_size=block_size, ratios=ratios, splits=splits)


def _int_field(payload: Mapping[str, Any], name: str, path: Path) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(f"Malformed split manifest {path}: '{name}' must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class _Entry:
    image_id: str
    split: str
    country: str
    image_path: Path
    annotation_path: Path


class Rdd2022VocRepository:
    """``DatasetRepository`` over one downloaded RDD2022 tree.

    Args:
        root: directory that contains ``splits.json`` and the country folders.
        countries: country folder names to include, for example
            ``("Japan", "India", "Czech")``.
        split: ``train``, ``val``, ``test`` or ``all``.
        image_loader: optional pixel decoder override, mainly for tests that
            count decodes. Defaults to Pillow in RGB mode.

    Raises:
        DatasetError: the root or split manifest is missing or invalid.
        ValueError: ``countries`` is empty or ``split`` is unknown.
    """

    def __init__(
        self,
        root: Path,
        countries: Sequence[str],
        *,
        split: str = "train",
        image_loader: Callable[[Path], RgbImage] | None = None,
    ) -> None:
        if not countries or any(not country.strip() for country in countries):
            raise ValueError("countries must be a non-empty sequence of names")
        if split != "all" and split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {(*SPLIT_NAMES, 'all')}, got {split!r}")

        root = Path(root)
        if not root.is_dir():
            raise DatasetError(f"Dataset root not found: {root}")
        self._root = root.resolve()
        self._countries = tuple(countries)
        self._split = split
        self._image_loader = image_loader
        self._manifest = SplitManifest.load(self._root / "splits.json")
        selected = SPLIT_NAMES if split == "all" else (split,)
        entries: list[_Entry] = []
        for split_name in selected:
            for image_id in sorted(self._manifest.splits[split_name]):
                image_path, annotation_path, country = self._resolve(image_id)
                if country not in self._countries:
                    continue
                entries.append(
                    _Entry(
                        image_id=image_id,
                        split=split_name,
                        country=country,
                        image_path=image_path,
                        annotation_path=annotation_path,
                    )
                )
        self._entries = tuple(entries)
        self._entries_by_id = {entry.image_id: entry for entry in entries}
        self._stats: DatasetStats | None = None

    @property
    def manifest(self) -> SplitManifest:
        """The validated split manifest backing this repository."""
        return self._manifest

    def _resolve(self, image_id: str) -> tuple[Path, Path, str]:
        relative = PurePosixPath(image_id)
        parts = relative.parts
        if relative.is_absolute() or ".." in parts or len(parts) < 4 or "images" not in parts:
            raise DatasetError(f"Invalid dataset image id: {image_id!r}")
        images_index = parts.index("images")
        country = parts[0]
        image_path = (self._root / Path(*parts)).resolve()
        annotation_path = (
            self._root
            / Path(*parts[:images_index])
            / "annotations"
            / "xmls"
            / f"{relative.stem}.xml"
        ).resolve()
        for candidate in (image_path, annotation_path):
            if not candidate.is_relative_to(self._root):
                raise DatasetError(f"Invalid dataset image id: {image_id!r}")
        return image_path, annotation_path, country

    def __iter__(self) -> Iterator[AnnotatedImage]:
        """Yield metadata-only images in split then id order."""
        for entry in self._entries:
            yield self._load_annotation(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def _load_annotation(self, entry: _Entry) -> AnnotatedImage:
        if not entry.image_path.is_file():
            raise DatasetError(f"Image file not found for {entry.image_id!r}: {entry.image_path}")
        parsed = parse_voc_file(entry.annotation_path)
        return AnnotatedImage(
            image_id=entry.image_id,
            width=parsed.width,
            height=parsed.height,
            annotations=parsed.annotations,
        )

    def load_image(self, image_id: str) -> RgbImage:
        """Decode one image to an HxWx3 ``uint8`` RGB array.

        Raises:
            DatasetError: the id is not selected by this repository, the file
                is missing, or it cannot be decoded.
        """
        entry = self._entries_by_id.get(image_id)
        if entry is None:
            raise DatasetError(f"Image id is not part of this repository: {image_id!r}")
        if not entry.image_path.is_file():
            raise DatasetError(f"Image file not found for {image_id!r}: {entry.image_path}")
        if self._image_loader is not None:
            return self._as_rgb(self._image_loader(entry.image_path), entry)
        try:
            with Image.open(entry.image_path) as image:
                rgb = image.convert("RGB")
                pixels = np.asarray(rgb, dtype=np.uint8)
        except (OSError, ValueError) as exc:
            raise DatasetError(f"Cannot decode image {entry.image_path}: {exc}") from exc
        return self._as_rgb(pixels, entry)

    @staticmethod
    def _as_rgb(pixels: RgbImage, entry: _Entry) -> RgbImage:
        array = np.asarray(pixels, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise DatasetError(
                f"Image {entry.image_path} decoded to shape {array.shape}, expected HxWx3"
            )
        return np.ascontiguousarray(array)

    def stats(self) -> DatasetStats:
        """Scan annotations once and cache aggregate counts."""
        if self._stats is not None:
            return self._stats
        images: Counter[str] = Counter()
        countries: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        dropped: Counter[str] = Counter()
        for entry in self._entries:
            if not entry.image_path.is_file():
                raise DatasetError(
                    f"Image file not found for {entry.image_id!r}: {entry.image_path}"
                )
            parsed = parse_voc_file(entry.annotation_path)
            images[entry.split] += 1
            countries[entry.country] += 1
            classes.update(annotation.damage_class.code for annotation in parsed.annotations)
            dropped.update(parsed.dropped_labels)
        self._stats = DatasetStats(
            images=dict(images),
            countries=dict(countries),
            classes=dict(classes),
            dropped_labels=dict(dropped),
        )
        return self._stats


@dataclass(frozen=True, slots=True)
class CountryScan:
    """Per-country image/label counts produced by :func:`scan_dataset_tree`."""

    country: str
    images: Mapping[str, int]
    labels: int
    classes: Mapping[str, int]
    dropped_labels: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation."""
        return {
            "country": self.country,
            "images": dict(self.images),
            "labels": self.labels,
            "classes": dict(self.classes),
            "dropped_labels": dict(self.dropped_labels),
        }


@dataclass(frozen=True, slots=True)
class DatasetTreeScan:
    """Aggregate counts of a downloaded ``rdd2022`` tree."""

    countries: Mapping[str, CountryScan]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation for ``rdv data stats --json``."""
        return {
            "schema": "rdv.dataset-stats/v1",
            "countries": {name: scan.as_dict() for name, scan in self.countries.items()},
            "totals": {
                "images": sum(sum(scan.images.values()) for scan in self.countries.values()),
                "labels": sum(scan.labels for scan in self.countries.values()),
            },
        }


def scan_dataset_tree(
    root: Path,
    countries: Sequence[str] | None = None,
) -> DatasetTreeScan:
    """Count downloaded images, labels and dropped labels per country.

    Args:
        root: ``<data_dir>/rdd2022``.
        countries: folders to scan; ``None`` discovers every child folder that
            contains ``train/images``.

    Raises:
        DatasetError: the root, a requested country, or an annotation file is
            missing, or an XML file is malformed.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset root not found: {root}")
    if countries is None:
        names = sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir() and (child / "train" / "images").is_dir()
        )
    else:
        names = list(countries)
    if not names:
        raise DatasetError(f"No downloaded countries found under {root}")

    scans: dict[str, CountryScan] = {}
    for country in names:
        country_root = root / country
        if not (country_root / "train" / "images").is_dir():
            raise DatasetError(f"Missing downloaded image folder for {country}: {country_root}")
        images: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        dropped: Counter[str] = Counter()
        for split_dir in sorted(path for path in country_root.iterdir() if path.is_dir()):
            images_dir = split_dir / "images"
            if not images_dir.is_dir():
                continue
            for image_path in sorted(images_dir.glob("*.jpg")):
                annotation_path = split_dir / "annotations" / "xmls" / f"{image_path.stem}.xml"
                if not annotation_path.is_file():
                    raise DatasetError(
                        f"Missing annotation for {image_path}: expected {annotation_path}"
                    )
                parsed = parse_voc_file(annotation_path)
                images[split_dir.name] += 1
                classes.update(annotation.damage_class.code for annotation in parsed.annotations)
                dropped.update(parsed.dropped_labels)
        scans[country] = CountryScan(
            country=country,
            images=dict(images),
            labels=sum(classes.values()),
            classes=dict(classes),
            dropped_labels=dict(dropped),
        )
    return DatasetTreeScan(countries=scans)
