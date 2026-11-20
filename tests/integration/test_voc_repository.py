"""The RDD2022 VOC repository: lazy pixels, deterministic order, real errors."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from rdv.adapters.datasets.rdd2022 import Rdd2022VocRepository
from rdv.application.ports import RgbImage
from rdv.errors import DatasetError
from tests.support.synthetic import SyntheticDataset, write_synthetic_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> SyntheticDataset:
    return write_synthetic_dataset(
        tmp_path / "rdd2022",
        countries=("Japan", "India"),
        images_per_country=8,
        seed=42,
    )


def _ordered_ids(dataset: SyntheticDataset, countries: tuple[str, ...]) -> list[str]:
    ids: list[str] = []
    for split in ("train", "val", "test"):
        ids.extend(
            sorted(
                image_id
                for image_id in dataset.split_ids[split]
                if image_id.split("/", 1)[0] in countries
            )
        )
    return ids


def _annotation_path(root: Path, image_id: str) -> Path:
    image_path = Path(image_id)
    return root / image_path.parent.parent / "annotations" / "xmls" / f"{image_path.stem}.xml"


def test_iterates_selected_countries_in_deterministic_order(
    dataset: SyntheticDataset, tmp_path: Path
) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan",), split="all")
    expected = _ordered_ids(dataset, ("Japan",))

    first = [image.image_id for image in repo]
    second = [image.image_id for image in repo]

    assert first == expected
    assert second == expected
    assert len(repo) == len(expected)


def test_iteration_reuses_exact_ground_truth(dataset: SyntheticDataset, tmp_path: Path) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan", "India"), split="all")
    expected = {image.image_id: image.annotations for image in dataset.images}

    for image in repo:
        assert image.annotations == expected[image.image_id]
        assert image.width == 96
        assert image.height == 64


def test_split_filter_returns_only_that_split(dataset: SyntheticDataset, tmp_path: Path) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan", "India"), split="val")

    assert [image.image_id for image in repo] == sorted(dataset.split_ids["val"])


def test_iteration_is_lazy_and_pixels_load_on_demand(
    dataset: SyntheticDataset, tmp_path: Path
) -> None:
    calls: list[Path] = []

    def counting_loader(path: Path) -> RgbImage:
        calls.append(path)
        return np.zeros((64, 96, 3), dtype=np.uint8)

    repo = Rdd2022VocRepository(
        tmp_path / "rdd2022", ("Japan",), split="all", image_loader=counting_loader
    )

    images = list(repo)
    assert images
    assert calls == []

    pixels = repo.load_image(images[0].image_id)
    assert calls == [tmp_path / "rdd2022" / images[0].image_id]
    assert pixels.dtype == np.uint8


def test_load_image_decodes_rgb_pixels(dataset: SyntheticDataset, tmp_path: Path) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan",), split="all")
    image_id = next(iter(repo)).image_id

    pixels = repo.load_image(image_id)

    assert isinstance(pixels, np.ndarray)
    assert pixels.shape == (64, 96, 3)
    assert pixels.dtype == np.uint8
    assert pixels.flags.c_contiguous


def test_malformed_annotation_names_the_file(dataset: SyntheticDataset, tmp_path: Path) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan", "India"), split="all")
    image_id = next(iter(repo)).image_id
    xml = _annotation_path(tmp_path / "rdd2022", image_id)
    xml.write_text("<annotation><size>", encoding="utf-8")

    with pytest.raises(DatasetError) as raised:
        list(repo)

    assert raised.value.exit_code == 3
    assert str(xml) in raised.value.message


def test_missing_image_names_the_file(dataset: SyntheticDataset, tmp_path: Path) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan", "India"), split="all")
    image_id = next(iter(repo)).image_id
    image_path = tmp_path / "rdd2022" / image_id
    image_path.unlink()

    with pytest.raises(DatasetError) as raised:
        list(repo)

    assert str(image_path) in raised.value.message


def test_unknown_labels_are_dropped_and_counted(dataset: SyntheticDataset, tmp_path: Path) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan", "India"), split="all")
    image_id = next(iter(repo)).image_id
    xml = _annotation_path(tmp_path / "rdd2022", image_id)
    tree = ET.parse(xml)
    obj = ET.SubElement(tree.getroot(), "object")
    ET.SubElement(obj, "name").text = "D99"
    box = ET.SubElement(obj, "bndbox")
    for tag, value in (("xmin", 1), ("ymin", 2), ("xmax", 10), ("ymax", 12)):
        ET.SubElement(box, tag).text = str(value)
    tree.write(xml, encoding="utf-8", xml_declaration=True)

    stats = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan", "India"), split="all").stats()

    assert stats.dropped_labels == {"D99": 1}


def test_stats_aggregate_images_countries_and_classes(
    dataset: SyntheticDataset, tmp_path: Path
) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan", "India"), split="all")

    stats = repo.stats()

    class_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    for image in dataset.images:
        split_counter[image.split] += 1
        country_counter[image.country] += 1
        class_counter.update(annotation.damage_class.code for annotation in image.annotations)
    assert stats.images == dict(split_counter)
    assert stats.countries == dict(country_counter)
    assert stats.classes == dict(class_counter)
    assert stats.dropped_labels == {}
    assert repo.stats() is stats


def test_missing_split_manifest_is_a_dataset_error(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(DatasetError) as raised:
        Rdd2022VocRepository(root, ("Japan",))

    assert str(root / "splits.json") in raised.value.message


def test_unsupported_split_manifest_schema_is_rejected(
    dataset: SyntheticDataset, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "rdd2022" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "rdv.splits/v0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetError) as raised:
        Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan",))

    assert "rdv.splits/v1" in raised.value.message


def test_path_traversal_in_the_manifest_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "rdd2022"
    root.mkdir()
    (root / "splits.json").write_text(
        json.dumps(
            {
                "schema": "rdv.splits/v1",
                "seed": 42,
                "block_size": 100,
                "ratios": [0.8, 0.1, 0.1],
                "splits": {"train": ["Japan/../../etc/passwd"], "val": [], "test": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError) as raised:
        Rdd2022VocRepository(root, ("Japan",))

    assert "Japan/../../etc/passwd" in raised.value.message


def test_unknown_split_name_is_a_value_error(dataset: SyntheticDataset, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan",), split="holdout")


def test_unknown_image_id_is_a_dataset_error(dataset: SyntheticDataset, tmp_path: Path) -> None:
    repo = Rdd2022VocRepository(tmp_path / "rdd2022", ("Japan",), split="all")

    with pytest.raises(DatasetError):
        repo.load_image("Japan/train/images/not-in-manifest.jpg")
