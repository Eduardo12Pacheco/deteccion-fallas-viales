"""Deterministic split preparation and its sha256 lock (D3/S8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdv.application.prepare_dataset import prepare_dataset
from rdv.domain.splits import block_split
from rdv.errors import DatasetError

RATIOS = (0.6, 0.2, 0.2)


def _write_country_tree(root: Path, country: str, count: int) -> list[str]:
    images = root / country / "train" / "images"
    images.mkdir(parents=True)
    ids: list[str] = []
    for index in range(count):
        name = f"{country}_train_{index:06d}.jpg"
        (images / name).write_bytes(b"jpeg")
        ids.append(f"{country}/train/images/{name}")
    return ids


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "rdd2022"
    _write_country_tree(root, "Japan", 180)
    _write_country_tree(root, "India", 120)
    return root


def test_prepare_writes_manifest_and_lock_then_is_reproducible(
    dataset_root: Path, tmp_path: Path
) -> None:
    lock_path = tmp_path / "splits.lock.json"

    first = prepare_dataset(
        dataset_root,
        countries=("Japan", "India"),
        lock_path=lock_path,
        seed=42,
        block_size=10,
        ratios=RATIOS,
    )

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "rdv.splits/v1"
    assert manifest["seed"] == 42
    assert manifest["block_size"] == 10
    assert manifest["ratios"] == list(RATIOS)
    all_ids = set(manifest["splits"]["train"])
    all_ids |= set(manifest["splits"]["val"])
    all_ids |= set(manifest["splits"]["test"])
    assert len(all_ids) == 300
    expected_counts = {name: len(manifest["splits"][name]) for name in ("train", "val", "test")}
    assert first.counts == expected_counts
    assert first.lock_written is True

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["sha256"] == first.sha256
    assert lock["counts"] == first.counts

    second = prepare_dataset(
        dataset_root,
        countries=("Japan", "India"),
        lock_path=lock_path,
        seed=42,
        block_size=10,
        ratios=RATIOS,
    )

    assert second.sha256 == first.sha256
    assert second.lock_written is False
    assert second.manifest_path.read_bytes() == first.manifest_path.read_bytes()


def test_per_country_block_split_is_used(dataset_root: Path, tmp_path: Path) -> None:
    result = prepare_dataset(
        dataset_root,
        countries=("Japan", "India"),
        lock_path=tmp_path / "splits.lock.json",
        seed=42,
        block_size=10,
        ratios=RATIOS,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    japan_ids = sorted(
        str(path.relative_to(dataset_root))
        for path in (dataset_root / "Japan" / "train" / "images").glob("*.jpg")
    )
    expected = block_split(japan_ids, seed=42, block_size=10, ratios=RATIOS)
    for split in ("train", "val", "test"):
        assert set(expected.as_dict()[split]) <= set(manifest["splits"][split])


def test_tampered_lock_is_rejected(dataset_root: Path, tmp_path: Path) -> None:
    lock_path = tmp_path / "splits.lock.json"
    prepare_dataset(
        dataset_root,
        countries=("Japan",),
        lock_path=lock_path,
        block_size=10,
        ratios=RATIOS,
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="sha256"):
        prepare_dataset(
            dataset_root,
            countries=("Japan",),
            lock_path=lock_path,
            block_size=10,
            ratios=RATIOS,
        )


def test_missing_country_is_a_dataset_error(dataset_root: Path, tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Czech"):
        prepare_dataset(
            dataset_root,
            countries=("Japan", "Czech"),
            lock_path=tmp_path / "splits.lock.json",
        )


def test_missing_root_is_a_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        prepare_dataset(
            tmp_path / "missing",
            countries=("Japan",),
            lock_path=tmp_path / "splits.lock.json",
        )
