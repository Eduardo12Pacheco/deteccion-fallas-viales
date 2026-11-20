"""The ``rdv data`` command surface and its error codes (S8)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from typer.testing import CliRunner

from rdv.entrypoints.cli.app import app
from tests.support.synthetic import SyntheticDataset, write_synthetic_dataset

runner = CliRunner()


def _dataset(tmp_path: Path) -> tuple[Path, SyntheticDataset]:
    data_dir = tmp_path / "data"
    dataset = write_synthetic_dataset(
        data_dir / "rdd2022", countries=("Japan", "India"), images_per_country=8
    )
    return data_dir, dataset


def test_data_help_lists_the_documented_subcommands() -> None:
    result = runner.invoke(app, ["data", "--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0
    for name in ("download", "verify", "prepare", "stats"):
        assert name in result.stdout


def test_stats_json_matches_the_generated_annotations(tmp_path: Path) -> None:
    data_dir, dataset = _dataset(tmp_path)

    result = runner.invoke(app, ["data", "stats", "--data-dir", str(data_dir), "--json"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "rdv.dataset-stats/v1"
    expected_train = Counter(image.country for image in dataset.images)
    for country in ("Japan", "India"):
        entry = payload["countries"][country]
        assert entry["images"]["train"] == expected_train[country]
        assert entry["images"].get("test", 0) == 0
        expected_classes = Counter(
            annotation.damage_class.code
            for image in dataset.images
            if image.country == country
            for annotation in image.annotations
        )
        assert entry["classes"] == dict(expected_classes)


def test_stats_table_renders_train_counts(tmp_path: Path) -> None:
    data_dir, _ = _dataset(tmp_path)

    result = runner.invoke(app, ["data", "stats", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.stderr
    assert "RDD2022 dataset" in result.stdout
    assert "Japan" in result.stdout
    assert "India" in result.stdout


def test_prepare_writes_manifest_and_lock(tmp_path: Path) -> None:
    data_dir, dataset = _dataset(tmp_path)
    lock = tmp_path / "splits.lock.json"

    result = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--countries",
            "Japan",
            "India",
            "--data-dir",
            str(data_dir),
            "--lock",
            str(lock),
            "--block-size",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stderr
    manifest = json.loads((data_dir / "rdd2022" / "splits.json").read_text(encoding="utf-8"))
    total = sum(len(ids) for ids in manifest["splits"].values())
    assert total == len(dataset.images)
    assert lock.is_file()
    assert "sha256:" in result.stdout


def test_prepare_rejects_a_tampered_lock(tmp_path: Path) -> None:
    data_dir, _ = _dataset(tmp_path)
    lock = tmp_path / "splits.lock.json"
    lock.write_text(
        json.dumps({"schema": "rdv.splits-lock/v1", "sha256": "0" * 64}), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--countries",
            "Japan",
            "India",
            "--data-dir",
            str(data_dir),
            "--lock",
            str(lock),
        ],
    )

    assert result.exit_code == 3
    assert "does not match" in result.stderr


def test_verify_without_archives_exits_three(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "verify", "--data-dir", str(tmp_path)])

    assert result.exit_code == 3
    assert "No downloaded archives" in result.stderr


def test_download_offline_fails_before_any_request(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["data", "download", "--countries", "Japan", "--data-dir", str(tmp_path)],
        env={"RDV_OFFLINE": "1"},
    )

    assert result.exit_code == 3
    assert "RDV_OFFLINE" in result.stderr
