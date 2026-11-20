"""End-to-end ``rdv train --dry-run`` and ``rdv export`` (S9, S4)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")

import yaml
from typer.testing import CliRunner

from rdv.entrypoints.cli.app import app
from tests.support.synthetic import write_synthetic_dataset
from tests.support.tiny_model import (
    save_tiny_checkpoint,
    tiny_train_payload,
)

pytestmark = pytest.mark.train


def _config(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    write_synthetic_dataset(
        data_dir / "rdd2022",
        countries=("Japan",),
        images_per_country=4,
        width=96,
        height=64,
        block_size=4,
        ratios=(1.0, 0.0, 0.0),
    )
    checkpoint, digest = save_tiny_checkpoint(tmp_path / "tiny")
    payload = tiny_train_payload(checkpoint, digest, epochs=1, effective_batch=4)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return data_dir, config_path


def test_train_dry_run_validates_the_config_and_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, config_path = _config(tmp_path)
    monkeypatch.setenv("RDV_DATA_DIR", str(data_dir))
    run_dir = tmp_path / "run"

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert (run_dir / "dry_run.json").is_file()


def test_export_writes_a_contract_valid_onnx_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = save_tiny_checkpoint(tmp_path / "tiny")
    output = tmp_path / "model.onnx"
    monkeypatch.setenv("RDV_DATA_DIR", str(tmp_path / "data"))

    result = CliRunner().invoke(
        app,
        ["export", "--checkpoint", str(checkpoint), "--out", str(output)],
    )

    assert result.exit_code == 0, result.stderr
    assert "Exported" in result.stdout
    assert output.is_file()


def test_train_missing_config_exits_six(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDV_DATA_DIR", str(tmp_path / "data"))

    result = CliRunner().invoke(
        app,
        ["train", "--config", str(tmp_path / "missing.yaml"), "--dry-run"],
    )

    assert result.exit_code == 6
    assert "Cannot read training config" in result.stderr
