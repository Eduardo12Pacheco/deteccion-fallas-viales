"""Train/export CLI surface and the missing-extra exit contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rdv import composition
from rdv.config.settings import Settings
from rdv.entrypoints.cli.app import app
from rdv.errors import MissingExtraError


def test_train_help_documents_the_config_and_dry_run() -> None:
    result = CliRunner().invoke(app, ["train", "--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0, result.stderr
    assert "--config" in result.stdout
    assert "--resume" in result.stdout
    assert "--benchmark-only" in result.stdout
    assert "--dry-run" in result.stdout


def test_export_help_documents_checkpoint_and_output() -> None:
    result = CliRunner().invoke(app, ["export", "--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--out" in result.stdout


def test_missing_train_extra_maps_to_exit_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "rdv.adapters.training.config", None)
    settings = Settings(data_dir=tmp_path)

    with pytest.raises(MissingExtraError, match="uv sync --extra train --extra cpu") as excinfo:
        composition.build_trainer(settings, config_path=tmp_path / "train.yaml")

    assert excinfo.value.exit_code == 6
    assert "train extra" in str(excinfo.value)
