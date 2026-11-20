"""``rdv models`` command surface and error codes (S20/D13)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rdv.entrypoints.cli.app import app

runner = CliRunner()
CONTENT = b"fake onnx model bytes"


def _write_manifest(tmp_path: Path, content: bytes = CONTENT) -> Path:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "schema": "rdv.models/v1",
                "models": [
                    {
                        "id": "demo",
                        "version": "v1",
                        "url": "https://example.invalid/demo.onnx",
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                        "input_size": 640,
                        "recommended_score_threshold": 0.3,
                        "classes": ["D00", "D10", "D20", "D40"],
                        "license": "Apache-2.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_models_help_lists_the_documented_subcommands() -> None:
    result = runner.invoke(app, ["models", "--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0
    for name in ("list", "pull", "path", "verify"):
        assert name in result.stdout


def test_list_reports_an_empty_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text('{"schema": "rdv.models/v1", "models": []}', encoding="utf-8")

    result = runner.invoke(app, ["models", "list", "--manifest", str(empty)])

    assert result.exit_code == 0, result.stderr
    assert "No models in the manifest" in result.stdout


def test_list_renders_manifest_entries(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    result = runner.invoke(
        app,
        [
            "models",
            "list",
            "--manifest",
            str(manifest),
            "--model-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert "demo" in result.stdout
    assert "Apache-2.0" in result.stdout


def test_pull_offline_exits_four_before_any_request(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    result = runner.invoke(
        app,
        ["models", "pull", "demo", "--manifest", str(manifest), "--model-dir", str(tmp_path)],
        env={"RDV_OFFLINE": "1"},
    )

    assert result.exit_code == 4
    assert "RDV_OFFLINE" in result.stderr


def test_pull_rejects_a_local_path(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    result = runner.invoke(
        app,
        [
            "models",
            "pull",
            "local.onnx",
            "--manifest",
            str(manifest),
            "--model-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 4
    assert "manifest id" in result.stderr


def test_path_requires_a_downloaded_model(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)

    result = runner.invoke(
        app,
        [
            "models",
            "path",
            "demo",
            "--manifest",
            str(manifest),
            "--model-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 4
    assert "rdv models pull" in result.stderr


def test_verify_accepts_a_published_artifact(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    model = tmp_path / "demo.onnx"
    model.write_bytes(CONTENT)

    result = runner.invoke(
        app,
        ["models", "verify", "demo", "--path", str(model), "--manifest", str(manifest)],
    )

    assert result.exit_code == 0, result.stderr
    assert hashlib.sha256(CONTENT).hexdigest() in result.stdout


def test_verify_rejects_a_tampered_artifact(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    model = tmp_path / "demo.onnx"
    model.write_bytes(b"tampered")

    result = runner.invoke(
        app,
        ["models", "verify", "demo", "--path", str(model), "--manifest", str(manifest)],
    )

    assert result.exit_code == 4
    assert "sha256 mismatch" in result.stderr


def test_verify_unknown_model_id_exits_four(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    model = tmp_path / "demo.onnx"
    model.write_bytes(CONTENT)

    result = runner.invoke(
        app,
        ["models", "verify", "nope", "--path", str(model), "--manifest", str(manifest)],
    )

    assert result.exit_code == 4
    assert "not in the manifest" in result.stderr


@pytest.mark.parametrize("command", ["list", "pull", "path", "verify"])
def test_manifest_flag_is_accepted_by_every_subcommand(command: str) -> None:
    result = runner.invoke(app, ["models", command, "--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0
    assert "--manifest" in result.stdout
