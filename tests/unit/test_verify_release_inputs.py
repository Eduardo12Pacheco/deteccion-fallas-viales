"""Release inputs are verified before any release artifact is built (S18)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "verify_release_inputs.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _manifest(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "rdv-test-model",
        "version": "v1",
        "url": (
            "https://github.com/Eduardo12Pacheco/deteccion-fallas-viales/"
            "releases/download/models-v1/rdv-test-model-v1.onnx"
        ),
        "sha256": "a" * 64,
        "size": 128,
        "classes": ["D00", "D10", "D20", "D40"],
        "input_size": 640,
        "recommended_score_threshold": 0.01,
        "license": "Apache-2.0 (weights); RDD2022 derivatives CC BY-SA 4.0",
        "metrics_ref": "reports/v1/metrics.json",
    }
    entry.update(overrides)
    return {"schema": "rdv.models/v1", "models": [entry]}


def _write_inputs(
    tmp_path: Path, *, version: str = "0.1.0", manifest: object | None = None
) -> Path:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[project]\nname = "road-damage-vision"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (tmp_path / "models.json").write_text(
        json.dumps(_manifest() if manifest is None else manifest), encoding="utf-8"
    )
    return pyproject


def test_current_repository_tag_passes() -> None:
    result = _run("v0.1.0")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "0.1.0" in result.stdout


def test_tag_must_match_the_project_version() -> None:
    result = _run("v9.9.9")
    assert result.returncode == 1
    assert "v9.9.9" in result.stderr
    assert "0.1.0" in result.stderr


def test_manifest_entry_must_pin_a_sha256(tmp_path: Path) -> None:
    pyproject = _write_inputs(tmp_path, manifest=_manifest(sha256=""))
    result = _run(
        "v0.1.0",
        "--pyproject",
        str(pyproject),
        "--manifest",
        str(tmp_path / "models.json"),
    )
    assert result.returncode == 1
    assert "sha256" in result.stderr


def test_manifest_must_not_be_empty(tmp_path: Path) -> None:
    pyproject = _write_inputs(tmp_path, manifest={"schema": "rdv.models/v1", "models": []})
    result = _run(
        "v0.1.0",
        "--pyproject",
        str(pyproject),
        "--manifest",
        str(tmp_path / "models.json"),
    )
    assert result.returncode == 1
    assert "models" in result.stderr


def test_downloaded_model_must_match_the_manifest(tmp_path: Path) -> None:
    payload = b"release model bytes"
    digest = hashlib.sha256(payload).hexdigest()
    pyproject = _write_inputs(
        tmp_path,
        manifest=_manifest(sha256=digest, size=len(payload)),
    )
    model = tmp_path / "model.onnx"
    model.write_bytes(payload)
    arguments = (
        "v0.1.0",
        "--pyproject",
        str(pyproject),
        "--manifest",
        str(tmp_path / "models.json"),
        "--model",
        str(model),
    )
    assert _run(*arguments).returncode == 0

    model.write_bytes(b"tampered")
    result = _run(*arguments)
    assert result.returncode == 1
    assert "sha256" in result.stderr.lower() or "size" in result.stderr.lower()


def test_accepts_a_tag_without_the_v_prefix(tmp_path: Path) -> None:
    pyproject = _write_inputs(tmp_path)
    result = _run(
        "0.1.0",
        "--pyproject",
        str(pyproject),
        "--manifest",
        str(tmp_path / "models.json"),
    )
    assert result.returncode == 0, result.stderr
