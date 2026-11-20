"""``rdv doctor`` and ``rdv samples`` command contracts (S1)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rdv.entrypoints.cli.app import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_SAMPLES = REPO_ROOT / "src" / "rdv" / "assets" / "samples"


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "RDV_MODEL_DIR": str(tmp_path / "models"),
        "RDV_DATA_DIR": str(tmp_path / "data"),
    }


def test_doctor_json_reports_environment_and_missing_model(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", "--json"], env=_env(tmp_path))

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "rdv.doctor/v1"
    assert report["rdv_version"] == "0.1.0"
    assert report["python"].startswith("3.12")
    assert "CPUExecutionProvider" in report["providers"]
    assert report["model"]["status"] == "missing"
    assert report["model_dir"] == str(tmp_path / "models")
    assert report["data_dir"] == str(tmp_path / "data")


def test_doctor_text_mode_is_human_readable(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor"], env=_env(tmp_path))

    assert result.exit_code == 0, result.stderr
    assert "Python" in result.stdout
    assert "ONNX Runtime" in result.stdout
    assert "Model:" in result.stdout


def test_doctor_reports_a_ready_local_model(tmp_path: Path) -> None:
    model = tmp_path / "local.onnx"
    model.write_bytes(b"fake")
    env = _env(tmp_path)
    env["RDV_MODEL"] = str(model)

    result = runner.invoke(app, ["doctor", "--json"], env=env)

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["model"]["status"] == "ready"
    assert report["model"]["path"] == str(model)


def test_samples_copies_every_packaged_file(tmp_path: Path) -> None:
    dest = tmp_path / "samples"

    result = runner.invoke(app, ["samples", "--dest", str(dest)])

    assert result.exit_code == 0, result.stderr
    copied = sorted(path.name for path in dest.iterdir())
    expected = sorted(path.name for path in PACKAGED_SAMPLES.iterdir() if path.is_file())
    assert copied == expected
    assert sum(name.endswith(".jpg") for name in copied) == 6
    assert "LICENSE" in copied
    assert "ATTRIBUTION.md" in copied
    for name in copied:
        assert (dest / name).read_bytes() == (PACKAGED_SAMPLES / name).read_bytes()


def test_samples_is_idempotent(tmp_path: Path) -> None:
    dest = tmp_path / "samples"

    first = runner.invoke(app, ["samples", "--dest", str(dest)])
    second = runner.invoke(app, ["samples", "--dest", str(dest)])

    assert first.exit_code == 0 and second.exit_code == 0
    assert len(list(dest.iterdir())) == len(
        [path for path in PACKAGED_SAMPLES.iterdir() if path.is_file()]
    )
