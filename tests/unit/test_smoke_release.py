"""The post-release smoke compares released detections with pinned expectations (S18)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "smoke_release.py"
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
SAMPLES = REPO_ROOT / "src" / "rdv" / "assets" / "samples"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _expectations(tmp_path: Path, samples: list[dict[str, object]]) -> Path:
    path = tmp_path / "sample_expectations.json"
    path.write_text(
        json.dumps(
            {
                "schema": "rdv.sample-expectations/v1",
                "threshold": 0.01,
                "margin_required": 0.15,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


def _arguments(tmp_path: Path) -> tuple[str, ...]:
    return (
        "--model",
        str(TINY_MODEL),
        "--samples",
        str(SAMPLES),
        "--expectations",
        str(tmp_path / "sample_expectations.json"),
    )


def test_help_exits_zero() -> None:
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "--expectations" in result.stdout


def test_tiny_model_matches_fixture_expectations(tmp_path: Path) -> None:
    # The scripted tiny model always reports D00 and D40 at ~0.99/0.95.
    _expectations(
        tmp_path,
        [
            {"image": "D00-1.jpg", "expected_classes": ["D00", "D40"]},
            {"image": "D40-2.jpg", "expected_classes": ["D40"]},
        ],
    )
    result = _run(*_arguments(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "2/2" in result.stdout


def test_missing_expected_class_exits_one(tmp_path: Path) -> None:
    _expectations(
        tmp_path,
        [{"image": "D00-1.jpg", "expected_classes": ["D00", "D20"]}],
    )
    result = _run(*_arguments(tmp_path))
    assert result.returncode == 1
    assert "D00-1.jpg" in result.stderr
    assert "D20" in result.stderr


def test_scores_below_the_threshold_do_not_count(tmp_path: Path) -> None:
    _expectations(
        tmp_path,
        [{"image": "D00-1.jpg", "expected_classes": ["D00"]}],
    )
    payload = json.loads((tmp_path / "sample_expectations.json").read_text(encoding="utf-8"))
    payload["threshold"] = 0.999
    (tmp_path / "sample_expectations.json").write_text(json.dumps(payload), encoding="utf-8")
    result = _run(*_arguments(tmp_path))
    assert result.returncode == 1
    assert "D00" in result.stderr


def test_unknown_sample_image_exits_one(tmp_path: Path) -> None:
    _expectations(
        tmp_path,
        [{"image": "does-not-exist.jpg", "expected_classes": ["D00"]}],
    )
    result = _run(*_arguments(tmp_path))
    assert result.returncode == 1
    assert "does-not-exist.jpg" in result.stderr


def test_invalid_expectations_schema_exits_one(tmp_path: Path) -> None:
    _expectations(
        tmp_path,
        [{"image": "D00-1.jpg", "expected_classes": ["D00"]}],
    )
    payload = json.loads((tmp_path / "sample_expectations.json").read_text(encoding="utf-8"))
    payload["schema"] = "rdv.sample-expectations/v0"
    (tmp_path / "sample_expectations.json").write_text(json.dumps(payload), encoding="utf-8")
    result = _run(*_arguments(tmp_path))
    assert result.returncode == 1
    assert "schema" in result.stderr
