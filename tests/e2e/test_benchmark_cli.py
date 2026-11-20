"""End-to-end ``rdv benchmark`` with the tiny ONNX model (S11)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"


def test_benchmark_writes_the_documented_keys(tmp_path: Path) -> None:
    out = tmp_path / "b.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rdv",
            "benchmark",
            "--model",
            str(TINY_MODEL),
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["schema"] == "rdv.benchmark/v1"
    assert document["cpu"]["p50_ms"] > 0.0
    assert document["cpu"]["p95_ms"] >= document["cpu"]["p50_ms"]
    assert document["hardware"]["cpu"]
    assert document["video"]["frames_per_s"] > 0.0
    assert document["config"]["runs"] == 100
    assert document["config"]["warmup"] == 20


def test_benchmark_in_process_writes_the_report(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from rdv.entrypoints.cli.app import app

    out = tmp_path / "in-process.json"

    result = CliRunner().invoke(
        app,
        ["benchmark", "--model", str(TINY_MODEL), "--out", str(out)],
    )

    assert result.exit_code == 0, result.stderr
    assert "CPU p50" in result.stdout
    assert "Report:" in result.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "rdv.benchmark/v1"
