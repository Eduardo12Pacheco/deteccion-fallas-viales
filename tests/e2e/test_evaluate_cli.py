"""End-to-end ``rdv evaluate`` acceptance tests (S7/T8)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI = REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini"
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden" / "eval-mini" / "metrics.json"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rdv", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _evaluate(out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run(
        "evaluate",
        "--data",
        str(MINI),
        "--splits",
        str(MINI / "splits.json"),
        "--split",
        "all",
        "--model",
        str(TINY_MODEL),
        "--out",
        str(out),
        *extra,
    )


def test_evaluate_writes_every_s7_file_and_matches_the_golden(tmp_path: Path) -> None:
    out = tmp_path / "eval"

    result = _evaluate(out)

    assert result.returncode == 0, result.stderr
    required = {
        "metrics.json",
        "report.md",
        "report.html",
        "pr_curves.png",
        "confusion_matrix.png",
        "confusion_matrix.csv",
        "error_analysis.json",
        "pr_curve_D00.csv",
        "pr_curve_D10.csv",
        "pr_curve_D20.csv",
        "pr_curve_D40.csv",
        "gallery/index.html",
    }
    for name in required:
        assert (out / name).is_file(), name

    actual = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert actual == expected
    assert actual["schema"] == "rdv.metrics/v1"
    assert actual["dataset"]["images"] == 16
    assert actual["params"]["operating_source"] == "val"

    errors = json.loads((out / "error_analysis.json").read_text(encoding="utf-8"))
    assert errors["schema"] == "rdv.error-analysis/v1"
    assert errors["counts"] == actual["errors"]["counts"]

    html = (out / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "data:image/jpeg;base64," in html
    assert "<script>" not in html


def test_gallery_index_references_only_existing_files(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    assert _evaluate(out).returncode == 0

    gallery = out / "gallery"
    html = (gallery / "index.html").read_text(encoding="utf-8")
    references = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert references
    for reference in references:
        assert (gallery / reference).is_file(), reference
    categories = {path.parent.name for path in gallery.glob("*/*.jpg")}
    assert categories
    for category in categories:
        assert f"<h2>{category}</h2>" in html


def test_evaluate_is_byte_identical_on_a_second_run(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert _evaluate(first).returncode == 0
    assert _evaluate(second).returncode == 0

    assert (first / "metrics.json").read_bytes() == (second / "metrics.json").read_bytes()
    assert (first / "report.md").read_bytes() == (second / "report.md").read_bytes()
    assert (first / "error_analysis.json").read_bytes() == (
        second / "error_analysis.json"
    ).read_bytes()


def test_explicit_operating_threshold_is_recorded(tmp_path: Path) -> None:
    out = tmp_path / "eval"

    result = _evaluate(out, "--op-threshold", "0.5")

    assert result.returncode == 0, result.stderr
    document = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert document["params"]["operating_source"] == "explicit"
    assert document["params"]["operating_threshold"] == 0.5
    assert document["operating_point"]["threshold"] == 0.5


def test_invalid_operating_threshold_exits_six(tmp_path: Path) -> None:
    result = _evaluate(tmp_path / "eval", "--op-threshold", "rubbish")

    assert result.returncode == 6
    assert "op-threshold" in result.stderr


def test_invalid_split_exits_six() -> None:
    result = _run(
        "evaluate",
        "--data",
        str(MINI),
        "--split",
        "nope",
        "--model",
        str(TINY_MODEL),
        "--out",
        "/tmp/rdv-eval-invalid-split",
    )

    assert result.returncode == 6
    assert "--split" in result.stderr
