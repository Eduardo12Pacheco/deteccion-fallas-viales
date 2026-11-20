"""The honest-metrics renderer (D17/S10): PASS, BELOW TARGET and stale blocks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_readme_metrics.py"


def _metrics(*, map50: float = 0.42, map50_95: float = 0.21, class_ap: float = 0.31) -> dict:
    return {
        "schema": "rdv.metrics/v1",
        "metrics": {
            "map_50": map50,
            "map_50_95": map50_95,
            "per_class": {
                code: {"num_gt": 10, "ap_50": class_ap, "ap_50_95": class_ap / 2}
                for code in ("D00", "D10", "D20", "D40")
            },
        },
        "ood": {"map_50": 0.18, "map_50_95": 0.07},
    }


def _benchmark() -> dict:
    return {
        "schema": "rdv.benchmark/v1",
        "cpu": {"p50_ms": 321.5, "p95_ms": 355.0},
        "gpu": {"p50_ms": 42.0, "p95_ms": 48.0},
    }


def _reports(tmp_path: Path, metrics: dict) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (reports / "benchmark.json").write_text(json.dumps(_benchmark()), encoding="utf-8")
    return reports


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_status_passes_at_the_minimum_bar(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _metrics())

    result = _run("--reports", str(reports), "--status")

    assert result.returncode == 0, result.stderr
    assert "PASS (>= min bar)" in result.stdout
    assert "mAP@0.5=0.4200" in result.stdout


def test_status_flags_a_below_target_run(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _metrics(map50=0.2, class_ap=0.05))

    result = _run("--reports", str(reports), "--status")

    assert result.returncode == 1
    assert "BELOW TARGET" in result.stdout
    assert "mAP@0.5 0.2000 < 0.35" in result.stdout
    assert "D00 AP@0.5 0.0500 < 0.10" in result.stdout


def test_renders_and_checks_the_readme_block(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _metrics())
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n\n<!-- metrics:start -->\nold\n<!-- metrics:end -->\n", "utf-8")

    rendered = _run("--reports", str(reports), "--readme", str(readme))
    checked = _run("--reports", str(reports), "--readme", str(readme), "--check")

    assert rendered.returncode == 0, rendered.stderr
    assert checked.returncode == 0, checked.stderr
    text = readme.read_text(encoding="utf-8")
    assert "| mAP@0.5 | 0.4200 |" in text
    assert "| AP@0.5 D40 | 0.3100 |" in text
    assert "| CPU p50 (ms/image) | 321.5 |" in text
    assert "| GPU p50 (ms/image) | 42.0 |" in text
    assert "| OOD Norway mAP@0.5 | 0.1800 |" in text
    assert "PASS (>= min bar)" in text
    assert "old" not in text


def test_check_fails_when_the_block_is_stale(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _metrics())
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n\n<!-- metrics:start -->\nstale\n<!-- metrics:end -->\n", "utf-8")

    result = _run("--reports", str(reports), "--readme", str(readme), "--check")

    assert result.returncode == 1
    assert "stale" in result.stderr
