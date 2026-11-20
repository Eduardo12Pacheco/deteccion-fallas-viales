"""README acceptance: generated metrics, structure and mirror parity (S19/D17)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "scripts" / "render_readme_metrics.py"
START = "<!-- metrics:start -->"
END = "<!-- metrics:end -->"


def _readme(name: str) -> Path:
    path = ROOT / name
    assert path.is_file(), f"{name} must exist"
    return path


def _metrics_block(name: str) -> str:
    text = _readme(name).read_text(encoding="utf-8")
    match = re.search(
        re.escape(START) + r"(?P<body>.*?)" + re.escape(END),
        text,
        flags=re.DOTALL,
    )
    assert match is not None, f"{name} has no metrics markers"
    return match.group("body").strip()


def test_both_readmes_have_the_same_generated_metrics_block() -> None:
    english = _metrics_block("README.md")
    spanish = _metrics_block("README.es.md")

    assert english == spanish, "README.md and README.es.md metrics must be generated together"
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_readme_stays_within_the_length_budget() -> None:
    for name in ("README.md", "README.es.md"):
        lines = _readme(name).read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 150, f"{name} has {len(lines)} lines (S19 limit is 150)"


def test_readmes_carry_the_required_sections() -> None:
    common = (
        "badge.svg",
        "docs/assets/demo.gif",
        "```mermaid",
        "metrics:start",
        "rdv models pull",
        "rdv serve",
    )
    requirements = {
        "README.md": (*common, "## Limitations", "## Licenses", "README.es.md"),
        "README.es.md": (*common, "## Limitaciones", "## Licencias", "README.md"),
    }
    for name, required in requirements.items():
        text = _readme(name).read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{name} is missing {needle!r}"


def test_screenshots_and_demo_exist() -> None:
    assets = ROOT / "docs" / "assets"
    for name in (
        "home-light.png",
        "home-dark.png",
        "image-result.png",
        "video-result.png",
        "mobile.png",
    ):
        screenshot = assets / "screenshots" / name
        assert screenshot.is_file(), f"missing screenshot {screenshot}"
        assert screenshot.stat().st_size > 10_000, f"{screenshot} looks empty"
    demo = assets / "demo.gif"
    assert demo.is_file() and demo.stat().st_size <= 8 * 1024 * 1024
