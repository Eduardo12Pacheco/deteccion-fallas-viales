"""Base installs stay torch-free and light (spec S2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_import_does_not_load_the_training_stack() -> None:
    code = (
        "import sys, rdv.entrypoints.cli.app; "
        "heavy = {'torch', 'torchvision', 'transformers', 'scipy'} & set(sys.modules); "
        "assert not heavy, sorted(heavy)"
    )

    result = _python(code)

    assert result.returncode == 0, result.stderr


def test_inference_and_web_imports_do_not_load_matplotlib() -> None:
    code = (
        "import sys, rdv.adapters.inference, rdv.entrypoints.web; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    )

    result = _python(code)

    assert result.returncode == 0, result.stderr
