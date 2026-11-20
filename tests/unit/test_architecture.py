"""Architecture is enforced, not suggested."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = REPO_ROOT / "src" / "rdv" / "domain"
ALLOWED_DOMAIN_ROOTS = set(sys.stdlib_module_names) | {"numpy", "rdv"}


def test_import_linter_contracts_hold() -> None:
    executable = shutil.which("lint-imports")
    if executable is None:
        pytest.fail("lint-imports is not installed; run 'uv sync' first")

    result = subprocess.run(
        [executable],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_domain_imports_only_stdlib_and_numpy() -> None:
    violations: list[str] = []
    for path in sorted(DOMAIN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            violations.extend(
                f"{path.relative_to(REPO_ROOT)} imports {root}"
                for root in roots
                if root not in ALLOWED_DOMAIN_ROOTS
            )

    assert not violations, "Domain layer must stay pure: " + ", ".join(violations)
