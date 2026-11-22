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


SRC_ROOT = REPO_ROOT / "src" / "rdv"
TORCH_ROOTS = frozenset({"torch", "torchvision", "transformers"})
"""Packages that only the training adapter may load (D9, ADR 0004)."""

TORCH_ALLOWED_FILES = frozenset(
    {
        # CUDA probe and optional GPU detector factory, both loaded through importlib.
        "composition.py",
        # Torch backend; it imports the stack through importlib at construction time.
        "adapters/inference/torch_detector.py",
    }
)
TORCH_ALLOWED_PREFIXES = ("adapters/training/",)


def _literal_dynamic_import(node: ast.Call) -> str | None:
    """Return the literal name of ``importlib.import_module("x")`` or ``__import__("x")``."""
    func = node.func
    is_dynamic = (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    ) or (isinstance(func, ast.Name) and func.id in {"import_module", "__import__"})
    if not is_dynamic or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def torch_roots_referenced(source: str, filename: str = "<source>") -> set[str]:
    """Return the torch-family packages a module imports statically or by literal name.

    Dynamic imports whose name is not a string literal cannot be resolved and are skipped.
    import-linter does not see importlib calls, so this AST scan covers them.
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Call):
            literal = _literal_dynamic_import(node)
            if literal is not None:
                names.append(literal)
    return {name.split(".")[0] for name in names} & TORCH_ROOTS


def torch_violations(src_root: Path = SRC_ROOT) -> list[str]:
    """List ``module references package`` entries outside the allowed torch modules."""
    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        relative = path.relative_to(src_root).as_posix()
        if relative in TORCH_ALLOWED_FILES or relative.startswith(TORCH_ALLOWED_PREFIXES):
            continue
        roots = torch_roots_referenced(path.read_text(encoding="utf-8"), relative)
        violations.extend(f"{relative} references {root}" for root in sorted(roots))
    return violations


def test_torch_stack_is_loaded_only_by_allowed_modules() -> None:
    violations = torch_violations()
    assert not violations, "Torch stack referenced outside its allowed modules: " + ", ".join(
        violations
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import torch\n", {"torch"}),
        ("from transformers.models.rt_detr import X\n", {"transformers"}),
        ("import importlib\nimportlib.import_module('torchvision.ops')\n", {"torchvision"}),
        ("from importlib import import_module\nimport_module('transformers')\n", {"transformers"}),
        ("__import__('torch')\n", {"torch"}),
        ("import importlib\nimportlib.import_module('rdv.adapters.media.video')\n", set()),
        ("doc = 'import torch'\n", set()),
    ],
)
def test_torch_scanner_sees_static_and_literal_dynamic_imports(
    source: str, expected: set[str]
) -> None:
    assert torch_roots_referenced(source) == expected
