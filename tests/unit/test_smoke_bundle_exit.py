"""The bundle smoke accepts Windows' Ctrl+C exit status as a clean shutdown (S15)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "smoke_bundle.py"

STATUS_CONTROL_C_EXIT = 0xC000013A


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_is_a_clean_exit_on_every_platform() -> None:
    smoke = _load()
    assert smoke.is_clean_exit(0, "linux")
    assert smoke.is_clean_exit(0, "win32")


def test_windows_control_c_status_is_a_clean_exit_on_windows() -> None:
    smoke = _load()
    assert smoke.is_clean_exit(STATUS_CONTROL_C_EXIT, "win32")


def test_windows_control_c_status_is_not_clean_elsewhere() -> None:
    smoke = _load()
    assert not smoke.is_clean_exit(STATUS_CONTROL_C_EXIT, "linux")


def test_other_non_zero_codes_still_fail() -> None:
    smoke = _load()
    assert not smoke.is_clean_exit(1, "win32")
    assert not smoke.is_clean_exit(1, "linux")
    assert not smoke.is_clean_exit(0xC0000005, "win32")
