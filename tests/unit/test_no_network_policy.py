"""The default test run must not reach the outside network (S16, D18)."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

ROOT = Path(__file__).resolve().parents[2]
PROJECT_SOCKET_FLAGS = ("--disable-socket", "--allow-unix-socket")


def test_default_pytest_run_disables_sockets() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    addopts_line = next(
        (line for line in pyproject.splitlines() if line.strip().startswith("addopts")),
        "",
    )
    for flag in PROJECT_SOCKET_FLAGS:
        assert flag in addopts_line, f"{flag} must be part of the default pytest addopts"


@pytest.mark.filterwarnings("ignore:A test tried to use socket.:UserWarning")
def test_outbound_connection_is_blocked(pytestconfig: pytest.Config) -> None:
    addopts = pytestconfig.getini("addopts")
    assert "--disable-socket" in addopts
    assert "--allow-hosts" not in addopts
    with (
        pytest.raises(SocketBlockedError),
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
    ):
        sock.settimeout(0.2)
        sock.connect(("1.1.1.1", 80))
