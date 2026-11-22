"""End-to-end ``rdv serve`` process test (S12/S14)."""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from rdv.entrypoints.cli.app import app

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()


def _free_local_port() -> int:
    """Pick a free port inside the portfolio's 16000-16999 range."""
    for port in range(16000, 17000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 16000-16999")


def test_serve_help_exposes_the_local_server_options() -> None:
    result = runner.invoke(app, ["serve", "--help"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    # Rich forces colour in CI, which splits "--host" into styled segments.
    plain = ANSI_SGR.sub("", result.output)
    for option in ("--host", "--port", "--no-browser", "--model"):
        assert option in plain


@pytest.mark.enable_socket
def test_serve_process_answers_health_rejects_foreign_hosts_and_cleans_up(
    tmp_path: Path,
) -> None:
    port = _free_local_port()
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    env = dict(os.environ)
    env["TMPDIR"] = str(temp_root)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rdv",
            "serve",
            "--no-browser",
            "--port",
            str(port),
            "--model",
            str(TINY_MODEL),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30.0
        health: httpx.Response | None = None
        while time.monotonic() < deadline and process.poll() is None:
            try:
                candidate = httpx.get(f"{base}/api/health", timeout=0.5)
            except httpx.HTTPError:
                time.sleep(0.1)
                continue
            if candidate.status_code == 200:
                health = candidate
                break
        assert health is not None, _process_output(process)
        assert health.json()["status"] == "ok"

        evil = httpx.get(f"{base}/api/health", headers={"Host": "evil.example"}, timeout=2.0)
        assert evil.status_code == 421
        assert not any(name.lower().startswith("access-control-") for name in evil.headers)
    finally:
        _request_graceful_shutdown(process)
        stdout, stderr = process.communicate(timeout=20)
    assert process.returncode is not None
    assert not list(temp_root.glob("rdv-web-*")), (stdout, stderr)


def _request_graceful_shutdown(process: subprocess.Popen[str]) -> None:
    """Stop the server through uvicorn's signal handlers so its lifespan removes the temp dir.

    On Windows ``terminate`` is ``TerminateProcess``, which skips every handler;
    uvicorn listens for ``SIGBREAK`` there, delivered as ``CTRL_BREAK_EVENT`` to the
    new process group created at spawn.
    """
    if sys.platform == "win32":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGTERM)


def _process_output(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        return "server still running but /api/health never answered"
    stdout, stderr = process.communicate(timeout=5)
    return f"exit={process.returncode}\nstdout={stdout}\nstderr={stderr}"
