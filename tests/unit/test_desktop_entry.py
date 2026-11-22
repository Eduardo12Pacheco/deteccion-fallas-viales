"""Unit tests for the desktop launcher (T13/S15).

The launcher is deliberately small and injectable: port selection, health
polling, browser opening and the Ctrl+C shutdown path are all seams so these
tests never start a real server or open a real browser.
"""

from __future__ import annotations

import io
import socket
from pathlib import Path

import pytest
import structlog

from rdv.config.logging import configure_logging
from rdv.entrypoints import desktop
from rdv.errors import ModelError, RdvError


class FakeServer:
    """Minimal uvicorn.Server stand-in."""

    def __init__(self) -> None:
        self.should_exit = False
        self.config = None


class FakeRunner:
    """Server runner stand-in driven by a list of ``wait`` answers."""

    def __init__(self, running: list[bool] | None = None, *, interrupted: bool = False) -> None:
        # ``running`` is the answer list for wait(): False = still running.
        self._running = list(running or [False])
        self._interrupted = interrupted
        self.started = False
        self.waits: list[float | None] = []

    def start(self) -> None:
        self.started = True

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        if self._interrupted:
            self._interrupted = False
            raise KeyboardInterrupt
        if len(self._running) > 1:
            return self._running.pop(0)
        return self._running[0] if self._running else True


def _bind_free_port() -> tuple[socket.socket, int]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    return probe, int(probe.getsockname()[1])


@pytest.mark.enable_socket
def test_find_free_port_skips_a_busy_port() -> None:
    busy, busy_port = _bind_free_port()
    try:
        selected = desktop.find_free_port(busy_port, "127.0.0.1", attempts=5)
    finally:
        busy.close()

    assert selected != busy_port
    assert busy_port < selected <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
        check.bind(("127.0.0.1", selected))


@pytest.mark.enable_socket
def test_find_free_port_raises_when_the_range_is_exhausted() -> None:
    busy, busy_port = _bind_free_port()
    try:
        with pytest.raises(RdvError, match="free port"):
            desktop.find_free_port(busy_port, "127.0.0.1", attempts=1)
    finally:
        busy.close()


def test_default_port_stays_inside_the_shared_range() -> None:
    assert 16000 <= desktop.DEFAULT_PORT <= 16999


def test_wait_for_health_returns_as_soon_as_the_probe_succeeds() -> None:
    probes: list[str] = []

    def probe(url: str) -> bool:
        probes.append(url)
        return len(probes) == 2

    assert (
        desktop.wait_for_health(
            "http://127.0.0.1:16080/api/health",
            timeout_s=5.0,
            probe=probe,
            sleep=lambda _: None,
        )
        is True
    )
    assert len(probes) == 2


def test_wait_for_health_times_out_without_sleeping_past_the_deadline() -> None:
    times = iter([0.0, 0.5, 2.0])
    sleeps: list[float] = []

    assert (
        desktop.wait_for_health(
            "http://127.0.0.1:16080/api/health",
            timeout_s=1.0,
            probe=lambda _: False,
            clock=lambda: next(times),
            sleep=sleeps.append,
        )
        is False
    )
    assert sleeps == [desktop.HEALTH_POLL_S]


def test_bundled_model_path_returns_none_without_a_packaged_model(
    tmp_path: Path,
) -> None:
    assert desktop.bundled_model_path(root=tmp_path) is None


def test_bundled_model_path_finds_a_single_packaged_model(tmp_path: Path) -> None:
    models = tmp_path / "assets" / "models"
    models.mkdir(parents=True)
    model = models / "detector.onnx"
    model.write_bytes(b"onnx")

    assert desktop.bundled_model_path(root=tmp_path) == model


def test_no_browser_option_skips_the_browser_and_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    printed: list[str] = []
    server = FakeServer()
    runner = FakeRunner([False, True])

    def make_server(*_args: object, **_kwargs: object) -> FakeServer:
        return server

    def make_runner(_server: FakeServer) -> FakeRunner:
        return runner

    def fake_wait(_url: str, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(desktop, "wait_for_health", fake_wait)
    code = desktop.run_desktop(
        desktop.DesktopOptions(no_browser=True),
        port_selector=lambda _preferred, _host: 16082,
        app_factory=lambda *_args: object(),
        server_factory=make_server,
        runner_factory=make_runner,
        open_browser=opened.append,
        print_fn=printed.append,
    )

    assert code == 0
    assert opened == []
    assert runner.started is True
    assert server.should_exit is True
    assert any("16082" in line for line in printed)


def test_desktop_opens_the_browser_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    server = FakeServer()
    runner = FakeRunner([False, True])
    monkeypatch.setattr(desktop, "wait_for_health", lambda *_args, **_kwargs: True)

    code = desktop.run_desktop(
        desktop.DesktopOptions(no_browser=False),
        port_selector=lambda _preferred, _host: 16083,
        app_factory=lambda *_args: object(),
        server_factory=lambda *_args, **_kwargs: server,
        runner_factory=lambda _server: runner,
        open_browser=opened.append,
        print_fn=lambda _line: None,
    )

    assert code == 0
    assert opened == ["http://127.0.0.1:16083/"]


def test_health_timeout_fails_without_opening_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    server = FakeServer()
    runner = FakeRunner([False, True])
    monkeypatch.setattr(desktop, "wait_for_health", lambda *_args, **_kwargs: False)

    with pytest.raises(RdvError, match="did not start"):
        desktop.run_desktop(
            desktop.DesktopOptions(),
            port_selector=lambda _preferred, _host: 16084,
            app_factory=lambda *_args: object(),
            server_factory=lambda *_args, **_kwargs: server,
            runner_factory=lambda _server: runner,
            open_browser=opened.append,
            print_fn=lambda _line: None,
        )

    assert opened == []
    assert server.should_exit is True


def test_ctrl_c_requests_a_graceful_server_shutdown() -> None:
    server = FakeServer()
    runner = FakeRunner(interrupted=True)

    desktop.serve_until_stopped(server, runner, poll_s=0.01)

    assert server.should_exit is True
    assert runner.waits == [0.01, desktop.STOP_TIMEOUT_S]


def test_parse_args_accepts_the_documented_flags() -> None:
    options = desktop.parse_args(["--no-browser", "--port", "16090", "--host", "127.0.0.1"])
    assert options == desktop.DesktopOptions(
        host="127.0.0.1",
        port=16090,
        no_browser=True,
        model=None,
    )

    with pytest.raises(SystemExit):
        desktop.parse_args(["--port", "0"])


def test_main_logs_a_launch_failure_and_returns_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(options: desktop.DesktopOptions) -> int:
        raise ModelError("the local app did not start")

    stream = io.StringIO()
    configure_logging(level="INFO", log_format="console", stream=stream)
    monkeypatch.setattr(desktop, "run_desktop", fail)
    try:
        exit_code = desktop.main([])
    finally:
        structlog.reset_defaults()

    assert exit_code == ModelError.exit_code
    assert "the local app did not start" in stream.getvalue()
    assert capsys.readouterr().err == ""
