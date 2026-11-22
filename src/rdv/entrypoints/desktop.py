"""Desktop launcher for the portable PyInstaller bundle (S15, decision D12).

The bundle ships the same local web app as ``rdv serve`` but has no CLI
arguments to remember: it picks a free port in the shared 16000-16999 range,
starts uvicorn on a daemon thread, waits for ``/api/health``, opens the default
browser and shuts down cleanly on Ctrl+C. Everything around the server is
injectable so the unit tests never bind a real socket or open a browser.
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import httpx
import structlog

from rdv.config.logging import configure_logging
from rdv.config.settings import Settings
from rdv.errors import RdvError

logger = structlog.get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
"""Loopback-only bind address (S14)."""

DEFAULT_PORT = 16080
"""First port to try; the whole shared range is 16000-16999 (amendment L5)."""

PORT_ATTEMPTS = 50
"""How many consecutive ports are probed before giving up."""

HEALTH_TIMEOUT_S = 60.0
"""Maximum wait for the bundled server to answer ``/api/health``."""

HEALTH_POLL_S = 0.1
"""Delay between health probes."""

STOP_TIMEOUT_S = 10.0
"""Maximum wait for the uvicorn thread when shutting down."""


class ServerLike(Protocol):
    """The subset of ``uvicorn.Server`` the launcher controls."""

    should_exit: bool

    def run(self) -> None:
        """Serve until ``should_exit`` is set."""


class RunnerLike(Protocol):
    """A server runner that can be waited on without relying on ``join``.

    ``threading.Thread.join`` (and ``is_alive``) can be left in a bogus state
    when a SIGINT raises ``KeyboardInterrupt`` inside ``join`` on CPython, so
    the launcher waits on a plain ``threading.Event`` instead.
    """

    def start(self) -> None:
        """Start the server thread."""

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the server stops or ``timeout`` expires (``True`` if stopped)."""


class ServerRunner:
    """Thread wrapper that signals completion through an ``Event``."""

    def __init__(self, server: ServerLike) -> None:
        self._server = server
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rdv-desktop-server", daemon=True)

    def _run(self) -> None:
        try:
            self._server.run()
        finally:
            self._done.set()

    def start(self) -> None:
        """Start the server thread."""
        self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the server thread to finish."""
        return self._done.wait(timeout)


@dataclass(frozen=True)
class DesktopOptions:
    """Command-line options passed to the bundled executable."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    no_browser: bool = False
    model: Path | None = None
    health_timeout_s: float = HEALTH_TIMEOUT_S


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return True
    return False


def find_free_port(
    preferred: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    *,
    attempts: int = PORT_ATTEMPTS,
) -> int:
    """Return the first bindable port at or after ``preferred``.

    Raises:
        RdvError: no port in the probed window is free, or the requested port
            cannot be valid.
    """
    if not 1 <= preferred <= 65535:
        raise RdvError(f"Invalid port {preferred}: must be between 1 and 65535")
    last = min(65535, preferred + max(1, attempts) - 1)
    for port in range(preferred, last + 1):
        if not _port_in_use(host, port):
            return port
    raise RdvError(f"No free port found between {preferred} and {last} on {host}")


def wait_for_health(
    url: str,
    *,
    timeout_s: float = HEALTH_TIMEOUT_S,
    interval_s: float = HEALTH_POLL_S,
    probe: Callable[[str], bool],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll ``url`` until ``probe`` succeeds or the deadline passes."""
    deadline = clock() + timeout_s
    while True:
        if probe(url):
            return True
        if clock() >= deadline:
            return False
        sleep(interval_s)


def http_health_probe(url: str) -> bool:
    """Return ``True`` for a healthy ``/api/health`` response."""
    try:
        response = httpx.get(url, timeout=1.0, trust_env=False)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _package_root() -> Path:
    """Directory of the packaged ``rdv`` module (source tree or bundle)."""
    return Path(str(resources.files("rdv")))


def bundled_model_path(*, root: Path | None = None) -> Path | None:
    """Return the single ONNX file embedded by ``build_desktop.py``, if any.

    A development checkout has no ``assets/models`` directory, so the launcher
    falls back to the manifest id (``RDV_MODEL`` still wins). The bundle always
    embeds exactly one model.
    """
    models_dir = (root if root is not None else _package_root()) / "assets" / "models"
    try:
        candidates = sorted(
            item for item in models_dir.iterdir() if item.is_file() and item.suffix == ".onnx"
        )
    except OSError:
        return None
    return candidates[0] if len(candidates) == 1 else None


def build_app(settings: Settings, model_path: Path | None) -> Any:
    """Build the FastAPI app with the bundled (or requested) detector."""
    from rdv import composition
    from rdv.entrypoints.web.app import create_app

    detector = composition.build_detector(
        settings,
        model=str(model_path) if model_path is not None else None,
    )
    return create_app(settings, lambda: detector)


def _uvicorn_server(app: Any, host: str, port: int, log_level: str) -> ServerLike:
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=False,
    )
    return uvicorn.Server(config)


def _server_thread(server: ServerLike) -> RunnerLike:
    return ServerRunner(server)


def serve_until_stopped(
    server: ServerLike,
    runner: RunnerLike,
    *,
    poll_s: float = 0.2,
) -> None:
    """Wait for the server thread, turning Ctrl+C into a graceful shutdown."""
    try:
        while not runner.wait(poll_s):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        runner.wait(timeout=STOP_TIMEOUT_S)


def run_desktop(
    options: DesktopOptions,
    *,
    port_selector: Callable[[int, str], int] = find_free_port,
    app_factory: Callable[[Settings, Path | None], Any] = build_app,
    server_factory: Callable[[Any, str, int, str], ServerLike] = _uvicorn_server,
    runner_factory: Callable[[ServerLike], RunnerLike] = _server_thread,
    open_browser: Callable[[str], Any] = webbrowser.open,
    probe: Callable[[str], bool] = http_health_probe,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Start the bundled app, wait for it, open the browser and block.

    Returns:
        Process exit code (0 after a clean Ctrl+C or server shutdown).

    Raises:
        RdvError: the server never became healthy or the settings are invalid.
    """
    port = port_selector(options.port, options.host)
    model_path = options.model if options.model is not None else bundled_model_path()
    overrides: dict[str, object] = {"host": options.host, "port": port}
    if model_path is not None:
        overrides["model"] = str(model_path)
    settings = Settings.from_env(**overrides)
    configure_logging(level=settings.log_level, log_format=settings.log_format)

    application = app_factory(settings, model_path)
    server = server_factory(application, options.host, port, settings.log_level)
    runner = runner_factory(server)
    runner.start()

    url = f"http://{options.host}:{port}/"
    healthy = wait_for_health(
        f"{url}api/health",
        timeout_s=options.health_timeout_s,
        probe=probe,
    )
    if not healthy:
        server.should_exit = True
        runner.wait(timeout=STOP_TIMEOUT_S)
        raise RdvError(
            f"The local app did not start at {url} within {options.health_timeout_s:.0f} s."
        )

    print_fn(f"Road Damage Vision is running at {url}")
    print_fn("Press Ctrl+C to stop.")
    if not options.no_browser:
        open_browser(url)
    serve_until_stopped(server, runner)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> DesktopOptions:
    """Parse the arguments accepted by the bundled executable."""
    parser = argparse.ArgumentParser(
        prog="RoadDamageVision",
        description="Start the Road Damage Vision local app and open it in the browser.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"First port to try, or the next free one (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="ONNX model to use instead of the embedded one.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the default browser (for smoke tests and servers).",
    )
    namespace = parser.parse_args(argv)
    if not 1 <= namespace.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return DesktopOptions(
        host=namespace.host,
        port=namespace.port,
        no_browser=namespace.no_browser,
        model=namespace.model,
    )


def install_ctrl_break_handler(
    *,
    platform: str = sys.platform,
    signal_module: Any = signal,
) -> None:
    """Make Windows Ctrl+Break raise ``KeyboardInterrupt`` like Ctrl+C does.

    Python leaves ``SIGBREAK`` at the OS default on Windows, which kills the
    process before the app's lifespan can delete its temporary directory.
    """
    if platform == "win32":
        signal_module.signal(signal_module.SIGBREAK, signal_module.default_int_handler)


def main(argv: Sequence[str] | None = None) -> int:
    """Bundled executable entry point."""
    options = parse_args(argv)
    install_ctrl_break_handler()
    try:
        return run_desktop(options)
    except RdvError as exc:
        logger.error("desktop_launch_failed", error=exc.message)
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover - defensive, handled in serve
        return 0


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DesktopOptions",
    "build_app",
    "bundled_model_path",
    "find_free_port",
    "http_health_probe",
    "main",
    "parse_args",
    "run_desktop",
    "serve_until_stopped",
    "wait_for_health",
]
