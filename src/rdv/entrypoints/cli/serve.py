"""``rdv serve``: run the local web app with uvicorn (S12/S14)."""

from __future__ import annotations

import threading
import webbrowser
from typing import Annotated

import typer
import uvicorn

from rdv import composition
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import configure, run
from rdv.entrypoints.web.app import create_app


def _browser_url(host: str, port: int) -> str:
    visible = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    if ":" in visible and not visible.startswith("["):
        visible = f"[{visible}]"
    return f"http://{visible}:{port}/"


def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Bind address (default from RDV_HOST, 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65535, help="Bind port (default 16080)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Manifest model id or path to a local ONNX file."),
    ] = None,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Do not open the default browser."),
    ] = False,
) -> None:
    """Start the local web app and open it in the browser."""

    def action() -> None:
        overrides: dict[str, object] = {}
        if host is not None:
            overrides["host"] = host
        if port is not None:
            overrides["port"] = port
        settings = Settings.from_env(**overrides)
        configure(settings)
        detector = composition.build_detector(settings, model=model)
        application = create_app(settings, lambda: detector)
        url = _browser_url(settings.host, settings.port)
        typer.echo(f"Serving on {url} (press Ctrl+C to stop)")
        if not no_browser:
            timer = threading.Timer(0.6, webbrowser.open, args=(url,))
            timer.daemon = True
            timer.start()
        uvicorn.run(
            application,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )

    run(action)
