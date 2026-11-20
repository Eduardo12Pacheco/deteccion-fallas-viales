"""Shared console helpers for the CLI entrypoints.

Every command body is wrapped in :func:`run`, which turns expected
``RdvError`` failures into the documented exit code and a one-line message,
and :func:`configure`, which installs structured logging once per process.
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from rdv.application.events import ProgressEvent
from rdv.config.logging import configure_logging
from rdv.config.settings import Settings
from rdv.errors import RdvError


def configure(settings: Settings) -> None:
    """Install structured logging using the validated settings."""
    configure_logging(level=settings.log_level, log_format=settings.log_format)


def run(action: Callable[[], None]) -> None:
    """Execute a command body, mapping expected errors to exit codes."""
    try:
        action()
    except RdvError as exc:
        typer.echo(f"error: {exc.message}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc


def format_mib(value: int) -> str:
    """Format a byte count as whole MiB for progress output."""
    return f"{value / (1024 * 1024):.0f} MiB"


class ConsoleProgressSink:
    """ProgressSink that prints one line per event (CLI Observer)."""

    def emit(self, event: ProgressEvent) -> None:
        """Print the event in a compact human-readable form."""
        counter = (
            f"{event.current}/{event.total}" if event.total is not None else str(event.current)
        )
        suffix = f" {event.message}" if event.message else ""
        typer.echo(f"{event.stage}: {counter}{suffix}")
