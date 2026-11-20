"""Structured logging with a console renderer for humans and JSON for machines.

The CLI configures logging once at startup. Every log line carries ``event``,
``level`` and ``timestamp``; the JSON renderer keeps that schema stable for
tooling and the web job log.
"""

from __future__ import annotations

import logging
from typing import TextIO

import structlog

from rdv.errors import ConfigError

_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _level_value(level: str) -> int:
    try:
        return _LOG_LEVELS[level.strip().upper()]
    except KeyError as exc:
        expected = ", ".join(_LOG_LEVELS)
        raise ConfigError(f"Invalid log level {level!r}; expected one of {expected}") from exc


def configure_logging(
    *,
    level: str = "INFO",
    log_format: str = "console",
    stream: TextIO | None = None,
) -> None:
    """Configure structlog globally.

    Args:
        level: minimum emitted level (case-insensitive).
        log_format: ``console`` (rendered) or ``json`` (one object per line).
        stream: destination for log lines; defaults to stdout. Tests pass a
            ``StringIO`` to assert the exact output.

    Raises:
        ConfigError: the level or format is not supported.
    """
    numeric_level = _level_value(level)
    if log_format not in {"console", "json"}:
        raise ConfigError(f"Invalid log format {log_format!r}; expected 'console' or 'json'")

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: structlog.typing.Processor
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=False,
    )
