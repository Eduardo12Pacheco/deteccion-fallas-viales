"""Structured logging: console and JSON renderers with a stable schema."""

from __future__ import annotations

import io
import json

import pytest
import structlog

from rdv.config.logging import configure_logging
from rdv.errors import ConfigError


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    yield
    structlog.reset_defaults()


def test_json_line_is_machine_parseable_and_carries_the_contract_fields() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="json", stream=stream)

    structlog.get_logger("test").info("dataset.loaded", images=3)

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "dataset.loaded"
    assert payload["level"] == "info"
    assert payload["timestamp"]
    assert payload["images"] == 3


def test_json_logger_respects_the_configured_level() -> None:
    stream = io.StringIO()
    configure_logging(level="WARNING", log_format="json", stream=stream)

    logger = structlog.get_logger("test")
    logger.info("hidden")
    logger.warning("shown")

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [payload["event"] for payload in payloads] == ["shown"]
    assert payloads[0]["level"] == "warning"


def test_console_logger_renders_event_and_level() -> None:
    stream = io.StringIO()
    configure_logging(level="DEBUG", log_format="console", stream=stream)

    structlog.get_logger("test").debug("repository.ready", country="Japan")

    output = stream.getvalue()
    assert "repository.ready" in output
    assert "debug" in output.lower()
    assert "Japan" in output


def test_invalid_level_raises_config_error() -> None:
    with pytest.raises(ConfigError) as raised:
        configure_logging(level="LOUD", log_format="json")

    assert raised.value.exit_code == 6
