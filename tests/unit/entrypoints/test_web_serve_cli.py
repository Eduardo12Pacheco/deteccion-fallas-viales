"""Unit tests for the ``rdv serve`` command wiring (S12/S14)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from rdv.entrypoints.cli import serve as serve_module
from rdv.entrypoints.cli.app import app
from rdv.errors import ModelError

runner = CliRunner()


class _ImmediateTimer:
    """Stand-in for ``threading.Timer`` that runs the callback synchronously."""

    def __init__(self, interval: float, function: Any, args: tuple[Any, ...] = ()) -> None:
        self.interval = interval
        self.function = function
        self.args = args
        self.daemon = False

    def start(self) -> None:
        self.function(*self.args)


def test_browser_url_maps_wildcard_and_ipv6_hosts() -> None:
    assert serve_module._browser_url("127.0.0.1", 16080) == "http://127.0.0.1:16080/"
    assert serve_module._browser_url("0.0.0.0", 16080) == "http://127.0.0.1:16080/"
    assert serve_module._browser_url("::", 16080) == "http://127.0.0.1:16080/"
    assert serve_module._browser_url("::1", 16080) == "http://[::1]:16080/"


def test_serve_no_browser_passes_settings_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = object()
    application = object()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(serve_module.composition, "build_detector", lambda *a, **k: detector)
    monkeypatch.setattr(serve_module, "create_app", lambda settings, factory: application)
    monkeypatch.setattr(
        serve_module.uvicorn, "run", lambda app, **kwargs: calls.update({"app": app, **kwargs})
    )
    opened: list[str] = []
    monkeypatch.setattr(serve_module.webbrowser, "open", lambda url: opened.append(url))

    result = runner.invoke(app, ["serve", "--no-browser", "--port", "16123"])

    assert result.exit_code == 0, result.output
    assert calls["app"] is application
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 16123
    assert opened == []


def test_serve_opens_the_browser_unless_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_module.composition, "build_detector", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(serve_module, "create_app", lambda settings, factory: object())
    monkeypatch.setattr(serve_module.uvicorn, "run", lambda app, **kwargs: None)
    opened: list[str] = []
    monkeypatch.setattr(serve_module.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(serve_module.threading, "Timer", _ImmediateTimer)

    result = runner.invoke(app, ["serve", "--port", "16124"])

    assert result.exit_code == 0, result.output
    assert opened == ["http://127.0.0.1:16124/"]


def test_missing_model_exits_four_with_the_pull_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise ModelError("Model not found; run `rdv models pull`")

    monkeypatch.setattr(serve_module.composition, "build_detector", missing)
    monkeypatch.setattr(
        serve_module.uvicorn, "run", lambda *a, **k: pytest.fail("uvicorn must not start")
    )

    result = runner.invoke(app, ["serve", "--no-browser", "--port", "16125"])

    assert result.exit_code == 4
    assert "rdv models pull" in result.output
