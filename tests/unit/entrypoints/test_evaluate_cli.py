"""Unit tests for the ``rdv evaluate`` command wiring (S7/T8)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import rdv.entrypoints.cli.evaluate as evaluate_module
from rdv.entrypoints.cli.app import app

runner = CliRunner()


class FakeOutcome:
    def __init__(self) -> None:
        self.images_evaluated = 2
        self.metrics = SimpleNamespace(mAP50=0.5, mAP50_95=0.25)
        self.operating_threshold = 0.5
        self.operating_source = "val"
        self.artifacts = SimpleNamespace(report_html=Path("/tmp/report.html"))


def _patch(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    dataset = object()
    selection_calls: list[dict[str, Any]] = []

    def open_repository(root: Path, **kwargs: Any) -> object:
        selection_calls.append(kwargs)
        calls.setdefault("repositories", []).append(kwargs)
        return dataset

    def fake_evaluate(*args: Any, **kwargs: Any) -> FakeOutcome:
        calls["evaluate"] = {"args": args, "kwargs": kwargs}
        return FakeOutcome()

    monkeypatch.setattr(evaluate_module.composition, "discover_countries", lambda root: ("Japan",))
    monkeypatch.setattr(evaluate_module.composition, "open_evaluation_repository", open_repository)
    monkeypatch.setattr(evaluate_module.composition, "build_detector", lambda *a, **k: object())
    monkeypatch.setattr(
        evaluate_module.composition, "build_evaluation_report_writer", lambda: object()
    )
    monkeypatch.setattr(
        evaluate_module.composition, "open_ood_repository", lambda root: ("ood", root)
    )
    monkeypatch.setattr(evaluate_module, "evaluate_model", fake_evaluate)


def test_auto_threshold_opens_the_validation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    _patch(monkeypatch, calls)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--data",
            "tests/fixtures/rdd2022-mini",
            "--split",
            "test",
            "--model",
            "tests/fixtures/models/tiny-detector.onnx",
            "--out",
            "/tmp/eval-out",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [entry["split"] for entry in calls["repositories"]] == ["test", "val"]
    kwargs = calls["evaluate"]["kwargs"]
    assert kwargs["selection_dataset"] is not None
    assert kwargs["operating_threshold"] is None
    assert kwargs["split"] == "test"
    assert kwargs["score_threshold"] == pytest.approx(0.001)
    assert "mAP@0.5=0.5000" in result.output


def test_explicit_threshold_does_not_open_a_validation_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    _patch(monkeypatch, calls)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--data",
            "tests/fixtures/rdd2022-mini",
            "--split",
            "test",
            "--op-threshold",
            "0.4",
            "--out",
            "/tmp/eval-out",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [entry["split"] for entry in calls["repositories"]] == ["test"]
    assert calls["evaluate"]["kwargs"]["selection_dataset"] is None
    assert calls["evaluate"]["kwargs"]["operating_threshold"] == pytest.approx(0.4)


def test_val_split_reuses_the_same_repository_for_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    _patch(monkeypatch, calls)

    result = runner.invoke(
        app,
        ["evaluate", "--data", "tests/fixtures/rdd2022-mini", "--split", "val"],
    )

    assert result.exit_code == 0, result.output
    assert [entry["split"] for entry in calls["repositories"]] == ["val"]
    kwargs = calls["evaluate"]["kwargs"]
    assert kwargs["selection_dataset"] is calls["evaluate"]["args"][0]


def test_ood_repository_is_opened_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    _patch(monkeypatch, calls)

    result = runner.invoke(
        app,
        ["evaluate", "--data", "tests/fixtures/rdd2022-mini", "--ood", "/tmp/norway"],
    )

    assert result.exit_code == 0, result.output
    assert calls["evaluate"]["kwargs"]["ood_dataset"] == ("ood", Path("/tmp/norway"))


def test_invalid_threshold_and_split_exit_six(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch(monkeypatch, calls)

    invalid_threshold = runner.invoke(
        app,
        ["evaluate", "--data", "tests/fixtures/rdd2022-mini", "--op-threshold", "x"],
    )
    invalid_split = runner.invoke(
        app,
        ["evaluate", "--data", "tests/fixtures/rdd2022-mini", "--split", "nope"],
    )

    assert invalid_threshold.exit_code == 6
    assert "op-threshold" in invalid_threshold.output
    assert invalid_split.exit_code == 6
    assert "--split" in invalid_split.output
    assert "evaluate" not in calls
