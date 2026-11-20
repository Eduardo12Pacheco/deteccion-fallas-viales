"""Train/export use cases delegate through their ports without torch."""

from __future__ import annotations

from pathlib import Path

import pytest

from rdv.application.export_model import (
    ExportRequest,
    ExportResult,
    export_model,
)
from rdv.application.train_model import TrainingRequest, TrainingResult, train_model


class _Trainer:
    def __init__(self) -> None:
        self.requests: list[TrainingRequest] = []

    def train(self, request: TrainingRequest) -> TrainingResult:
        self.requests.append(request)
        return TrainingResult(run_dir=request.run_dir, epochs_completed=2, steps=7, last_loss=0.5)


class _Exporter:
    def __init__(self) -> None:
        self.requests: list[ExportRequest] = []

    def export(self, request: ExportRequest) -> ExportResult:
        self.requests.append(request)
        return ExportResult(
            output=request.output,
            opset=18,
            logits_max_abs_diff=0.0,
            boxes_max_abs_diff=0.0,
            num_queries=300,
        )


def test_train_use_case_passes_the_request_through(tmp_path: Path) -> None:
    trainer = _Trainer()
    request = TrainingRequest(run_dir=tmp_path / "run", resume="none")

    result = train_model(trainer, request)

    assert result.epochs_completed == 2
    assert trainer.requests == [request]


def test_export_use_case_passes_the_request_through(tmp_path: Path) -> None:
    exporter = _Exporter()
    request = ExportRequest(checkpoint=tmp_path / "best", output=tmp_path / "model.onnx")

    result = export_model(exporter, request)

    assert result.opset == 18
    assert exporter.requests == [request]


def test_requests_validate_their_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        TrainingRequest(run_dir=tmp_path, max_steps=0)
    with pytest.raises(ValueError, match="resume"):
        TrainingRequest(run_dir=tmp_path, resume=" ")
    with pytest.raises(ValueError, match="input_size"):
        ExportRequest(checkpoint=tmp_path, output=tmp_path / "x.onnx", input_size=0)
