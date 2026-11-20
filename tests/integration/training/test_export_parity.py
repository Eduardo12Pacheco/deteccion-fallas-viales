"""ONNX export contract and torch-vs-ORT parity on the tiny model (S4, D8)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from rdv.adapters.training.export_onnx import (
    CONTRACT_QUERIES,
    OnnxModelExporter,
    export_onnx_model,
    inspect_graph,
    parity_report,
)
from rdv.application.export_model import ExportRequest
from rdv.errors import ModelError
from tests.support.tiny_model import save_tiny_checkpoint

pytestmark = pytest.mark.train


def test_exported_graph_follows_the_s4_contract(tmp_path: Path) -> None:
    checkpoint, _ = save_tiny_checkpoint(tmp_path / "tiny")
    from transformers import RTDetrV2ForObjectDetection

    model = RTDetrV2ForObjectDetection.from_pretrained(str(checkpoint))
    output = tmp_path / "model.onnx"

    graph = export_onnx_model(model, output)

    assert output.is_file()
    assert graph.opset >= 17
    assert [name for name, _ in graph.inputs] == ["pixel_values"]
    assert [name for name, _ in graph.outputs] == ["logits", "pred_boxes"]
    assert graph.inputs[0][1][1:] == (3, 640, 640)
    assert graph.outputs[0][1][0] == "batch"
    assert graph.outputs[0][1][1:] == (CONTRACT_QUERIES, 4)
    assert graph.outputs[1][1][1:] == (CONTRACT_QUERIES, 4)


def test_parity_stays_within_the_s4_limit(tmp_path: Path) -> None:
    checkpoint, _ = save_tiny_checkpoint(tmp_path / "tiny")
    from transformers import RTDetrV2ForObjectDetection

    model = RTDetrV2ForObjectDetection.from_pretrained(str(checkpoint))
    output = tmp_path / "model.onnx"
    export_onnx_model(model, output)

    logits_diff, boxes_diff = parity_report(model, output)

    assert logits_diff <= 1e-3
    assert boxes_diff <= 1e-3


def test_exporter_implements_the_application_port(tmp_path: Path) -> None:
    checkpoint, _ = save_tiny_checkpoint(tmp_path / "tiny")
    exporter = OnnxModelExporter()

    result = exporter.export(ExportRequest(checkpoint=checkpoint, output=tmp_path / "model.onnx"))

    assert result.output.is_file()
    assert result.opset >= 17
    assert result.num_queries == CONTRACT_QUERIES
    assert result.logits_max_abs_diff <= 1e-3
    assert result.boxes_max_abs_diff <= 1e-3
    assert inspect_graph(result.output).outputs[0][1][1] == CONTRACT_QUERIES


def test_missing_checkpoint_is_a_model_error(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="Checkpoint directory not found") as excinfo:
        OnnxModelExporter().export(
            ExportRequest(checkpoint=tmp_path / "missing", output=tmp_path / "model.onnx")
        )

    assert excinfo.value.exit_code == 4
