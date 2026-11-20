"""The committed tiny ONNX model is reproducible and contract-shaped."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_fixture_model.py"
COMMITTED = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
)


def test_regenerating_the_tiny_model_is_byte_identical(tmp_path: Path) -> None:
    generated = tmp_path / "tiny-detector.onnx"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(generated)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert generated.read_bytes() == COMMITTED.read_bytes()
    assert COMMITTED.stat().st_size < 100 * 1024


def test_tiny_model_is_valid_and_follows_the_model_contract() -> None:
    model = onnx.load(COMMITTED)

    onnx.checker.check_model(model)
    assert model.opset_import[0].version >= 17
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    assert set(inputs) == {"pixel_values"}
    assert set(outputs) == {"logits", "pred_boxes"}


def test_tiny_model_returns_the_contract_shapes_through_ort() -> None:
    session = ort.InferenceSession(str(COMMITTED), providers=["CPUExecutionProvider"])

    logits, boxes = session.run(
        None,
        {"pixel_values": np.zeros((1, 3, 640, 640), dtype=np.float32)},
    )

    assert logits.shape == (1, 300, 4)
    assert boxes.shape == (1, 300, 4)
    assert logits.dtype == "float32"
    assert boxes.dtype == "float32"
