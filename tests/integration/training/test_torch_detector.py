"""Torch checkpoint detector matches the ONNX export on the same weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.adapters.inference.torch_detector import TorchDetector
from rdv.adapters.training.export_onnx import export_onnx_model
from rdv.errors import ModelError
from tests.support.tiny_model import save_tiny_checkpoint

pytestmark = pytest.mark.train


def test_torch_detector_reports_and_decodes_like_the_onnx_graph(tmp_path: Path) -> None:
    checkpoint, digest = save_tiny_checkpoint(tmp_path / "tiny")
    torch_detector = TorchDetector(
        checkpoint,
        device="cpu",
        score_threshold=0.0,
        model_id="tiny",
    )
    assert torch_detector.info.device == "cpu"
    assert torch_detector.info.id == "tiny"
    assert torch_detector.info.sha256 == digest

    from transformers import RTDetrV2ForObjectDetection

    model = RTDetrV2ForObjectDetection.from_pretrained(str(checkpoint))
    onnx_path = tmp_path / "model.onnx"
    export_onnx_model(model, onnx_path)
    onnx_detector = OnnxDetector(onnx_path, device="cpu", score_threshold=0.0, model_id="tiny")

    image = np.full((64, 64, 3), 80, dtype=np.uint8)
    torch_detections = torch_detector.detect([image])[0]
    onnx_detections = onnx_detector.detect([image])[0]

    assert len(torch_detections) == len(onnx_detections)
    for left, right in zip(torch_detections, onnx_detections, strict=True):
        assert left.damage_class is right.damage_class
        assert left.score == pytest.approx(right.score, abs=1e-4)
        assert left.bbox.to_xyxy() == pytest.approx(right.bbox.to_xyxy(), abs=1e-3)


def test_torch_detector_rejects_a_non_checkpoint_path(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="checkpoint directory") as excinfo:
        TorchDetector(tmp_path / "not-a-checkpoint")

    assert excinfo.value.exit_code == 4
