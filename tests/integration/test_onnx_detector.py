"""ONNX detector integration against the committed tiny model (S4, D8)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.application.ports import DEFAULT_SCORE_THRESHOLD
from rdv.domain.classes import DamageClass
from rdv.errors import ModelError
from rdv.hashing import sha256_file

TINY_MODEL = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
)


def _image(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _numpy_reference(
    image: np.ndarray,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Independent numpy implementation of the S4 decode contract."""
    session = ort.InferenceSession(str(TINY_MODEL), providers=["CPUExecutionProvider"])
    height, width = image.shape[:2]
    resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    tensor = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
    logits, boxes = session.run(None, {"pixel_values": tensor})
    probabilities = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float64)))
    results: list[tuple[str, float, tuple[float, float, float, float]]] = []
    ranked = np.argsort(-probabilities.reshape(-1), kind="stable")[:100]
    for flat in ranked:
        score = float(probabilities.reshape(-1)[flat])
        if score < threshold:
            continue
        query, class_id = divmod(int(flat), 4)
        cx, cy, w, h = (float(value) for value in boxes[0][query])
        if w <= 0.0 or h <= 0.0:
            continue
        px, py = cx * width, cy * height
        half_w, half_h = w * width / 2.0, h * height / 2.0
        x1 = min(max(px - half_w, 0.0), float(width))
        y1 = min(max(py - half_h, 0.0), float(height))
        x2 = min(max(px + half_w, 0.0), float(width))
        y2 = min(max(py + half_h, 0.0), float(height))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        results.append((DamageClass(class_id).code, score, (x1, y1, x2, y2)))
    return results


def test_info_reports_identity_threshold_and_device() -> None:
    detector = OnnxDetector(TINY_MODEL, device="cpu", providers=["CPUExecutionProvider"])

    info = detector.info
    assert info.id == "tiny-detector"
    assert info.input_size == 640
    assert info.score_threshold == DEFAULT_SCORE_THRESHOLD
    assert info.device == "cpu"
    assert info.classes == ("D00", "D10", "D20", "D40")
    assert info.sha256 == sha256_file(TINY_MODEL)
    assert info.path == str(TINY_MODEL)
    assert detector.providers == ("CPUExecutionProvider",)


def test_detect_decodes_the_two_scripted_queries() -> None:
    detector = OnnxDetector(TINY_MODEL, device="cpu", providers=["CPUExecutionProvider"])

    detections = detector.detect([_image(600, 400)])[0]

    assert [detection.damage_class for detection in detections] == [
        DamageClass.D00,
        DamageClass.D40,
    ]
    assert detections[0].score == pytest.approx(1.0 / (1.0 + math.exp(-5.0)))
    assert detections[0].bbox.to_xyxy() == pytest.approx((240.0, 160.0, 360.0, 240.0))
    assert detections[1].score == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))
    assert detections[1].bbox.to_xyxy() == pytest.approx((120.0, 80.0, 180.0, 120.0))


def test_detect_matches_an_independent_numpy_reference() -> None:
    detector = OnnxDetector(TINY_MODEL, device="cpu", providers=["CPUExecutionProvider"])
    image = _image(3650, 2044)

    detections = detector.detect([image])[0]
    reference = _numpy_reference(image)

    assert len(detections) == len(reference) == 2
    for detection, (code, score, box) in zip(detections, reference, strict=True):
        assert detection.damage_class.code == code
        assert detection.score == pytest.approx(score, abs=1e-9)
        assert detection.bbox.to_xyxy() == pytest.approx(box, abs=1e-6)


def test_detect_preserves_input_order_and_scales_per_image() -> None:
    detector = OnnxDetector(TINY_MODEL, device="cpu", providers=["CPUExecutionProvider"])

    small, large = detector.detect([_image(200, 100), _image(600, 400)])

    assert small[0].bbox.to_xyxy() == pytest.approx((80.0, 40.0, 120.0, 60.0))
    assert large[0].bbox.to_xyxy() == pytest.approx((240.0, 160.0, 360.0, 240.0))


def test_threshold_filters_the_low_confidence_query() -> None:
    detector = OnnxDetector(
        TINY_MODEL,
        device="cpu",
        score_threshold=0.99,
        providers=["CPUExecutionProvider"],
    )

    detections = detector.detect([_image(600, 400)])[0]

    assert [detection.damage_class for detection in detections] == [DamageClass.D00]


def test_missing_model_points_at_models_pull(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="rdv models pull") as excinfo:
        OnnxDetector(tmp_path / "missing.onnx", providers=["CPUExecutionProvider"])

    assert excinfo.value.exit_code == 4


def test_corrupt_model_is_a_model_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not an onnx graph")

    with pytest.raises(ModelError, match="Cannot load ONNX model") as excinfo:
        OnnxDetector(corrupt, providers=["CPUExecutionProvider"])

    assert excinfo.value.exit_code == 4


def test_graph_without_the_contract_outputs_is_rejected(tmp_path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["pixel_values"], ["boxes"])],
        "wrong_outputs",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 300, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "wrong-outputs.onnx"
    onnx.save(model, path)

    with pytest.raises(ModelError, match="missing outputs") as excinfo:
        OnnxDetector(path, providers=["CPUExecutionProvider"])

    assert excinfo.value.exit_code == 4


def test_cuda_without_the_provider_is_a_model_error() -> None:
    with pytest.raises(ModelError, match="CUDA inference was requested") as excinfo:
        OnnxDetector(TINY_MODEL, device="cuda", providers=["CPUExecutionProvider"])

    assert excinfo.value.exit_code == 4


def test_invalid_score_threshold_is_a_model_error() -> None:
    with pytest.raises(ModelError, match="Score threshold") as excinfo:
        OnnxDetector(
            TINY_MODEL,
            score_threshold=1.5,
            providers=["CPUExecutionProvider"],
        )

    assert excinfo.value.exit_code == 4
