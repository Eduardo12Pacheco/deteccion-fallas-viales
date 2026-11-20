"""Real-model smoke tests (marker ``realmodel``; never runs in CI).

Two independent checks that only make sense with the T10 artifact:

* the packaged samples still produce the classes recorded in
  ``reports/v1/sample_expectations.json`` (the post-release smoke contract);
* the ONNX graph matches the fine-tuned torch checkpoint on eight validation
  images within the S4 1e-3 parity gate.

Both are skipped unless the corresponding environment variable is set, so the
default suite stays offline and torch-free.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rdv.adapters.inference.onnx_detector import OnnxDetector

pytestmark = pytest.mark.realmodel

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS = REPO_ROOT / "reports" / "v1" / "sample_expectations.json"
SAMPLES = REPO_ROOT / "src" / "rdv" / "assets" / "samples"


def _real_model() -> Path:
    value = os.environ.get("RDV_REAL_MODEL")
    if not value:
        pytest.skip("RDV_REAL_MODEL is not set")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"RDV_REAL_MODEL does not exist: {path}")
    return path


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def test_packaged_samples_match_the_committed_expectations() -> None:
    model = _real_model()
    if not EXPECTATIONS.is_file():
        pytest.skip(f"missing {EXPECTATIONS.relative_to(REPO_ROOT)}")
    payload = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    detector = OnnxDetector(model, score_threshold=threshold)

    assert payload["samples"], "sample expectations must not be empty"
    for sample in payload["samples"]:
        image_path = SAMPLES / sample["image"]
        assert image_path.is_file(), image_path
        detections = detector.detect([_load_rgb(image_path)])[0]
        by_class = {detection.damage_class.code: detection.score for detection in detections}
        assert sample["expected_classes"], sample
        for code in sample["expected_classes"]:
            assert code in by_class, f"{sample['image']} lost class {code}"
            assert by_class[code] >= threshold
        assert max(by_class.values()) >= threshold


def test_onnx_matches_the_checkpoint_on_eight_validation_images() -> None:
    onnx_path = _real_model()
    checkpoint = os.environ.get("RDV_REAL_CHECKPOINT")
    if not checkpoint:
        pytest.skip("RDV_REAL_CHECKPOINT is not set")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from rdv.adapters.datasets.rdd2022 import Rdd2022VocRepository
    from rdv.adapters.inference.preprocess import preprocess_image
    from rdv.adapters.training.export_onnx import ContractExportWrapper

    data_root = Path(os.environ.get("RDV_DATA_DIR", "data")) / "rdd2022"
    splits = data_root / "splits.json"
    if not splits.is_file():
        pytest.skip(f"missing split manifest {splits}")
    repository = Rdd2022VocRepository(data_root, ("Japan", "India", "Czech"), split="val")
    images = sorted(repository, key=lambda image: image.image_id)[:8]
    assert len(images) == 8

    from transformers import RTDetrV2ForObjectDetection

    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint).eval()
    wrapper = ContractExportWrapper(model)
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for annotation in images:
            tensor = preprocess_image(repository.load_image(annotation.image_id))
            logits_t, boxes_t = wrapper(torch.from_numpy(tensor))
            logits_o, boxes_o = session.run(None, {"pixel_values": tensor})
            logits_diff = float(np.max(np.abs(logits_o - logits_t.cpu().numpy())))
            boxes_diff = float(np.max(np.abs(boxes_o - boxes_t.cpu().numpy())))
            assert logits_diff <= 1e-3, (annotation.image_id, logits_diff)
            assert boxes_diff <= 1e-3, (annotation.image_id, boxes_diff)
