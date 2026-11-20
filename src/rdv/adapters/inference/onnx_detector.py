"""ONNX Runtime detector: the default, torch-free inference adapter (D8).

The adapter owns session creation, provider selection and preprocessing; the
decoding itself stays in the pure domain (``rdv.domain.decoding``) so the
inherited COCO head is decoded exactly like the fine-tuned one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import onnxruntime as ort

from rdv.adapters.inference.preprocess import INPUT_SIZE, preprocess_image
from rdv.adapters.inference.providers import resolve_providers, selected_device
from rdv.application.ports import (
    DEFAULT_CLASS_CODES,
    DEFAULT_SCORE_THRESHOLD,
    ModelInfo,
    RgbImage,
)
from rdv.domain.decoding import decode_detr_outputs
from rdv.domain.detection import Detection
from rdv.errors import ModelError
from rdv.hashing import sha256_file

REQUIRED_OUTPUTS = ("logits", "pred_boxes")


class OnnxDetector:
    """Detector backed by a single-file ONNX graph (contract S4).

    Args:
        path: ONNX model file.
        score_threshold: minimum score kept by the decoder.
        device: ``auto``, ``cpu`` or ``cuda`` (ONNX Runtime providers).
        input_size: square input side expected by the graph.
        model_id: manifest id; defaults to the file stem.
        model_version: manifest version, purely informational.
        model_license: manifest license identifier, purely informational.
        providers: provider names to consider (tests inject a fake list).
        session_options: optional ONNX Runtime session options.

    Raises:
        ModelError: the file is missing, the runtime cannot load it, its
            outputs do not follow the model contract, or CUDA was requested
            without the CUDA provider.
    """

    def __init__(
        self,
        path: Path,
        *,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        device: str = "auto",
        input_size: int = INPUT_SIZE,
        model_id: str | None = None,
        model_version: str | None = None,
        model_license: str | None = None,
        providers: Sequence[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"Model file not found: {path}. Run `rdv models pull` to download it.")
        if not 0.0 <= score_threshold <= 1.0:
            raise ModelError(f"Score threshold must be within [0, 1], got {score_threshold!r}")
        if input_size <= 0:
            raise ModelError(f"Input size must be positive, got {input_size!r}")

        self._path = path
        self._score_threshold = float(score_threshold)
        self._input_size = input_size
        resolved = resolve_providers(device, providers)
        self._device = selected_device(resolved)
        try:
            self._session = ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=resolved,
            )
        except Exception as exc:  # onnxruntime raises several types for bad files
            raise ModelError(f"Cannot load ONNX model {path.name}: {exc}") from exc

        inputs = self._session.get_inputs()
        if not inputs:
            raise ModelError(f"ONNX model {path.name} has no inputs")
        output_names = [output.name for output in self._session.get_outputs()]
        missing = [name for name in REQUIRED_OUTPUTS if name not in output_names]
        if missing:
            raise ModelError(
                f"ONNX model {path.name} is missing outputs {missing!r}; "
                f"expected {REQUIRED_OUTPUTS!r}"
            )
        self._input_name = inputs[0].name
        self._output_names = output_names
        self._info = ModelInfo(
            id=model_id or path.stem,
            input_size=input_size,
            score_threshold=self._score_threshold,
            device=self._device,
            sha256=sha256_file(path),
            classes=DEFAULT_CLASS_CODES,
            path=str(path),
            version=model_version,
            license=model_license,
        )

    @property
    def info(self) -> ModelInfo:
        """Identity, threshold and device of the loaded model."""
        return self._info

    @property
    def providers(self) -> tuple[str, ...]:
        """Providers actually active in the session, in priority order."""
        return tuple(self._session.get_providers())

    def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
        """Detect damage in each RGB image, preserving input order.

        Raises:
            ModelError: the runtime fails to execute the graph.
            ValueError: an input is not an ``HxWx3`` RGB ``uint8`` array.
        """
        results: list[list[Detection]] = []
        for image in images:
            tensor = preprocess_image(image, self._input_size)
            try:
                raw = self._session.run(None, {self._input_name: tensor})
            except Exception as exc:
                raise ModelError(
                    f"ONNX inference failed for model {self._path.name}: {exc}"
                ) from exc
            outputs: dict[str, Any] = dict(zip(self._output_names, raw, strict=True))
            height, width = int(image.shape[0]), int(image.shape[1])
            detections = decode_detr_outputs(
                outputs["logits"][0],
                outputs["pred_boxes"][0],
                (width, height),
                score_threshold=self._score_threshold,
            )
            results.append(detections)
        return results
