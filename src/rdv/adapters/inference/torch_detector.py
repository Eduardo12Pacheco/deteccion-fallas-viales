"""Optional PyTorch detector for bf16 CUDA inference and GPU benchmarks (D8).

The base install must stay torch-free (S2) and the architecture contract only
allows a static torch import inside ``rdv.adapters.training``. This adapter
therefore imports the optional stack through ``importlib`` at construction time
and raises :class:`MissingExtraError` with the documented install hint when the
extra is absent.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rdv.adapters.inference.preprocess import INPUT_SIZE, preprocess_image
from rdv.application.ports import (
    DEFAULT_CLASS_CODES,
    DEFAULT_SCORE_THRESHOLD,
    ModelInfo,
    RgbImage,
)
from rdv.domain.decoding import decode_detr_outputs
from rdv.domain.detection import Detection
from rdv.errors import MissingExtraError, ModelError
from rdv.hashing import sha256_file

TRAINING_HINT = (
    "Install it with `uv sync --extra train --extra cpu` (CPU) or "
    "`uv sync --extra train --extra cuda` (CUDA)."
)


def _resolve_device(torch: Any, device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ModelError(f"Unsupported device {device!r}; expected auto, cpu or cuda")
    if normalized == "cuda" and not bool(torch.cuda.is_available()):
        raise ModelError("CUDA inference was requested but torch.cuda.is_available() is false")
    return normalized


class TorchDetector:
    """Detector backed by a Hugging Face RT-DETRv2 checkpoint directory.

    Args:
        path: checkpoint directory saved by ``save_pretrained``.
        score_threshold: minimum decoded score.
        device: ``auto``, ``cpu`` or ``cuda``.
        input_size: square network input side.
        model_id: override for the reported model id (default: directory name).
        offline: pass ``local_files_only`` to the transformers loader.
        dtype: ``bf16`` on CUDA (D8); CPU always runs float32.

    Raises:
        MissingExtraError: torch/transformers are not installed (exit 6).
        ModelError: the path is not a checkpoint, the device is unavailable,
            or the weights fail to load.
    """

    def __init__(
        self,
        path: Path,
        *,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        device: str = "auto",
        input_size: int = INPUT_SIZE,
        model_id: str | None = None,
        offline: bool = False,
        dtype: str = "bf16",
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - exercised on a base install
            raise MissingExtraError(
                f"The torch backend requires the training extra. {TRAINING_HINT}"
            ) from exc
        if not 0.0 <= score_threshold <= 1.0:
            raise ModelError(f"Score threshold must be within [0, 1], got {score_threshold!r}")
        if input_size <= 0:
            raise ModelError(f"Input size must be positive, got {input_size!r}")
        checkpoint = Path(path)
        if not checkpoint.is_dir():
            raise ModelError(
                f"TorchDetector needs a saved checkpoint directory, got {checkpoint}. "
                "Pass the directory written by the trainer (`runs/.../best`)."
            )
        self._torch = torch
        self._device = _resolve_device(torch, device)
        self._dtype = dtype if self._device == "cuda" and dtype == "bf16" else "fp32"
        self._score_threshold = float(score_threshold)
        self._input_size = input_size
        try:
            model = transformers.RTDetrV2ForObjectDetection.from_pretrained(
                str(checkpoint), local_files_only=offline
            )
        except Exception as exc:
            raise ModelError(f"Cannot load torch checkpoint {checkpoint}: {exc}") from exc
        self._model = model.eval().to(self._device)
        weights = checkpoint / "model.safetensors"
        self._info = ModelInfo(
            id=model_id or checkpoint.name,
            input_size=input_size,
            score_threshold=self._score_threshold,
            device=self._device,
            sha256=sha256_file(weights) if weights.is_file() else None,
            classes=DEFAULT_CLASS_CODES,
            path=str(checkpoint),
        )

    @property
    def info(self) -> ModelInfo:
        """Identity, threshold and device of the loaded model."""
        return self._info

    def detect(self, images: Sequence[RgbImage]) -> list[list[Detection]]:
        """Detect damage with bf16 autocast on CUDA, float32 on CPU."""
        results: list[list[Detection]] = []
        torch = self._torch
        for image in images:
            tensor = preprocess_image(image, self._input_size)
            pixel_values = torch.from_numpy(tensor).to(self._device)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self._dtype == "bf16",
                ),
            ):
                outputs = self._model(pixel_values=pixel_values)
            logits = outputs.logits[0].float().to("cpu").numpy()
            boxes = outputs.pred_boxes[0].float().to("cpu").numpy()
            height, width = int(image.shape[0]), int(image.shape[1])
            results.append(
                decode_detr_outputs(
                    logits,
                    boxes,
                    (width, height),
                    score_threshold=self._score_threshold,
                )
            )
        return results
