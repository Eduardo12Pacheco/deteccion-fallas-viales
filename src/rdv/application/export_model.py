"""Export use case: a thin command object over the ``ModelExporter`` port (T9).

The adapter owns PyTorch and ONNX; this module only describes what to export
and carries the parity measurements back to the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Checkpoint directory to export and the destination ONNX file."""

    checkpoint: Path
    output: Path
    input_size: int = 640

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output", Path(self.output))
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Exported artifact plus the torch-vs-ONNX parity measurements (S4)."""

    output: Path
    opset: int
    logits_max_abs_diff: float
    boxes_max_abs_diff: float
    num_queries: int


class ModelExporter(Protocol):
    """Port implemented by the ONNX export adapter."""

    def export(self, request: ExportRequest) -> ExportResult:
        """Export the checkpoint and verify the graph contract."""
        ...


def export_model(exporter: ModelExporter, request: ExportRequest) -> ExportResult:
    """Execute one export request through the injected adapter."""
    return exporter.export(request)
