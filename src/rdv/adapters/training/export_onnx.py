"""ONNX export recipe with the S4 contract and a torch-vs-ORT parity gate (D8).

RT-DETRv2 is exported through a small wrapper that returns ``logits`` and
``pred_boxes``. The wrapper pads/truncates the query dimension to the S4 fixed
300, so the tiny test model (20 queries) and the real checkpoint share exactly
the same graph contract. The exporter tries the current dynamo path at opsets
18 and 17, then falls back to the legacy TorchScript exporter (the stable path
for the pinned transformers ``<5``), and always re-saves a single-file model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch

from rdv.application.export_model import ExportRequest, ExportResult
from rdv.application.ports import ModelInfo
from rdv.errors import ModelError
from rdv.hashing import sha256_file

CONTRACT_QUERIES = 300
CONTRACT_INPUT = "pixel_values"
CONTRACT_OUTPUTS = ("logits", "pred_boxes")
PARITY_LIMIT = 1e-3
PAD_LOGIT = -10000.0


def _fit_queries(tensor: torch.Tensor, target: int, pad_value: float) -> torch.Tensor:
    current = int(tensor.shape[1])
    if current == target:
        return tensor
    if current > target:
        return tensor[:, :target, :]
    padding = torch.full(
        (tensor.shape[0], target - current, tensor.shape[2]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=1)


class ContractExportWrapper(torch.nn.Module):
    """Return the two S4 outputs with a fixed 300-query dimension."""

    def __init__(self, model: torch.nn.Module, *, queries: int = CONTRACT_QUERIES) -> None:
        super().__init__()
        self.model = model
        self.queries = queries

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values)
        logits = _fit_queries(outputs.logits, self.queries, PAD_LOGIT)
        boxes = _fit_queries(outputs.pred_boxes, self.queries, 0.0)
        return logits, boxes


@dataclass(frozen=True, slots=True)
class ExportedGraph:
    """Inspected ONNX contract of an exported file."""

    opset: int
    inputs: tuple[tuple[str, tuple[int | str | None, ...]], ...]
    outputs: tuple[tuple[str, tuple[int | str | None, ...]], ...]


def _dims(value: onnx.ValueInfoProto) -> tuple[int | str | None, ...]:
    dimensions: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.dim_value:
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _fix_contract_shapes(model: onnx.ModelProto) -> None:
    """Declare the constant query/channel dimensions that tracing leaves symbolic.

    The wrapper always pads the query axis to ``CONTRACT_QUERIES`` and the class
    axis to four; the legacy exporter only infers them as symbolic Concat/Pad
    outputs. Writing the true static shape keeps S4's ``[N, 300, 4]`` contract
    visible to downstream validators without changing the computed graph.
    """
    for output in model.graph.output:
        dimensions = output.type.tensor_type.shape.dim
        if len(dimensions) >= 3:
            dimensions[1].dim_param = ""
            dimensions[1].dim_value = CONTRACT_QUERIES
            dimensions[2].dim_param = ""
            dimensions[2].dim_value = 4


def inspect_graph(path: Path) -> ExportedGraph:
    """Load an ONNX file, run the checker and inspect its I/O contract."""
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ModelError(f"Invalid ONNX export {path}: {exc}") from exc
    return ExportedGraph(
        opset=max(import_.version for import_ in model.opset_import),
        inputs=tuple((value.name, _dims(value)) for value in model.graph.input),
        outputs=tuple((value.name, _dims(value)) for value in model.graph.output),
    )


def _attempt_export(
    wrapper: ContractExportWrapper,
    destination: Path,
    *,
    input_size: int,
    opset: int,
    dynamo: bool,
) -> None:
    example = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (example,),
        str(destination),
        input_names=[CONTRACT_INPUT],
        output_names=list(CONTRACT_OUTPUTS),
        opset_version=opset,
        dynamo=dynamo,
        do_constant_folding=True,
        dynamic_axes={
            CONTRACT_INPUT: {0: "batch"},
            CONTRACT_OUTPUTS[0]: {0: "batch"},
            CONTRACT_OUTPUTS[1]: {0: "batch"},
        },
    )


def export_onnx_model(
    model: torch.nn.Module,
    output: Path,
    *,
    input_size: int = 640,
    opsets: tuple[int, ...] = (18, 17),
) -> ExportedGraph:
    """Export ``model`` to a single-file ONNX graph, trying the recipe ladder."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ContractExportWrapper(model.eval())
    temporary = output.with_name(output.name + ".part")
    attempts: list[tuple[int, bool]] = [(opset, True) for opset in opsets]
    attempts.append((opsets[-1] if opsets else 18, False))
    errors: list[str] = []
    for opset, dynamo in attempts:
        temporary.unlink(missing_ok=True)
        try:
            with torch.no_grad():
                _attempt_export(
                    wrapper,
                    temporary,
                    input_size=input_size,
                    opset=opset,
                    dynamo=dynamo,
                )
            inspect_graph(temporary)
        except Exception as exc:
            errors.append(f"opset {opset} dynamo={dynamo}: {type(exc).__name__}: {exc}")
            continue
        onnx_model = onnx.load(temporary)
        _fix_contract_shapes(onnx_model)
        onnx.save_model(onnx_model, str(output), save_as_external_data=False)
        temporary.unlink(missing_ok=True)
        return inspect_graph(output)
    temporary.unlink(missing_ok=True)
    raise ModelError("ONNX export failed after every recipe attempt: " + " | ".join(errors))


def parity_report(
    model: torch.nn.Module,
    onnx_path: Path,
    *,
    input_size: int = 640,
) -> tuple[float, float]:
    """Max absolute torch-vs-ORT difference for logits and boxes (S4)."""
    wrapper = ContractExportWrapper(model.eval())
    generator = torch.Generator().manual_seed(0)
    example = torch.rand((1, 3, input_size, input_size), generator=generator)
    with torch.no_grad():
        torch_logits, torch_boxes = wrapper(example)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    raw = session.run(
        None,
        {CONTRACT_INPUT: example.numpy().astype(np.float32)},
    )
    outputs = dict(zip([output.name for output in session.get_outputs()], raw, strict=True))
    logits_diff = float(np.max(np.abs(outputs["logits"] - torch_logits.detach().cpu().numpy())))
    boxes_diff = float(np.max(np.abs(outputs["pred_boxes"] - torch_boxes.detach().cpu().numpy())))
    return logits_diff, boxes_diff


def load_export_model(checkpoint: Path, *, offline: bool = False) -> torch.nn.Module:
    """Load an HF checkpoint directory for export."""
    from transformers import RTDetrV2ForObjectDetection

    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise ModelError(f"Checkpoint directory not found: {checkpoint}")
    try:
        model = RTDetrV2ForObjectDetection.from_pretrained(
            str(checkpoint), local_files_only=offline
        )
    except Exception as exc:
        raise ModelError(f"Cannot load checkpoint {checkpoint}: {exc}") from exc
    return cast(torch.nn.Module, model.to("cpu"))


class OnnxModelExporter:
    """``ModelExporter`` adapter used by ``rdv export`` (T9)."""

    def __init__(self, *, offline: bool = False) -> None:
        self._offline = offline

    def export(self, request: ExportRequest) -> ExportResult:
        model = load_export_model(request.checkpoint, offline=self._offline)
        graph = export_onnx_model(
            model,
            request.output,
            input_size=request.input_size,
        )
        logits_diff, boxes_diff = parity_report(
            model,
            request.output,
            input_size=request.input_size,
        )
        if logits_diff > PARITY_LIMIT or boxes_diff > PARITY_LIMIT:
            raise ModelError(
                f"ONNX parity failed: logits {logits_diff:.3e}, boxes {boxes_diff:.3e} "
                f"(limit {PARITY_LIMIT:.1e})"
            )
        output_names = [name for name, _ in graph.outputs]
        if output_names != list(CONTRACT_OUTPUTS):
            raise ModelError(
                f"Exported graph outputs {output_names!r}, expected {list(CONTRACT_OUTPUTS)!r}"
            )
        query_dim = graph.outputs[0][1][1] if len(graph.outputs[0][1]) > 1 else None
        if query_dim != CONTRACT_QUERIES:
            raise ModelError(
                f"Exported graph query dimension is {query_dim!r}, expected {CONTRACT_QUERIES}"
            )
        return ExportResult(
            output=request.output,
            opset=graph.opset,
            logits_max_abs_diff=logits_diff,
            boxes_max_abs_diff=boxes_diff,
            num_queries=CONTRACT_QUERIES,
        )


def exported_model_info(path: Path, *, model_id: str | None = None) -> ModelInfo:
    """Describe an exported ONNX file (used by ``rdv doctor`` style output)."""
    graph = inspect_graph(path)
    input_size = graph.inputs[0][1][2] if graph.inputs else 640
    return ModelInfo(
        id=model_id or path.stem,
        input_size=int(input_size) if isinstance(input_size, int) else 640,
        score_threshold=0.25,
        device="cpu",
        sha256=sha256_file(path),
        path=str(path),
    )
