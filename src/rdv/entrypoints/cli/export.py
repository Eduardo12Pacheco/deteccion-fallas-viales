"""``rdv export``: turn a trained checkpoint into the S4 ONNX artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rdv import composition
from rdv.application.export_model import ExportRequest, ExportResult, export_model
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import configure, run


def export(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", help="Saved HF checkpoint directory (for example best/)."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Destination single-file ONNX model."),
    ],
) -> None:
    """Export a fine-tuned checkpoint and verify torch-vs-ONNX parity."""

    def action() -> None:
        settings = Settings.from_env()
        configure(settings)
        exporter = composition.build_exporter(settings)
        result = export_model(
            exporter,
            ExportRequest(checkpoint=checkpoint, output=out),
        )
        _report(result)

    run(action)


def _report(result: ExportResult) -> None:
    typer.echo(
        f"Exported {result.output} (opset {result.opset}, "
        f"{result.num_queries} queries, parity logits {result.logits_max_abs_diff:.2e}, "
        f"boxes {result.boxes_max_abs_diff:.2e})"
    )
