"""``rdv benchmark`` measures CPU/GPU latency and video throughput (S11)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rdv import composition
from rdv.adapters.reporting.detections_json import dumps_document
from rdv.application.benchmark import benchmark_model
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import configure, run
from rdv.errors import MediaError

DEFAULT_OUTPUT = Path("benchmark.json")


def benchmark(
    model: Annotated[
        str | None,
        typer.Option("--model", help="Manifest model id or path to a local ONNX file."),
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", help="JSON file receiving the benchmark report.")
    ] = DEFAULT_OUTPUT,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Torch checkpoint directory for the optional GPU row.",
        ),
    ] = None,
) -> None:
    """Measure inference latency and video throughput on the local machine."""

    def action() -> None:
        settings = Settings.from_env()
        configure(settings)
        detector = composition.build_detector(settings, model=model)
        gpu_detector = composition.build_benchmark_gpu_detector(
            settings, model=model, checkpoint=checkpoint
        )
        result = benchmark_model(
            detector,
            gpu_detector=gpu_detector,
            hardware_gpu=composition.describe_gpu(),
        )
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(dumps_document(result.as_dict()), encoding="utf-8")
        except OSError as exc:
            raise MediaError(f"Cannot write benchmark report {out}: {exc}") from exc
        cpu = result.cpu
        typer.echo(
            f"CPU p50 {cpu['p50_ms']} ms, p95 {cpu['p95_ms']} ms; "
            f"video {result.video['frames_per_s']} frames/s"
        )
        if result.gpu is not None:
            typer.echo(
                f"GPU p50 {result.gpu['p50_ms']} ms, p95 {result.gpu['p95_ms']} ms "
                f"({result.hardware['gpu']})"
            )
        typer.echo(f"Report: {out}")

    run(action)
