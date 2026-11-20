"""``rdv train``: validate a config or fine-tune the detector (S9, D16)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rdv import composition
from rdv.application.train_model import TrainingRequest, TrainingResult, train_model
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import ConsoleProgressSink, configure, run

DEFAULT_RUN_DIR = Path("runs/train")


def train(
    config: Annotated[
        Path,
        typer.Option("--config", help="YAML training configuration."),
    ],
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Directory for checkpoints, logs and reports."),
    ] = DEFAULT_RUN_DIR,
    resume: Annotated[
        str,
        typer.Option("--resume", help="auto, none, or a checkpoint path."),
    ] = "auto",
    benchmark_only: Annotated[
        bool,
        typer.Option(
            "--benchmark-only",
            help="Measure throughput and peak VRAM instead of training.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate config, dataset and model instantiation without training.",
        ),
    ] = False,
) -> None:
    """Fine-tune RT-DETRv2 on the configured RDD2022 split."""

    def action() -> None:
        settings = Settings.from_env()
        configure(settings)
        trainer = composition.build_trainer(
            settings,
            config_path=config,
            progress=ConsoleProgressSink(),
        )
        result = train_model(
            trainer,
            TrainingRequest(
                run_dir=run_dir,
                resume=resume,
                benchmark_only=benchmark_only,
                dry_run=dry_run,
            ),
        )
        _report(result)

    run(action)


def _report(result: TrainingResult) -> None:
    if result.dry_run:
        typer.echo(f"Dry run OK: {result.run_dir / 'dry_run.json'}")
        return
    if result.benchmark is not None:
        typer.echo(
            f"Micro-batch {result.benchmark['micro_batch']} "
            f"({result.benchmark['img_per_s']} img/s, "
            f"peak {result.benchmark['peak_memory_gb']} GB) -> "
            f"{result.run_dir / 'benchmark.json'}"
        )
        return
    loss = "n/a" if result.last_loss is None else f"{result.last_loss:.4f}"
    typer.echo(
        f"Trained {result.epochs_completed} epoch(s), {result.steps} step(s), "
        f"last loss {loss}; checkpoints in {result.run_dir}"
    )
