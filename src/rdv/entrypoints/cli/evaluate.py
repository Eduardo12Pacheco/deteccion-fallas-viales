"""``rdv evaluate``: metrics, reports, plots and error gallery (S7)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rdv import composition
from rdv.application.evaluate_model import (
    EVALUATION_SCORE_THRESHOLD,
    evaluate_model,
)
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import configure, run
from rdv.errors import ConfigError

DEFAULT_OUTPUT = Path("reports/v1")
_SPLITS = ("train", "val", "test", "all")


def _parse_operating_threshold(value: str) -> float | None:
    if value.strip().lower() == "auto":
        return None
    try:
        threshold = float(value)
    except ValueError as exc:
        raise ConfigError(
            f"--op-threshold must be 'auto' or a number in [0, 1], got {value!r}"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("--op-threshold must be within [0, 1]")
    return threshold


def evaluate(
    data: Annotated[
        Path | None,
        typer.Option("--data", help="Dataset root containing the train folders."),
    ] = None,
    splits: Annotated[
        Path | None,
        typer.Option("--splits", help="Split manifest (defaults to <data>/splits.json)."),
    ] = None,
    split: Annotated[
        str,
        typer.Option("--split", help="Split to evaluate: train, val, test or all."),
    ] = "test",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Manifest model id or path to a local ONNX file."),
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", help="Directory receiving the report artifacts.")
    ] = DEFAULT_OUTPUT,
    op_threshold: Annotated[
        str,
        typer.Option(
            "--op-threshold",
            help="Operating threshold ('auto' selects the F1 optimum on val).",
        ),
    ] = "auto",
    ood: Annotated[
        Path | None,
        typer.Option("--ood", help="Optional out-of-domain VOC directory (Norway)."),
    ] = None,
    max_images: Annotated[
        int | None,
        typer.Option("--max-images", min=1, help="Evaluate at most this many images."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Execution device: auto, cpu or cuda."),
    ] = None,
) -> None:
    """Evaluate a model on a labeled split and write the S7 reports."""

    def action() -> None:
        if split not in _SPLITS:
            raise ConfigError(f"--split must be one of {_SPLITS}, got {split!r}")
        overrides: dict[str, object] = {}
        if device is not None:
            overrides["device"] = device
        settings = Settings.from_env(**overrides)
        configure(settings)
        data_root = Path(data) if data is not None else settings.data_dir / "rdd2022"
        splits_path = Path(splits) if splits is not None else data_root / "splits.json"
        threshold = _parse_operating_threshold(op_threshold)
        countries = composition.discover_countries(data_root)

        dataset = composition.open_evaluation_repository(
            data_root,
            split=split,
            countries=countries,
            splits_path=splits_path,
        )
        selection_dataset = None
        if threshold is None:
            selection_dataset = (
                dataset
                if split == "val"
                else composition.open_evaluation_repository(
                    data_root,
                    split="val",
                    countries=countries,
                    splits_path=splits_path,
                )
            )
        detector = composition.build_detector(
            settings,
            model=model,
            score_threshold=EVALUATION_SCORE_THRESHOLD,
        )
        ood_dataset = composition.open_ood_repository(Path(ood)) if ood is not None else None

        outcome = evaluate_model(
            dataset,
            selection_dataset=selection_dataset,
            ood_dataset=ood_dataset,
            detector=detector,
            writer=composition.build_evaluation_report_writer(),
            output_dir=out,
            split=split,
            score_threshold=EVALUATION_SCORE_THRESHOLD,
            operating_threshold=threshold,
            max_images=max_images,
        )
        typer.echo(
            f"Evaluated {outcome.images_evaluated} image(s) on split {split}: "
            f"mAP@0.5={outcome.metrics.mAP50:.4f} "
            f"mAP@0.5:0.95={outcome.metrics.mAP50_95:.4f} "
            f"(threshold {outcome.operating_threshold:.4f}, {outcome.operating_source})"
        )
        typer.echo(f"Report: {outcome.artifacts.report_html}")

    run(action)
