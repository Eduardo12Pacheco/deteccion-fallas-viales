"""``rdv detect`` for images (S5); video detection is added in T7."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rdv import composition
from rdv.adapters.media.images import VIDEO_EXTENSIONS
from rdv.application.detect_image import detect_images
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import ConsoleProgressSink, configure, run
from rdv.errors import MediaError

DEFAULT_OUTPUT = Path("rdv-out")


def detect(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Image files to analyze."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Manifest model id or path to a local ONNX file."),
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", help="Directory receiving the annotated outputs.")
    ] = DEFAULT_OUTPUT,
    conf: Annotated[
        float | None,
        typer.Option("--conf", min=0.0, max=1.0, help="Score threshold override."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Execution device: auto, cpu or cuda."),
    ] = None,
) -> None:
    """Detect road damage in images and write annotated PNG/JSON reports."""

    def action() -> None:
        if not paths:
            raise MediaError("No input images were provided")
        for path in paths:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                raise MediaError(f"Video detection is not available yet: {path}. Pass image files.")
        overrides: dict[str, object] = {}
        if device is not None:
            overrides["device"] = device
        if conf is not None:
            overrides["score_threshold"] = conf
        settings = Settings.from_env(**overrides)
        configure(settings)
        detector = composition.build_detector(settings, model=model)
        result = detect_images(
            paths,
            detector=detector,
            media=composition.build_media_source(),
            renderer=composition.build_renderer(),
            detections_writer=composition.build_detections_writer(),
            summary_writer=composition.build_summary_writer(),
            output_dir=out,
            progress=ConsoleProgressSink(),
        )
        typer.echo(
            f"Wrote {len(result.images)} image(s), {result.total_detections} detection(s) to {out}"
        )
        typer.echo(f"Summary: {result.summary_markdown}")

    run(action)
