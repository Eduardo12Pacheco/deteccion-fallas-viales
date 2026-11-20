"""``rdv detect`` for images (S5) and videos (S6)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rdv import composition
from rdv.adapters.media.images import VIDEO_EXTENSIONS
from rdv.application.detect_image import detect_images
from rdv.application.detect_video import DEFAULT_ANALYSIS_FPS, detect_video
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import ConsoleProgressSink, configure, run
from rdv.errors import MediaError

DEFAULT_OUTPUT = Path("rdv-out")


def detect(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Image or video files to analyze."),
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
    analysis_fps: Annotated[
        float | None,
        typer.Option(
            "--analysis-fps",
            min=0.1,
            help="Analyzed frames per second for videos (default 5).",
        ),
    ] = None,
    max_frames: Annotated[
        int | None,
        typer.Option(
            "--max-frames",
            min=1,
            help="Stop a video after this many source frames.",
        ),
    ] = None,
) -> None:
    """Detect road damage in images and videos and write annotated reports."""

    def action() -> None:
        if not paths:
            raise MediaError("No input images or videos were provided")
        video_paths = [path for path in paths if path.suffix.lower() in VIDEO_EXTENSIONS]
        image_paths = [path for path in paths if path.suffix.lower() not in VIDEO_EXTENSIONS]
        if not video_paths and (analysis_fps is not None or max_frames is not None):
            raise MediaError("--analysis-fps and --max-frames only apply to video inputs")
        overrides: dict[str, object] = {}
        if device is not None:
            overrides["device"] = device
        if conf is not None:
            overrides["score_threshold"] = conf
        settings = Settings.from_env(**overrides)
        configure(settings)
        detector = composition.build_detector(settings, model=model)
        progress = ConsoleProgressSink()

        if image_paths:
            result = detect_images(
                image_paths,
                detector=detector,
                media=composition.build_media_source(),
                renderer=composition.build_renderer(),
                detections_writer=composition.build_detections_writer(),
                summary_writer=composition.build_summary_writer(),
                output_dir=out,
                progress=progress,
            )
            typer.echo(
                f"Wrote {len(result.images)} image(s), {result.total_detections} detection(s) "
                f"to {out}"
            )
            typer.echo(f"Summary: {result.summary_markdown}")

        for video_path in video_paths:
            source = composition.build_video_source(video_path)
            outcome = detect_video(
                source,
                detector=detector,
                renderer=composition.build_renderer(),
                sink=composition.build_video_sink(),
                detections_writer=composition.build_detections_writer(),
                summary_writer=composition.build_summary_writer(),
                output_dir=out,
                analysis_fps=analysis_fps if analysis_fps is not None else DEFAULT_ANALYSIS_FPS,
                max_frames=max_frames,
                progress=progress,
            )
            typer.echo(
                f"Wrote {outcome.annotated.name}: {outcome.frames_analyzed} analyzed frame(s) "
                f"of {outcome.frames_processed}, {outcome.detections_total} detection(s)"
            )
            typer.echo(f"Summary: {outcome.summary_markdown}")

    run(action)
