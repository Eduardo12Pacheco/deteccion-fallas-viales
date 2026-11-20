"""``rdv samples``: copy the packaged demo images to a directory (S19)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Annotated

import typer

from rdv.entrypoints.cli.console import run
from rdv.errors import MediaError


def samples(
    dest: Annotated[
        Path, typer.Option("--dest", help="Directory receiving the sample files.")
    ] = Path("samples"),
) -> None:
    """Copy the six packaged sample images (and their license) to ``--dest``."""

    def action() -> None:
        source = resources.files("rdv").joinpath("assets/samples")
        try:
            files = sorted(
                (item for item in source.iterdir() if item.is_file()),
                key=lambda item: item.name,
            )
            dest.mkdir(parents=True, exist_ok=True)
            for item in files:
                (dest / item.name).write_bytes(item.read_bytes())
        except OSError as exc:
            raise MediaError(f"Cannot copy samples to {dest}: {exc}") from exc
        images = sum(1 for item in files if item.name.endswith(".jpg"))
        typer.echo(f"Copied {images} images and {len(files) - images} notice file(s) to {dest}")

    run(action)
