"""``rdv models`` commands: list, pull, path, verify (S20/D13)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rdv import composition
from rdv.adapters.models.store import CachedModelStore
from rdv.application import manage_models
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import format_mib, run
from rdv.errors import ModelError
from rdv.hashing import sha256_file

models_app = typer.Typer(
    name="models",
    help="List, pull, locate, and verify model files.",
    no_args_is_help=True,
)

ManifestOption = Annotated[
    Path | None, typer.Option("--manifest", help="Model manifest override (tests/development).")
]
ModelDirOption = Annotated[
    Path | None, typer.Option("--model-dir", help="Model cache directory override.")
]


def _settings(model_dir: Path | None) -> Settings:
    if model_dir is None:
        return Settings.from_env()
    return Settings.from_env(model_dir=model_dir)


def _store(manifest: Path | None, model_dir: Path | None) -> tuple[Settings, CachedModelStore]:
    settings = _settings(model_dir)
    return settings, composition.build_model_store(
        settings, manifest=composition.load_model_manifest(manifest)
    )


@models_app.command("list")
def list_models(manifest: ManifestOption = None, model_dir: ModelDirOption = None) -> None:
    """Show the models pinned in the manifest."""

    def action() -> None:
        settings, store = _store(manifest, model_dir)
        entries = manage_models.list_models(store)
        if not entries:
            typer.echo("No models in the manifest yet. Pass --model PATH to use a local ONNX file.")
            return
        table = Table(title="Model manifest")
        for column in ("Id", "Version", "Size", "sha256", "License", "Cached"):
            table.add_column(column)
        for entry in entries:
            cached = "yes" if (settings.model_dir / Path(entry.url).name).is_file() else "no"
            table.add_row(
                entry.id,
                entry.version,
                format_mib(entry.size),
                f"{entry.sha256[:12]}…",
                entry.license,
                cached,
            )
        Console().print(table)

    run(action)


@models_app.command("pull")
def pull_model(
    model_id: Annotated[
        str | None, typer.Argument(help="Manifest model id; defaults to RDV_MODEL.")
    ] = None,
    manifest: ManifestOption = None,
    model_dir: ModelDirOption = None,
) -> None:
    """Download a model, verify its sha256, and install it atomically."""

    def action() -> None:
        settings, store = _store(manifest, model_dir)
        selected = model_id or settings.model
        if selected.endswith(".onnx") or Path(selected).is_file():
            raise ModelError("rdv models pull expects a manifest id, not a file path")
        last = -1

        def progress(done: int, total: int) -> None:
            nonlocal last
            if done == total or done - last >= 32 * 1024 * 1024:
                last = done
                typer.echo(f"{selected}: {format_mib(done)} / {format_mib(total)}")

        path = manage_models.pull_model(store, selected, progress)
        typer.echo(f"ok {selected}: {path}")

    run(action)


@models_app.command("path")
def model_path(
    model_id: Annotated[
        str | None, typer.Argument(help="Manifest model id; defaults to RDV_MODEL.")
    ] = None,
    manifest: ManifestOption = None,
    model_dir: ModelDirOption = None,
) -> None:
    """Print the local path of a model, verifying that it exists."""

    def action() -> None:
        settings, store = _store(manifest, model_dir)
        path = manage_models.resolve_model(store, model_id or settings.model)
        typer.echo(str(path))

    run(action)


@models_app.command("verify")
def verify_model(
    model_id: Annotated[
        str | None, typer.Argument(help="Manifest model id; defaults to the manifest match.")
    ] = None,
    path: Annotated[
        Path | None, typer.Option("--path", help="File to verify; defaults to the cached model.")
    ] = None,
    manifest: ManifestOption = None,
    model_dir: ModelDirOption = None,
) -> None:
    """Verify a model file against the pinned sha256."""

    def action() -> None:
        settings, store = _store(manifest, model_dir)
        target = path
        if target is None:
            target = manage_models.resolve_model(store, model_id or settings.model)
        if model_id is not None and store.entry(model_id) is None:
            raise ModelError(f"Model {model_id!r} is not in the manifest")
        entry = manage_models.verify_model(store, target, model_id)
        typer.echo(f"ok {target}: sha256 {sha256_file(target)} (model {entry.id})")

    run(action)
