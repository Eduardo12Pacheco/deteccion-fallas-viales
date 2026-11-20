"""Command-line interface for road-damage-vision."""

import typer

from rdv import __version__
from rdv.entrypoints.cli.benchmark import benchmark as benchmark_command
from rdv.entrypoints.cli.data import data_app
from rdv.entrypoints.cli.detect import detect as detect_command
from rdv.entrypoints.cli.doctor import doctor as doctor_command
from rdv.entrypoints.cli.evaluate import evaluate as evaluate_command
from rdv.entrypoints.cli.export import export as export_command
from rdv.entrypoints.cli.models import models_app
from rdv.entrypoints.cli.samples import samples as samples_command
from rdv.entrypoints.cli.serve import serve as serve_command
from rdv.entrypoints.cli.train import train as train_command

app = typer.Typer(
    name="rdv",
    help="Detect road damage in images and video.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rdv {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Detect road damage in images and video."""


app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.command()(benchmark_command)
app.command()(detect_command)
app.command()(doctor_command)
app.command()(evaluate_command)
app.command()(export_command)
app.command()(samples_command)
app.command()(serve_command)
app.command()(train_command)


def main() -> None:
    """Console-script entry point."""
    app()
