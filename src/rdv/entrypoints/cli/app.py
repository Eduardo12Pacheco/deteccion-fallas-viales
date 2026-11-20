"""Command-line interface for road-damage-vision."""

import typer

from rdv import __version__
from rdv.entrypoints.cli.data import data_app
from rdv.entrypoints.cli.detect import detect as detect_command
from rdv.entrypoints.cli.doctor import doctor as doctor_command
from rdv.entrypoints.cli.models import models_app
from rdv.entrypoints.cli.samples import samples as samples_command

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


@app.command()
def evaluate() -> None:
    """Evaluate a model on a labeled split."""


@app.command()
def train() -> None:
    """Fine-tune the detector from a configuration file."""


@app.command()
def export() -> None:
    """Export a checkpoint to ONNX."""


@app.command()
def serve() -> None:
    """Start the local web app."""


app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.command()(detect_command)
app.command()(doctor_command)
app.command()(samples_command)


@app.command()
def benchmark() -> None:
    """Measure inference latency and throughput."""


def main() -> None:
    """Console-script entry point."""
    app()
