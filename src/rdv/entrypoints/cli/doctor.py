"""``rdv doctor``: environment, providers, versions and model status (S1)."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from rdv import composition
from rdv.config.settings import Settings
from rdv.entrypoints.cli.console import run


def doctor(
    json_output: Annotated[
        bool, typer.Option("--json", help="Print a machine-readable report.")
    ] = False,
) -> None:
    """Report Python/OS/package versions, execution providers and model status."""

    def action() -> None:
        settings = Settings.from_env()
        report = composition.doctor_report(settings)
        if json_output:
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
            return
        model = report["model"]
        typer.echo(f"rdv {report['rdv_version']} on Python {report['python']}")
        typer.echo(
            f"{report['platform']['system']} {report['platform']['release']} "
            f"({report['platform']['machine']})"
        )
        typer.echo(f"ONNX Runtime {report['packages']['onnxruntime']}")
        typer.echo(f"Providers: {', '.join(report['providers']) or 'none'}")
        typer.echo(f"Device preference: {report['device']}")
        typer.echo(f"Model: {model['requested']} ({model['status']})")
        if model.get("path"):
            typer.echo(f"  path: {model['path']}")
        if model.get("error"):
            typer.echo(f"  hint: {model['error']}")
        typer.echo(f"Model dir: {report['model_dir']}")
        typer.echo(f"Data dir: {report['data_dir']}")
        typer.echo(f"Offline: {report['offline']}")

    run(action)
