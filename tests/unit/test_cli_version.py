"""The CLI surface is a contract: name, version, and command list."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from rdv.entrypoints.cli.app import app

runner = CliRunner()

EXPECTED_COMMANDS = {
    "benchmark",
    "data",
    "detect",
    "doctor",
    "evaluate",
    "export",
    "models",
    "samples",
    "serve",
    "train",
}


def test_version_prints_name_and_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "rdv 0.1.0"


def test_help_lists_exactly_the_documented_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    listed: set[str] = set()
    for line in result.stdout.splitlines():
        row = line.strip().strip("│").strip()
        match = re.match(r"([a-z][a-z0-9-]*)\s{2,}\S", row)
        if match:
            listed.add(match.group(1))

    assert listed == EXPECTED_COMMANDS
