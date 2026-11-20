"""The CLI surface is a contract: name, version, and command list."""

from __future__ import annotations

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
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0
    registered = {
        command.name or command.callback.__name__
        for command in app.registered_commands
        if command.callback is not None
    }
    registered |= {group.name for group in app.registered_groups if group.name}
    assert registered == EXPECTED_COMMANDS
    # The rendered help can be a Rich table or plain text depending on the
    # terminal, so assert on the names instead of the table layout.
    for name in EXPECTED_COMMANDS:
        assert name in result.stdout
