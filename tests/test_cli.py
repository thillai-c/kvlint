"""CLI surface tests (M0 acceptance: `kvlint --help` works, all six commands present)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from kvlint import __version__
from kvlint.cli import ExitCode, app

runner = CliRunner()

COMMANDS = ["analyze", "lint", "fix", "validate", "demo", "report"]


def test_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == ExitCode.OK


@pytest.mark.parametrize("command", COMMANDS)
def test_help_lists_every_command(command: str) -> None:
    result = runner.invoke(app, ["--help"])
    assert command in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_has_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == ExitCode.OK


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == ExitCode.OK
    assert __version__ in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_stubs_fail_loudly_rather_than_silently_succeeding(command: str) -> None:
    """A stub must never exit 0 and let a caller think it did the work."""
    args = {
        "analyze": ["analyze", "x.jsonl", "--model", "m"],
        "lint": ["lint", "x.jsonl", "--model", "m"],
        "fix": ["fix", "x.jsonl", "--model", "m"],
        "validate": ["validate", "x.jsonl", "--model", "m", "--server", "http://localhost:8000"],
        "demo": ["demo"],
        "report": ["report", "r.json", "--html", "out.html"],
    }[command]
    result = runner.invoke(app, args)
    assert result.exit_code == ExitCode.ERROR


def test_exit_codes_match_spec() -> None:
    assert (ExitCode.OK, ExitCode.ERROR, ExitCode.FINDINGS) == (0, 1, 2)
