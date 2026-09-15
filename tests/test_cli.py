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


def test_no_command_is_a_stub_any_more() -> None:
    """Every command in the spec is implemented.

    This replaced a test that asserted the opposite. While commands were being
    built, a stub exiting 0 would have let a caller believe work happened; now
    the risk is the reverse, a command quietly regressing to a stub.
    """
    import inspect

    from kvlint import cli
    from kvlint.cli import _not_yet  # noqa: F401  (still used for future commands)

    source = inspect.getsource(cli)
    for command in COMMANDS:
        assert f'_not_yet("{command}"' not in source, f"{command} is still a stub"


@pytest.mark.parametrize(
    "command",
    ["analyze", "lint", "fix", "validate", "report"],
)
def test_commands_fail_cleanly_on_a_missing_file(command: str) -> None:
    """An error must exit non-zero rather than tracebacking at the user."""
    args = {
        "analyze": ["analyze", "nope.jsonl", "--model", "m"],
        "lint": ["lint", "nope.jsonl", "--model", "m"],
        "fix": ["fix", "nope.jsonl", "--model", "m"],
        "validate": ["validate", "nope.jsonl", "--model", "m", "--server", "http://localhost:1"],
        "report": ["report", "nope.json", "--html", "out.html"],
    }[command]
    result = runner.invoke(app, args)
    assert result.exit_code != ExitCode.OK
    assert "Traceback" not in result.output


def test_exit_codes_match_spec() -> None:
    assert (ExitCode.OK, ExitCode.ERROR, ExitCode.FINDINGS) == (0, 1, 2)
