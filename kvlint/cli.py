"""Command-line interface (KVLINT_BUILD_PLAN.md §6).

Command bodies land in later milestones; the signatures are fixed here so the
surface stays stable and every milestone only fills in an implementation.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from kvlint import __version__

console = Console()
err_console = Console(stderr=True)


class ExitCode(enum.IntEnum):
    """Process exit codes (§6).

    NOTE: click/typer also use 2 for *usage* errors, so `FINDINGS` overlaps with
    "you typed the command wrong". Keeping the spec's value because CI configs
    depend on it; the two are distinguishable by stderr, and `--fail-on` is only
    meaningful once the command has actually parsed and run.
    """

    OK = 0
    ERROR = 1
    FINDINGS = 2


class LogFormat(enum.StrEnum):
    openai = "openai"
    sharegpt = "sharegpt"
    plain = "plain"


class EngineName(enum.StrEnum):
    vllm = "vllm"
    sglang = "sglang"


class Severity(enum.StrEnum):
    high = "high"
    medium = "medium"
    low = "low"


app = typer.Typer(
    name="kvlint",
    help="A linter for your prefix cache. Analyze LLM request logs offline for "
    "vLLM/SGLang KV-cache hit rate, and find what breaks it.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"kvlint {__version__}")
        raise typer.Exit(ExitCode.OK)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """kvlint: a linter for your prefix cache."""


def _not_yet(command: str, milestone: str) -> None:
    """Fail honestly rather than pretending a stub did something."""
    err_console.print(
        f"[yellow]`kvlint {command}` is not implemented yet[/] (scheduled for {milestone})."
    )
    raise typer.Exit(ExitCode.ERROR)


# ------------------------------------------------------------------ shared options

LogsArg = Annotated[Path, typer.Argument(help="Path to the request log (JSONL).")]
ModelOpt = Annotated[str, typer.Option("--model", help="HuggingFace model id for the tokenizer.")]
FormatOpt = Annotated[LogFormat, typer.Option("--format", help="Input log format.")]
EnginesOpt = Annotated[
    list[EngineName], typer.Option("--engine", help="Engine(s) to simulate. Repeatable.")
]
BlockSizeOpt = Annotated[int, typer.Option("--block-size", help="vLLM block size in tokens.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", help="Show per-request detail.")]


# ------------------------------------------------------------------ commands


@app.command()
def analyze(
    logs: LogsArg,
    model: ModelOpt,
    fmt: FormatOpt = LogFormat.openai,
    engine: EnginesOpt = [EngineName.vllm, EngineName.sglang],  # noqa: B006
    block_size: BlockSizeOpt = 16,
    kv_budget_blocks: Annotated[
        int | None,
        typer.Option("--kv-budget-blocks", help="vLLM KV budget in blocks. Omit for infinite."),
    ] = None,
    kv_budget_tokens: Annotated[
        int | None,
        typer.Option("--kv-budget-tokens", help="SGLang KV budget in tokens. Omit for infinite."),
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the JSON report here.")
    ] = None,
    html_out: Annotated[
        Path | None, typer.Option("--html", help="Write an HTML report here (needs [report]).")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Simulate prefix-cache hit rate for a log."""
    _not_yet("analyze", "M2/M3")


@app.command()
def lint(
    logs: LogsArg,
    model: ModelOpt,
    fmt: FormatOpt = LogFormat.openai,
    block_size: BlockSizeOpt = 16,
    fail_on: Annotated[
        Severity | None,
        typer.Option("--fail-on", help="Exit 2 if a finding at or above this severity fires."),
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the JSON report here.")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Analyze, then explain *why* the cache misses."""
    _not_yet("lint", "M4")


@app.command()
def fix(
    logs: LogsArg,
    model: ModelOpt,
    fmt: FormatOpt = LogFormat.openai,
    apply: Annotated[bool, typer.Option("--apply", help="Write the rewritten log.")] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Destination for --apply.")] = None,
    price_per_1m_input: Annotated[
        float | None, typer.Option("--price-per-1m-input", help="For the cost estimate.")
    ] = None,
    cached_discount: Annotated[
        float, typer.Option("--cached-discount", help="Discount on cached input tokens.")
    ] = 0.9,
    prefill_ms_per_1k_tokens: Annotated[
        float, typer.Option("--prefill-ms-per-1k-tokens", help="For the TTFT estimate.")
    ] = 40.0,
    verbose: VerboseOpt = False,
) -> None:
    """Apply auto-fixable findings and report before/after."""
    _not_yet("fix", "M5")


@app.command()
def validate(
    logs: LogsArg,
    model: ModelOpt,
    server: Annotated[str, typer.Option("--server", help="Base URL of the running engine.")],
    engine: Annotated[
        EngineName, typer.Option("--engine", help="Which engine is serving.")
    ] = EngineName.vllm,
    fmt: FormatOpt = LogFormat.openai,
    verbose: VerboseOpt = False,
) -> None:
    """Replay a log against a live server and compare simulated vs real hit rate."""
    _not_yet("validate", "M6")


@app.command()
def demo(
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the generated synthetic log here.")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Generate a synthetic log with injected anti-patterns, then lint and fix it."""
    _not_yet("demo", "M7")


@app.command()
def report(
    result: Annotated[Path, typer.Argument(help="A JSON report produced by analyze/lint/fix.")],
    html_out: Annotated[Path, typer.Option("--html", help="Where to write the HTML report.")],
) -> None:
    """Render a saved JSON report as HTML."""
    _not_yet("report", "M7")


if __name__ == "__main__":  # pragma: no cover
    app()
