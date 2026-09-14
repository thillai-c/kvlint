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
from kvlint.models import Report, SimResult, TokenizedRequest

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


class FixTarget(enum.StrEnum):
    system_end = "system_end"
    user_start = "user_start"


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
    from kvlint.errors import KvlintError
    from kvlint.ingest import load as load_log
    from kvlint.report.terminal import print_analysis
    from kvlint.sim import sglang_radix, vllm_blocks
    from kvlint.tokenize.renderer import tokenize_requests

    try:
        requests = load_log(logs, fmt.value)
        tokenized = tokenize_requests(model, requests)
    except KvlintError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(ExitCode.ERROR) from exc

    selected = engine or [EngineName.vllm, EngineName.sglang]
    results = []
    for name in selected:
        if name is EngineName.vllm:
            results.append(vllm_blocks.simulate(tokenized, block_size, kv_budget_blocks))
        else:
            results.append(sglang_radix.simulate(tokenized, kv_budget_tokens))

    print_analysis(
        console,
        results,
        {
            "path": str(logs),
            "format": fmt.value,
            "model": model,
            "requests": len(tokenized),
        },
        verbose=verbose,
    )

    if json_out is not None:
        report_model = Report(
            input_summary={
                "path": str(logs),
                "format": fmt.value,
                "model": model,
                "requests": len(tokenized),
            },
            sims=results,
            findings=[],
            version=__version__,
        )
        json_out.write_text(report_model.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Wrote JSON report to [bold]{json_out}[/]")

    if html_out is not None:
        err_console.print("[yellow]HTML reports arrive in M7.[/]")


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
    from kvlint.errors import KvlintError
    from kvlint.ingest import load as load_log
    from kvlint.lint.engine import exceeds
    from kvlint.lint.engine import run as run_rules
    from kvlint.report.terminal import print_findings
    from kvlint.sim import sglang_radix, vllm_blocks
    from kvlint.tokenize.renderer import tokenize_requests

    try:
        requests = load_log(logs, fmt.value)
        tokenized = tokenize_requests(model, requests)
    except KvlintError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(ExitCode.ERROR) from exc

    sims = [
        vllm_blocks.simulate(tokenized, block_size),
        sglang_radix.simulate(tokenized),
    ]
    findings = run_rules(requests, tokenized, block_size)

    console.print(
        f"Linted [bold]{len(tokenized)}[/] requests from [bold]{logs}[/] "
        f"(vLLM hit rate {sims[0].hit_rate * 100:.1f}%, "
        f"SGLang {sims[1].hit_rate * 100:.1f}%)"
    )
    console.print()
    print_findings(console, findings, verbose=verbose)

    if json_out is not None:
        report_model = Report(
            input_summary={
                "path": str(logs),
                "format": fmt.value,
                "model": model,
                "requests": len(tokenized),
            },
            sims=sims,
            findings=findings,
            version=__version__,
        )
        json_out.write_text(report_model.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Wrote JSON report to [bold]{json_out}[/]")

    if fail_on is not None and exceeds(findings, fail_on.value):
        raise typer.Exit(ExitCode.FINDINGS)


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
    block_size: BlockSizeOpt = 16,
    target: Annotated[
        FixTarget, typer.Option("--target", help="Where relocated dynamic lines land.")
    ] = FixTarget.system_end,
    verbose: VerboseOpt = False,
) -> None:
    """Apply auto-fixable findings and report before/after."""
    from kvlint.errors import KvlintError
    from kvlint.fix.estimate import Rates, deltas
    from kvlint.fix.rewriter import fix_requests
    from kvlint.fix.writer import write as write_log
    from kvlint.ingest import load as load_log
    from kvlint.models import BeforeAfter
    from kvlint.report.terminal import print_before_after
    from kvlint.sim import sglang_radix, vllm_blocks
    from kvlint.tokenize.renderer import tokenize_requests

    if apply and out is None:
        err_console.print("[red]--apply needs --out to say where to write the fixed log.[/]")
        raise typer.Exit(ExitCode.ERROR)

    try:
        requests = load_log(logs, fmt.value)
        before_tokens = tokenize_requests(model, requests)
        report = fix_requests(requests, target=target.value)
        after_tokens = tokenize_requests(model, report.requests)
    except KvlintError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(ExitCode.ERROR) from exc

    def simulate_both(tokens: list[TokenizedRequest]) -> dict[str, SimResult]:
        return {
            "vllm": vllm_blocks.simulate(tokens, block_size),
            "sglang": sglang_radix.simulate(tokens),
        }

    before = simulate_both(before_tokens)
    after = simulate_both(after_tokens)

    rates = Rates(
        price_per_1m_input=price_per_1m_input,
        cached_discount=cached_discount,
        prefill_ms_per_1k_tokens=prefill_ms_per_1k_tokens,
    )
    ttft_delta, cost_delta = deltas(before["vllm"], after["vllm"], rates)

    before_after = BeforeAfter(
        hit_rate_before=before["vllm"].hit_rate,
        hit_rate_after=after["vllm"].hit_rate,
        per_engine={
            engine: {"before": before[engine].hit_rate, "after": after[engine].hit_rate}
            for engine in ("vllm", "sglang")
        },
        est_ttft_delta_ms=ttft_delta,
        est_cost_delta=cost_delta,
    )

    print_before_after(console, before_after, report.changed, len(requests))

    if verbose:
        console.print()
        console.print(
            f"[dim]moved {report.moved_lines} volatile lines, "
            f"canonicalized JSON in {report.canonicalized_json} requests[/]"
        )

    if apply and out is not None:
        try:
            written = write_log(out, report.requests, fmt.value)
        except KvlintError as exc:
            err_console.print(f"[red]{exc}[/]")
            raise typer.Exit(ExitCode.ERROR) from exc
        console.print()
        console.print(f"Wrote [bold]{written}[/] fixed requests to [bold]{out}[/]")


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
