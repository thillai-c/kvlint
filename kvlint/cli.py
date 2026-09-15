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
from rich.markup import escape

from kvlint import __version__
from kvlint.models import Report, SimResult, TokenizedRequest

DEMO_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
"""Small, ungated, and the model the simulator was validated against."""

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


def _fail(message: str, code: ExitCode = ExitCode.ERROR) -> typer.Exit:
    """Print an error and return the exception to raise.

    Escapes the message first. Rich treats square brackets as markup, so an
    unescaped error would silently swallow parts of itself: the plotly hint
    `pip install 'kvlint[report]'` printed as `pip install 'kvlint'`, which is
    advice that does not work.
    """
    colour = "yellow" if code is ExitCode.ERROR and message.startswith("HTML") else "red"
    err_console.print(f"[{colour}]{escape(message)}[/]")
    return typer.Exit(code)


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
    from kvlint.report.json_out import write as json_write
    from kvlint.report.terminal import print_analysis
    from kvlint.sim import sglang_radix, vllm_blocks
    from kvlint.tokenize.renderer import tokenize_requests

    try:
        requests = load_log(logs, fmt.value)
        tokenized = tokenize_requests(model, requests)
    except KvlintError as exc:
        raise _fail(str(exc)) from exc

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
        json_write(json_out, report_model)
        console.print(f"Wrote JSON report to [bold]{json_out}[/]")

    if html_out is not None:
        from kvlint.report import html
        from kvlint.sim.metrics import hit_rate_curve, suggest_budgets

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
        try:
            html.write(
                html_out,
                report_model,
                hit_rate_curve(tokenized, suggest_budgets(tokenized, block_size), block_size),
            )
        except KvlintError as exc:
            raise _fail(str(exc)) from exc
        console.print(f"Wrote the HTML report to [bold]{html_out}[/]")


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
    from kvlint.report.json_out import write as json_write
    from kvlint.report.terminal import print_findings
    from kvlint.sim import sglang_radix, vllm_blocks
    from kvlint.tokenize.renderer import tokenize_requests

    try:
        requests = load_log(logs, fmt.value)
        tokenized = tokenize_requests(model, requests)
    except KvlintError as exc:
        raise _fail(str(exc)) from exc

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
        json_write(json_out, report_model)
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
        raise _fail("--apply needs --out to say where to write the fixed log.")

    try:
        requests = load_log(logs, fmt.value)
        before_tokens = tokenize_requests(model, requests)
        report = fix_requests(requests, target=target.value)
        after_tokens = tokenize_requests(model, report.requests)
    except KvlintError as exc:
        raise _fail(str(exc)) from exc

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
            raise _fail(str(exc)) from exc
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
    block_size: BlockSizeOpt = 16,
    reset_cache: Annotated[
        bool,
        typer.Option(
            "--reset-cache/--no-reset-cache",
            help="Clear the server prefix cache before replaying.",
        ),
    ] = True,
    verbose: VerboseOpt = False,
) -> None:
    """Replay a log against a live server and compare simulated vs real hit rate."""
    from kvlint.errors import KvlintError
    from kvlint.ingest import load as load_log
    from kvlint.live.replay import build_client
    from kvlint.live.validate import validate_sglang, validate_vllm
    from kvlint.report.terminal import print_validation
    from kvlint.tokenize.renderer import tokenize_requests

    try:
        requests = load_log(logs, fmt.value)
        tokenized = tokenize_requests(model, requests)
    except KvlintError as exc:
        raise _fail(str(exc)) from exc

    console.print(
        f"Replaying [bold]{len(requests)}[/] requests against [bold]{server}[/] "
        f"({engine.value}, model {model})"
    )
    console.print()

    try:
        client = build_client(server)
        with client:
            if engine is EngineName.vllm:
                result = validate_vllm(
                    client, requests, tokenized, model, block_size, reset_cache=reset_cache
                )
            else:
                result = validate_sglang(
                    client, requests, tokenized, model, reset_cache=reset_cache
                )
    except KvlintError as exc:
        raise _fail(str(exc)) from exc
    except Exception as exc:
        raise _fail(f"could not reach {server}: {exc}") from exc

    print_validation(console, result)

    # A token mismatch means our rendering differs from the server's, which makes
    # every hit rate suspect. That is a failure, not a note.
    if not result.tokens_agree or result.within_tolerance is False:
        raise typer.Exit(ExitCode.FINDINGS)


@app.command()
def demo(
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the generated synthetic log here.")
    ] = None,
    html_out: Annotated[
        Path | None, typer.Option("--html", help="Write an HTML report here (needs [report]).")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help="Tokenizer to use for the demo.")
    ] = DEMO_MODEL,
    requests: Annotated[int, typer.Option("--requests", help="How many to generate.")] = 60,
    verbose: VerboseOpt = False,
) -> None:
    """Generate a synthetic log with injected anti-patterns, then lint and fix it."""
    from kvlint.demo.synthetic import generate
    from kvlint.errors import KvlintError
    from kvlint.fix.rewriter import fix_requests
    from kvlint.fix.writer import write as write_log
    from kvlint.lint.engine import run as run_rules
    from kvlint.report.terminal import print_findings
    from kvlint.sim import sglang_radix, vllm_blocks
    from kvlint.tokenize.renderer import tokenize_requests

    console.print(f"Generating a {requests} request support workload with known problems.")
    console.print(f"Tokenizing with [bold]{model}[/] (downloads once, tokenizer only).")
    console.print()

    dirty = generate(count=requests)
    try:
        before_tokens = tokenize_requests(model, dirty)
    except KvlintError as exc:
        raise _fail(str(exc)) from exc

    fixed = fix_requests(dirty).requests
    after_tokens = tokenize_requests(model, fixed)

    before = {
        "vllm": vllm_blocks.simulate(before_tokens),
        "sglang": sglang_radix.simulate(before_tokens),
    }
    after = {
        "vllm": vllm_blocks.simulate(after_tokens),
        "sglang": sglang_radix.simulate(after_tokens),
    }
    findings = run_rules(dirty, before_tokens)

    print_findings(console, findings, verbose=verbose)
    console.print()

    # The headline. One line, the number first, because it is the whole point.
    console.print(
        f"[bold]Hit rate {before['vllm'].hit_rate * 100:.1f}% to "
        f"{after['vllm'].hit_rate * 100:.1f}% after fixing {len(findings)} issues "
        f"(vLLM), {before['sglang'].hit_rate * 100:.1f}% to "
        f"{after['sglang'].hit_rate * 100:.1f}% (SGLang).[/]"
    )

    if out is not None:
        write_log(out, dirty, "openai")
        console.print(f"Wrote the demo log to [bold]{out}[/]")

    if html_out is not None:
        from kvlint.models import BeforeAfter
        from kvlint.report import html
        from kvlint.sim.metrics import hit_rate_curve, suggest_budgets

        report_model = Report(
            input_summary={
                "path": "synthetic demo log",
                "format": "openai",
                "model": model,
                "requests": len(dirty),
            },
            sims=[before["vllm"], before["sglang"]],
            findings=findings,
            before_after=BeforeAfter(
                hit_rate_before=before["vllm"].hit_rate,
                hit_rate_after=after["vllm"].hit_rate,
                per_engine={
                    e: {"before": before[e].hit_rate, "after": after[e].hit_rate}
                    for e in ("vllm", "sglang")
                },
            ),
            version=__version__,
        )
        # The curve is drawn on the *fixed* log. On the broken one almost nothing
        # is cacheable, so the line is flat at every budget and says only "your
        # prompts are the problem", which the other two charts already said. On
        # the fixed log it answers the next question: how much KV memory this
        # workload actually needs.
        curve = hit_rate_curve(after_tokens, suggest_budgets(after_tokens))
        try:
            html.write(html_out, report_model, curve)
        except KvlintError as exc:
            raise _fail(str(exc)) from exc
        console.print(f"Wrote the HTML report to [bold]{html_out}[/]")


@app.command()
def report(
    result: Annotated[Path, typer.Argument(help="A JSON report produced by analyze/lint/fix.")],
    html_out: Annotated[Path, typer.Option("--html", help="Where to write the HTML report.")],
) -> None:
    """Render a saved JSON report as HTML."""
    import json

    from kvlint.errors import KvlintError
    from kvlint.report import html

    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _fail(f"{result}: file not found") from None
    except json.JSONDecodeError as exc:
        raise _fail(f"{result}: not valid JSON: {exc.msg}") from exc

    try:
        report_model = Report.model_validate(payload)
    except Exception as exc:
        raise _fail(f"{result}: not a kvlint report: {exc}") from exc

    try:
        html.write(html_out, report_model)
    except KvlintError as exc:
        raise _fail(str(exc)) from exc

    console.print(f"Wrote the HTML report to [bold]{html_out}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
