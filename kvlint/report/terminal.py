"""Terminal output.

Terse by default, detail behind `--verbose` (build plan section 9).
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from kvlint.models import BeforeAfter, LintFinding, SimResult
from kvlint.sim.metrics import divergence_histogram, summarize


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def engine_comparison(results: list[SimResult]) -> Table:
    """Side-by-side hit rates, one column per engine.

    The two engines are not expected to agree: vLLM rounds reuse down to whole
    16-token blocks while SGLang matches at token granularity, so SGLang is
    normally the higher of the two. A large gap is a signal in itself, usually
    that prompts diverge just after a block boundary.
    """
    table = Table(title="Prefix cache hit rate", title_justify="left")
    table.add_column("Metric", style="bold")
    for result in results:
        table.add_column(result.engine, justify="right")

    rows = [summarize(result) for result in results]

    def add(label: str, key: str, fmt: str = "plain") -> None:
        cells = []
        for row in rows:
            value = row[key]
            if value is None:
                cells.append("-")
            elif fmt == "pct" and isinstance(value, float):
                cells.append(_pct(value))
            else:
                cells.append(f"{value:,}" if isinstance(value, int) else str(value))
        table.add_row(label, *cells)

    add("Hit rate", "hit_rate", "pct")
    add("Ideal hit rate", "ideal_hit_rate", "pct")
    add("Lost to eviction", "eviction_loss", "pct")
    add("Prompt tokens", "prompt_tokens")
    add("Cached tokens", "cached_tokens")
    add("Requests", "requests")
    add("Full hits", "full_hits")
    add("Diverged", "misses")
    add("Block size", "block_size")
    return table


def divergence_table(result: SimResult, bucket_size: int = 64) -> Table:
    """Where in the prompt requests stop matching."""
    table = Table(title=f"Divergence position ({result.engine})", title_justify="left")
    table.add_column("Token range", justify="right")
    table.add_column("Requests", justify="right")

    histogram = divergence_histogram(result, bucket_size)
    for start, count in histogram.items():
        table.add_row(f"{start:,} to {start + bucket_size - 1:,}", f"{count:,}")
    if not histogram:
        table.add_row("-", "0")
    return table


def print_analysis(
    console: Console,
    results: list[SimResult],
    input_summary: dict[str, object],
    verbose: bool = False,
) -> None:
    console.print(
        f"Analyzed [bold]{input_summary['requests']}[/] requests "
        f"from [bold]{input_summary['path']}[/] "
        f"({input_summary['format']} format, model {input_summary['model']})"
    )
    console.print()
    console.print(engine_comparison(results))

    if verbose:
        for result in results:
            console.print()
            console.print(divergence_table(result))


_SEVERITY_STYLE = {"high": "red", "medium": "yellow", "low": "dim"}


def findings_table(findings: list[LintFinding]) -> Table:
    """One row per rule that fired, most expensive first."""
    table = Table(title="Findings", title_justify="left")
    table.add_column("Severity")
    table.add_column("Rule", style="bold")
    table.add_column("Affected", justify="right")
    table.add_column("Tokens lost", justify="right")
    table.add_column("Fix", justify="center")
    table.add_column("Evidence", overflow="fold")

    for finding in findings:
        style = _SEVERITY_STYLE.get(finding.severity, "")
        table.add_row(
            f"[{style}]{finding.severity}[/]" if style else finding.severity,
            finding.rule_id,
            _pct(finding.affected_fraction),
            f"{finding.est_tokens_lost:,}",
            "auto" if finding.auto_fixable else "manual",
            finding.evidence,
        )
    return table


def print_findings(console: Console, findings: list[LintFinding], verbose: bool = False) -> None:
    if not findings:
        console.print("[green]No cache-breaking patterns found.[/]")
        return

    console.print(findings_table(findings))
    console.print()
    console.print(
        "[dim]Tokens lost is an upper bound and rules can overlap, so the column does not sum.[/]"
    )

    if verbose:
        for finding in findings:
            console.print()
            console.print(f"[bold]{finding.rule_id}[/]: {finding.suggestion}")
            shown = ", ".join(finding.affected_request_ids[:5])
            more = len(finding.affected_request_ids) - 5
            console.print(f"  requests: {shown}" + (f" (+{more} more)" if more > 0 else ""))


def before_after_table(before_after: BeforeAfter) -> Table:
    """Hit rate for each engine, before and after the rewrite."""
    table = Table(title="Before and after", title_justify="left")
    table.add_column("Engine", style="bold")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Change", justify="right")

    for engine, rates in before_after.per_engine.items():
        delta = rates["after"] - rates["before"]
        arrow = "green" if delta > 0 else "red" if delta < 0 else "dim"
        table.add_row(
            engine,
            _pct(rates["before"]),
            _pct(rates["after"]),
            f"[{arrow}]{delta * 100:+.1f} pp[/]",
        )
    return table


def print_before_after(
    console: Console,
    before_after: BeforeAfter,
    changed: int,
    total: int,
) -> None:
    console.print(f"Rewrote [bold]{changed}[/] of {total} requests.")
    console.print()
    console.print(before_after_table(before_after))

    if before_after.est_ttft_delta_ms is None and before_after.est_cost_delta is None:
        return

    console.print()
    if before_after.est_ttft_delta_ms is not None:
        console.print(
            f"Estimated TTFT change: [bold]{before_after.est_ttft_delta_ms:+.1f} ms[/] per request"
        )
    if before_after.est_cost_delta is not None:
        console.print(f"Estimated cost change: [bold]{before_after.est_cost_delta:+.4f}[/]")
    console.print(
        "[dim]TTFT and cost are estimates from the rates you supplied, not measurements.[/]"
    )
