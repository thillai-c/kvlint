"""Terminal output.

Terse by default, detail behind `--verbose` (build plan section 9).
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from kvlint.models import SimResult
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
