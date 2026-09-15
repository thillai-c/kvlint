"""Derived metrics over a completed simulation.

Pure functions over `SimResult`, kept out of the simulators so both engines
report the same way and the report layer has one place to read from.
"""

from __future__ import annotations

from collections import Counter

from kvlint.models import SimResult, TokenizedRequest


def cached_token_fraction(result: SimResult) -> float:
    """Share of prompt tokens served from cache. Same as `hit_rate`, named for reports."""
    return result.hit_rate


def eviction_loss_fraction(result: SimResult) -> float:
    """Share of prompt tokens that would have hit but were evicted first.

    This is the number that answers "would more KV memory help", as opposed to
    "is my prompt template broken".
    """
    total = sum(r.prompt_tokens for r in result.per_request)
    if not total:
        return 0.0
    return sum(r.evicted_hit_loss for r in result.per_request) / total


def miss_count(result: SimResult) -> int:
    """Requests that diverged from the best prior prefix."""
    return sum(1 for r in result.per_request if r.divergence_token_idx is not None)


def full_hit_count(result: SimResult) -> int:
    """Requests served entirely from cache."""
    return sum(
        1 for r in result.per_request if r.prompt_tokens and r.cached_tokens >= r.prompt_tokens
    )


def divergence_histogram(result: SimResult, bucket_size: int = 64) -> dict[int, int]:
    """Where prompts stop agreeing, bucketed by token position.

    A spike in the first bucket means the divergence is near the top of the
    prompt, which is the expensive place for it to be: everything after is lost.
    """
    if bucket_size < 1:
        raise ValueError("bucket_size must be at least 1")

    histogram: Counter[int] = Counter()
    for record in result.per_request:
        if record.divergence_token_idx is None:
            continue
        histogram[(record.divergence_token_idx // bucket_size) * bucket_size] += 1
    return dict(sorted(histogram.items()))


def summarize(result: SimResult) -> dict[str, float | int | str | None]:
    """A flat summary row, for the side-by-side engine table."""
    return {
        "engine": result.engine,
        "hit_rate": result.hit_rate,
        "ideal_hit_rate": result.ideal_hit_rate,
        "eviction_loss": eviction_loss_fraction(result),
        "requests": len(result.per_request),
        "full_hits": full_hit_count(result),
        "misses": miss_count(result),
        "prompt_tokens": sum(r.prompt_tokens for r in result.per_request),
        "cached_tokens": sum(r.cached_tokens for r in result.per_request),
        "block_size": result.block_size,
    }


def hit_rate_curve(
    tokenized: list[TokenizedRequest],
    budgets: list[int],
    block_size: int = 16,
) -> list[tuple[int, float]]:
    """Hit rate at each KV budget, for the memory-pressure chart.

    The shape is the answer to "would more GPU help". A curve that rises steeply
    means the workload is memory-limited; one that is already flat means the
    prompts are the problem and more memory changes nothing.
    """
    from kvlint.sim import vllm_blocks

    return [
        (budget, vllm_blocks.simulate(tokenized, block_size, budget).hit_rate)
        for budget in sorted(budgets)
    ]


def suggest_budgets(
    tokenized: list[TokenizedRequest], block_size: int = 16, points: int = 8
) -> list[int]:
    """A spread of budgets from tight to roomy, based on the log's own size."""
    total_blocks = sum(len(t.token_ids) // block_size for t in tokenized)
    if total_blocks < 2:
        return []
    step = max(1, total_blocks // points)
    return sorted({max(1, i) for i in range(step, total_blocks + step, step)})
