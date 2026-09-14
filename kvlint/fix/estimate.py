"""Time-to-first-token and cost estimates.

Every number here is an **estimate** derived from rates the user supplies, not a
measurement. Build plan section 5.5 requires them to be labelled as such wherever
they appear, and the report layer does that.

The model is deliberately crude: prefill cost is proportional to uncached tokens,
and cached tokens bill at a discount. Real TTFT also depends on batching,
scheduling, chunked prefill and hardware, none of which kvlint simulates.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvlint.models import SimResult

DEFAULT_PREFILL_MS_PER_1K = 40.0
DEFAULT_CACHED_DISCOUNT = 0.9


@dataclass(frozen=True)
class Rates:
    """User-supplied assumptions behind the estimates."""

    price_per_1m_input: float | None = None
    cached_discount: float = DEFAULT_CACHED_DISCOUNT
    prefill_ms_per_1k_tokens: float = DEFAULT_PREFILL_MS_PER_1K

    def __post_init__(self) -> None:
        if not 0.0 <= self.cached_discount <= 1.0:
            raise ValueError("cached_discount must be between 0 and 1")
        if self.prefill_ms_per_1k_tokens < 0:
            raise ValueError("prefill_ms_per_1k_tokens cannot be negative")


def _totals(result: SimResult) -> tuple[int, int]:
    prompt = sum(r.prompt_tokens for r in result.per_request)
    cached = sum(r.cached_tokens for r in result.per_request)
    return prompt, cached


def mean_ttft_ms(result: SimResult, rates: Rates) -> float:
    """Average prefill time per request, in milliseconds.

    Cached tokens are treated as free: they skip prefill entirely, which is the
    whole point of a prefix cache.
    """
    if not result.per_request:
        return 0.0
    prompt, cached = _totals(result)
    uncached = max(0, prompt - cached)
    total_ms = (uncached / 1000.0) * rates.prefill_ms_per_1k_tokens
    return total_ms / len(result.per_request)


def total_cost(result: SimResult, rates: Rates) -> float | None:
    """Input-token cost for the whole log, or None without a price."""
    if rates.price_per_1m_input is None:
        return None
    prompt, cached = _totals(result)
    uncached = max(0, prompt - cached)
    billable = uncached + cached * (1.0 - rates.cached_discount)
    return billable / 1_000_000.0 * rates.price_per_1m_input


def deltas(before: SimResult, after: SimResult, rates: Rates) -> tuple[float, float | None]:
    """How much the fix saved: (TTFT ms per request, total cost).

    Positive means the fix helped.
    """
    ttft = mean_ttft_ms(before, rates) - mean_ttft_ms(after, rates)

    cost_before = total_cost(before, rates)
    cost_after = total_cost(after, rates)
    cost = None if cost_before is None or cost_after is None else cost_before - cost_after
    return ttft, cost
