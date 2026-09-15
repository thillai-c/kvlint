"""Compare a simulated hit rate against a live server's own counters.

This is the milestone that decides whether anything else in kvlint is worth
believing. Every number the tool prints elsewhere is a model of an engine; here
the engine speaks for itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kvlint.live import vllm_metrics
from kvlint.live.replay import TokenCheck, replay, reset_prefix_cache
from kvlint.models import Request, TokenizedRequest
from kvlint.sim import vllm_blocks

ACCEPTABLE_ERROR_PP = 3.0
"""Build plan M6: simulated and real must agree within 3 percentage points."""


@dataclass
class ValidationResult:
    engine: str
    requests: int
    simulated_hit_rate: float
    real_hit_rate: float | None
    token_mismatches: list[TokenCheck] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cache_was_reset: bool = False
    exact: bool = True
    """False when the engine only exposes a gauge, so the comparison is indicative."""

    @property
    def abs_error_pp(self) -> float | None:
        if self.real_hit_rate is None:
            return None
        return abs(self.simulated_hit_rate - self.real_hit_rate) * 100.0

    @property
    def within_tolerance(self) -> bool | None:
        error = self.abs_error_pp
        return None if error is None else error <= ACCEPTABLE_ERROR_PP

    @property
    def tokens_agree(self) -> bool:
        return not self.token_mismatches


def validate_vllm(
    client: Any,
    requests: list[Request],
    tokenized: list[TokenizedRequest],
    model: str,
    block_size: int = 16,
    reset_cache: bool = True,
) -> ValidationResult:
    """Replay against vLLM and difference its prefix-cache counters."""
    notes: list[str] = []

    was_reset = reset_prefix_cache(client) if reset_cache else False
    if reset_cache and not was_reset:
        notes.append(
            "could not reset the prefix cache, so earlier traffic may inflate the real hit rate"
        )

    before = vllm_metrics.scrape(client)
    outcome = replay(client, requests, tokenized, model)
    after = vllm_metrics.scrape(client)

    window = after - before
    real = window.hit_rate
    if real is None:
        notes.append(
            "the server reported no prefix-cache queries during the replay, so "
            "there is nothing to compare against"
        )

    simulated = vllm_blocks.simulate(tokenized, block_size).hit_rate

    return ValidationResult(
        engine="vllm",
        requests=len(requests),
        simulated_hit_rate=simulated,
        real_hit_rate=real,
        token_mismatches=outcome.mismatches,
        failures=outcome.failures,
        notes=notes,
        cache_was_reset=was_reset,
        exact=True,
    )
