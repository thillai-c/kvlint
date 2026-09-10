"""The simulator protocol shared by the vLLM and SGLang backends.

Both engines answer the same question, "how much of this prompt was already in
the cache", but with different data structures and different granularity. Keeping
them behind one protocol is what lets `analyze` print them side by side, and lets
the lint layer stay engine-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kvlint.models import PerRequestResult, SimResult, TokenizedRequest


@runtime_checkable
class Simulator(Protocol):
    """A cache simulator fed requests in arrival order.

    Implementations are stateful: `feed` mutates the cache, so the result for a
    request depends on every request before it. That is the point.
    """

    engine: str

    def feed(self, request: TokenizedRequest) -> PerRequestResult:
        """Simulate one request and record its blocks in the cache."""
        ...

    def config(self) -> dict[str, Any]:
        """The knobs this run used, for the report's reproducibility record."""
        ...


def aggregate(
    engine: str,
    config: dict[str, Any],
    actual: list[PerRequestResult],
    ideal: list[PerRequestResult],
    block_size: int | None,
) -> SimResult:
    """Combine a budgeted pass and an infinite-budget pass into one result.

    The gap between the two hit rates is the cost of memory pressure specifically,
    as distinct from the cost of badly built prompts. Separating those is the whole
    reason for running twice: a user with a 20% hit rate needs to know whether to
    buy more GPU or fix their template.
    """
    if len(actual) != len(ideal):  # pragma: no cover
        raise ValueError("actual and ideal passes disagree on request count")

    total_prompt = sum(r.prompt_tokens for r in actual)
    total_cached = sum(r.cached_tokens for r in actual)
    total_ideal = sum(r.cached_tokens for r in ideal)

    # Attribute the shortfall per request so the report can point at the ones that
    # actually got evicted, rather than only quoting a corpus-wide delta.
    merged: list[PerRequestResult] = []
    for actual_result, ideal_result in zip(actual, ideal, strict=True):
        merged.append(
            actual_result.model_copy(
                update={
                    "evicted_hit_loss": max(
                        0, ideal_result.cached_tokens - actual_result.cached_tokens
                    )
                }
            )
        )

    return SimResult(
        engine=engine,  # type: ignore[arg-type]
        config=config,
        per_request=merged,
        hit_rate=(total_cached / total_prompt) if total_prompt else 0.0,
        ideal_hit_rate=(total_ideal / total_prompt) if total_prompt else 0.0,
        block_size=block_size,
    )
