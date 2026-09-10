"""The simulator protocol shared by the vLLM and SGLang backends.

Both engines answer the same question, "how much of this prompt was already in
the cache", but with different data structures and different granularity. Keeping
them behind one protocol is what lets `analyze` print them side by side, and lets
the lint layer stay engine-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kvlint.models import PerRequestResult, TokenizedRequest


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
