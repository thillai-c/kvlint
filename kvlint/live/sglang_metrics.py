"""Validate the SGLang simulator against a running SGLang server. Best effort.

Metric read from the SGLang production-metrics documentation (2026-09-14):

    sglang:cache_hit_rate   gauge   "The cache hit rate"

This is a **gauge**, not a counter pair, and that difference matters. With vLLM
we difference two counters across the replay window and isolate exactly our own
traffic. A gauge only reports the server's own running average at the moment we
read it, so:

- It includes any traffic the server handled before us.
- It cannot be windowed, so a busy shared server makes it meaningless.
- Its averaging basis is the server's, not ours.

So this comparison is indicative rather than exact, and the report says so. The
spec calls SGLang validation best-effort for exactly this reason. Treat a close
match as encouraging and a mismatch as inconclusive rather than as a simulator
bug, unless the server was idle and the cache was reset first.
"""

from __future__ import annotations

from typing import Any

from kvlint.errors import KvlintError
from kvlint.live.prometheus import first_present, parse

HIT_RATE_METRICS = ("sglang:cache_hit_rate", "sglang:prefix_cache_hit_rate")


def scrape_hit_rate(client: Any) -> float:
    """Read the cache-hit-rate gauge, normalized to a 0..1 fraction."""
    response = client.get("/metrics")
    if response.status_code >= 400:
        raise KvlintError(f"could not read /metrics: HTTP {response.status_code}")

    value = first_present(parse(response.text), HIT_RATE_METRICS)
    if value is None:
        raise KvlintError(
            f"server did not expose {HIT_RATE_METRICS[0]}. Is this an SGLang server "
            "started with --enable-metrics?"
        )

    # Some builds report a percentage rather than a fraction.
    return value / 100.0 if value > 1.0 else value
