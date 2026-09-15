"""Validate the vLLM simulator against a running vLLM server.

Metric names read from vllm-project/vllm @ main (2026-09-10),
`vllm/v1/metrics/loggers.py`:

    vllm:prefix_cache_queries_total  "Prefix cache queries, in terms of number of
                                      queried tokens."
    vllm:prefix_cache_hits_total     "Prefix cache hits, in terms of number of
                                      cached tokens."

Both are **counters in tokens**, which is the unit our simulator reports, so the
comparison needs no conversion. Because they are counters we can difference them
across the replay window and isolate our own traffic from anything else the
server handled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvlint.errors import KvlintError
from kvlint.live.prometheus import first_present, parse

# Newest spelling first. The `_total` suffix is the Prometheus convention; some
# versions expose the bare name.
QUERY_METRICS = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
HIT_METRICS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")


@dataclass(frozen=True)
class CacheCounters:
    queries: float
    hits: float

    def __sub__(self, other: CacheCounters) -> CacheCounters:
        return CacheCounters(self.queries - other.queries, self.hits - other.hits)

    @property
    def hit_rate(self) -> float | None:
        """None rather than zero when nothing was queried, so callers cannot
        mistake "no data" for "zero percent"."""
        if self.queries <= 0:
            return None
        return self.hits / self.queries


def scrape(client: Any) -> CacheCounters:
    """Read the prefix-cache counters from `/metrics`."""
    response = client.get("/metrics")
    if response.status_code >= 400:
        raise KvlintError(f"could not read /metrics: HTTP {response.status_code}")

    metrics = parse(response.text)
    queries = first_present(metrics, QUERY_METRICS)
    hits = first_present(metrics, HIT_METRICS)

    if queries is None or hits is None:
        raise KvlintError(
            "server did not expose vLLM prefix-cache counters. Looked for "
            f"{QUERY_METRICS[0]} and {HIT_METRICS[0]}. Is prefix caching enabled, "
            "and is this a vLLM server?"
        )
    return CacheCounters(queries=queries, hits=hits)
