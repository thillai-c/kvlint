"""Minimal Prometheus text-format parser.

Engines expose `/metrics` in the standard exposition format. We need a handful of
values from it and nothing else, so parsing it directly avoids a dependency on
`prometheus_client` just to read four numbers.
"""

from __future__ import annotations

from collections import defaultdict


def parse(text: str) -> dict[str, float]:
    """Sum each metric across its label sets.

    A server with several models or several engine cores reports one line per
    label combination. We want the total, since a replay touches whichever core
    the scheduler picked.
    """
    totals: dict[str, float] = defaultdict(float)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        name, _, remainder = line.partition("{")
        if remainder:
            # Labelled: name{a="b",c="d"} 1.0
            _, _, value_part = remainder.partition("}")
        else:
            # Unlabelled: name 1.0
            name, _, value_part = line.partition(" ")

        try:
            # Parse before touching `totals`: augmented assignment on a defaultdict
            # would create the key first, leaving a phantom 0.0 behind on failure.
            value = float(value_part.strip().split()[0])
        except (ValueError, IndexError):
            # Exemplars, NaN, and malformed lines are not worth failing over.
            continue
        totals[name.strip()] += value

    return dict(totals)


def first_present(metrics: dict[str, float], names: tuple[str, ...]) -> float | None:
    """Value of the first metric name that exists.

    Metric names drift between engine versions, so each call site passes the
    spellings it knows about, newest first, rather than hardcoding one.
    """
    for name in names:
        if name in metrics:
            return metrics[name]
    return None
