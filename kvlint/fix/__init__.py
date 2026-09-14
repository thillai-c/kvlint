"""Auto-fix: rewrite logs to remove cache-breaking patterns."""

from __future__ import annotations

from kvlint.fix.estimate import Rates, deltas, mean_ttft_ms, total_cost
from kvlint.fix.rewriter import FixReport, canonicalize, fix_requests, normalize_text
from kvlint.fix.writer import write

__all__ = [
    "FixReport",
    "Rates",
    "canonicalize",
    "deltas",
    "fix_requests",
    "mean_ttft_ms",
    "normalize_text",
    "total_cost",
    "write",
]
