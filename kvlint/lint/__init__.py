"""Lint engine: turns cache divergences into explained, actionable findings."""

from __future__ import annotations

from kvlint.lint.attribution import Attribution, attribute, shared_prefix_length
from kvlint.lint.engine import exceeds, run, worst_severity
from kvlint.lint.rules import ALL_RULE_IDS

__all__ = [
    "ALL_RULE_IDS",
    "Attribution",
    "attribute",
    "exceeds",
    "run",
    "shared_prefix_length",
    "worst_severity",
]
