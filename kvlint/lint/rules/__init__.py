"""Individual lint rules. Each is a pure function over a divergence context.

Two shapes, because two of the eight questions cannot be answered one divergence
at a time:

- Per-divergence rules see a single miss and the prompt it drifted from.
- Corpus rules see the whole log. `short_prefix` is about what every request has
  in common, and `volatile_fewshot` is about variation across a family, so
  neither has a single divergence to point at.

Both produce the same `LintFinding`, so the engine, CLI and report do not care
which kind fired.
"""

from __future__ import annotations

from types import ModuleType

from kvlint.lint.rules import (
    block_straddle,
    dynamic_id,
    dynamic_time,
    key_order,
    short_prefix,
    user_in_system,
    volatile_fewshot,
    whitespace_drift,
)

# Ordered by how expensive the problem usually is, which is also the order
# findings are reported in when severity ties.
PER_DIVERGENCE_RULES: tuple[ModuleType, ...] = (
    dynamic_time,
    dynamic_id,
    user_in_system,
    whitespace_drift,
    key_order,
    block_straddle,
)

CORPUS_RULES: tuple[ModuleType, ...] = (
    short_prefix,
    volatile_fewshot,
)

ALL_RULE_IDS: tuple[str, ...] = tuple(
    module.RULE_ID for module in (*PER_DIVERGENCE_RULES, *CORPUS_RULES)
)

__all__ = ["ALL_RULE_IDS", "CORPUS_RULES", "PER_DIVERGENCE_RULES"]
