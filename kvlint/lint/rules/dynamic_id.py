"""UUIDs, hex digests, correlation ids and counters that change between requests."""

from __future__ import annotations

import re

from kvlint.lint.context import RuleContext
from kvlint.lint.rules._shared import differing_pair

RULE_ID = "dynamic_id"
SEVERITY = "high"
AUTO_FIXABLE = True
SUGGESTION = (
    "Drop the identifier from the prompt, or move it to the end. Models rarely "
    "need a request id, and it invalidates every token that follows it."
)

ID_PATTERN = re.compile(
    r"""
    \b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b  # UUID
    | \b(?:req|sess|session|trace|span|conv|txn|corr|run|job)[-_][A-Za-z0-9]{4,}\b   # prefixed
    | \b[0-9a-fA-F]{16,64}\b                                                          # hex digest
    | \b\d{6,}\b                                                                      # counter
    """,
    re.VERBOSE,
)


def detect(ctx: RuleContext) -> str | None:
    pair = differing_pair(ID_PATTERN, ctx.current_window, ctx.prior_window)
    if pair is None:
        return None
    here, there = pair
    where = "system prompt" if ctx.in_system_prompt else "prompt"
    return f"identifier in {where} changed: {here!r} vs {there!r}"
