"""Timestamps, dates and clock times that change between requests."""

from __future__ import annotations

import re

from kvlint.lint.context import RuleContext
from kvlint.lint.rules._shared import differing_pair

RULE_ID = "dynamic_time"
SEVERITY = "high"
AUTO_FIXABLE = True
SUGGESTION = (
    "Move the timestamp to the end of the system prompt, or into the last user "
    "message, so the stable text above it stays cacheable."
)

TIME_PATTERN = re.compile(
    r"""
    \d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?  # ISO date/time
    | \d{2}/\d{2}/\d{4}                                            # US date
    | \b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?\b               # clock time
    | \b1[0-9]{9}(?:[0-9]{3})?\b                                   # epoch s or ms
    | \b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day|sday|nesday|rsday|urday)?\b
    | \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b
    """,
    re.VERBOSE,
)


def detect(ctx: RuleContext) -> str | None:
    pair = differing_pair(TIME_PATTERN, ctx.current_window, ctx.prior_window)
    if pair is None:
        return None
    here, there = pair
    where = "system prompt" if ctx.in_system_prompt else "prompt"
    return f"time value in {where} changed: {here!r} vs {there!r}"
