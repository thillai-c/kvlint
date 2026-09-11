"""Per-user details embedded in the system prompt.

Personalization belongs in the first user turn. In the system prompt it gives
every user their own cache entry, so a shared instruction block that could have
been reused across the entire fleet is reused by nobody.
"""

from __future__ import annotations

import re

from kvlint.lint.context import RuleContext
from kvlint.lint.rules._shared import differing_pair

RULE_ID = "user_in_system"
SEVERITY = "high"
AUTO_FIXABLE = True
SUGGESTION = (
    "Move per-user fields out of the system prompt and into the first user "
    "message, so the instruction block stays identical for every user."
)

# The value stops at whitespace or '<': chat templates put markers such as
# <|im_end|> flush against the content, and quoting those in evidence is noise.
USER_FIELD_PATTERN = re.compile(
    r"""
    \b[\w.+-]+@[\w-]+\.[\w.]{2,}\b                                   # email
    | \b(?:user(?:_?name)?|customer|account|tenant|org(?:anization)?|
        first_?name|last_?name|full_?name|locale|timezone|tz)
      \s*[:=]\s*[^\s<]+                                              # labelled field
    | \b[a-z]{2}[-_][A-Z]{2}\b                                       # locale code
    | \b(?:\+\d{1,3}[ -]?)?\(?\d{3}\)?[ -]\d{3}[ -]\d{4}\b           # phone
    """,
    re.VERBOSE,
)


def detect(ctx: RuleContext) -> str | None:
    if not ctx.in_system_prompt:
        return None
    pair = differing_pair(USER_FIELD_PATTERN, ctx.current_window, ctx.prior_window)
    if pair is None:
        return None
    here, there = pair
    return f"per-user field in system prompt changed: {here!r} vs {there!r}"
