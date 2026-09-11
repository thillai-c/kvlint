"""Prompts that differ only in whitespace or Unicode normalization form.

The most frustrating class of cache miss: the prompts are semantically identical
and a human diffing them sees nothing, but the tokenizer produces different ids
so nothing after the drift can be reused.
"""

from __future__ import annotations

import re
import unicodedata

from kvlint.lint.context import RuleContext

RULE_ID = "whitespace_drift"
SEVERITY = "high"
AUTO_FIXABLE = True
SUGGESTION = (
    "Normalize the template before sending: convert CRLF to LF, strip trailing "
    "spaces, collapse repeated blanks, and apply NFC."
)

_WHITESPACE = re.compile(r"\s+")


def _canonical(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _describe(current: str, prior: str) -> str:
    if current.replace("\r\n", "\n") == prior.replace("\r\n", "\n"):
        return "CRLF vs LF line endings"
    if unicodedata.normalize("NFC", current) == unicodedata.normalize("NFC", prior):
        return "Unicode normalization form (NFC vs NFD)"
    if current.rstrip() == prior.rstrip():
        return "trailing whitespace"
    return "whitespace only"


def detect(ctx: RuleContext) -> str | None:
    current, prior = ctx.current_window, ctx.prior_window
    if not current or not prior or current == prior:
        return None

    # Fires only when the two windows are identical once whitespace and Unicode
    # form are normalized away. Any real content difference disqualifies it.
    if _canonical(current) != _canonical(prior):
        return None

    return f"prompts differ by {_describe(current, prior)} only"
