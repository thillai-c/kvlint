"""Helpers shared by the pattern-matching rules.

Rules compare a window spanning both sides of the divergence, not just the text
after it. A changing value is usually detected part-way through: two prompts
holding "2026-09-11T08:00:00Z" and "2026-09-11T08:01:00Z" agree right up to the
minute digit, so everything after the divergence is the meaningless tail
"0:00Z". Only by reaching back before the divergence does the full timestamp
become visible.

The rules stay conservative despite the wider window: each demands that the same
slot holds a different value on both sides, rather than firing on the mere
presence of a suspicious-looking string.
"""

from __future__ import annotations

import re


def differing_pair(
    pattern: re.Pattern[str],
    current: str,
    prior: str,
) -> tuple[str, str] | None:
    """First position where both windows match `pattern` but disagree.

    Matches are compared in order rather than by first occurrence, so an
    unchanged timestamp earlier in the window does not mask a changed one later.
    """
    here = pattern.findall(current)
    there = pattern.findall(prior)
    if not here or not there:
        return None

    for mine, theirs in zip(here, there, strict=False):
        # findall returns tuples when a pattern has groups; ours use none, but
        # guard anyway so a future edit cannot silently break this.
        mine = mine if isinstance(mine, str) else "".join(mine)
        theirs = theirs if isinstance(theirs, str) else "".join(theirs)
        if mine != theirs:
            return mine, theirs
    return None
