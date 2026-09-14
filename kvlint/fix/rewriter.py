"""Rewrite a log to remove cache-breaking patterns.

Three transformations, all at the **message level** (build plan section 5.5).
Never at token level: the engine sees a rendered string, and editing tokens would
produce something no template could have generated.

1. Normalize whitespace: CRLF to LF, strip trailing spaces, NFC.
2. Canonicalize JSON in tools and fenced blocks with sorted keys.
3. Move volatile lines out of the shared prefix and into a trailing context
   block, so the stable instructions above them stay identical.

The moves preserve content. A timestamp the model was relying on is still in the
prompt, just after the stable text rather than before it. Deleting it would raise
the hit rate further and change what the model sees, which is not a trade this
tool gets to make on the user's behalf.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from kvlint.lint.rules.dynamic_id import ID_PATTERN
from kvlint.lint.rules.dynamic_time import TIME_PATTERN
from kvlint.lint.rules.user_in_system import USER_FIELD_PATTERN
from kvlint.models import Request

CONTEXT_HEADER = "[context]"

# Patterns whose values change between requests and therefore poison the prefix.
MOVABLE_PATTERNS = (TIME_PATTERN, ID_PATTERN, USER_FIELD_PATTERN)

_PLACEHOLDER = "\x00"
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_FENCED_JSON = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class FixReport:
    """What the rewriter actually changed, for honest reporting."""

    requests: list[Request]
    changed_request_ids: list[str] = field(default_factory=list)
    normalized_whitespace: int = 0
    canonicalized_json: int = 0
    moved_lines: int = 0

    @property
    def changed(self) -> int:
        return len(self.changed_request_ids)


# ---------------------------------------------------------------- normalization


def normalize_text(text: str) -> str:
    """CRLF to LF, no trailing spaces, NFC. The `whitespace_drift` fix."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("", text)
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------- JSON ordering


def canonicalize(value: Any) -> Any:
    """Recursively rebuild dicts with sorted keys.

    Python preserves insertion order, so a schema assembled from a set or a merge
    serializes differently between processes while being semantically identical.
    Sorting makes the rendered text stable. The `key_order` fix.
    """
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def _canonicalize_fenced_json(text: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        rendered = json.dumps(canonicalize(parsed), indent=2)
        if rendered != match.group(1):
            changed = True
        return f"```json\n{rendered}\n```"

    return _FENCED_JSON.sub(replace, text), changed
