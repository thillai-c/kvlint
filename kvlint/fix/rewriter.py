"""Rewrite a log to remove cache-breaking patterns.

Transformations run at the **message level** (build plan section 5.5). Never at
token level: the engine sees a rendered string, and editing tokens would produce
something no template could have generated.
"""

from __future__ import annotations

import re
import unicodedata

_TRAILING_SPACE = re.compile(r"[ 	]+$", re.MULTILINE)

# ---------------------------------------------------------------- normalization


def normalize_text(text: str) -> str:
    """CRLF to LF, no trailing spaces, NFC. The `whitespace_drift` fix."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("", text)
    return unicodedata.normalize("NFC", text)
