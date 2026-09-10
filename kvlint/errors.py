"""Exception types.

Not in the layout in KVLINT_BUILD_PLAN.md §3, but ingest and tokenize both need
to fail with a message that names the offending line or model, and burying that
in either package would make the other import it awkwardly.
"""

from __future__ import annotations


class KvlintError(Exception):
    """Base for every error kvlint raises deliberately."""


class IngestError(KvlintError):
    """A log file could not be parsed.

    Always names the file and line so a user with a 100k-line log can find it.
    """

    def __init__(self, path: str, line_no: int, reason: str) -> None:
        self.path = path
        self.line_no = line_no
        self.reason = reason
        super().__init__(f"{path}:{line_no}: {reason}")


class TokenizerError(KvlintError):
    """A tokenizer could not be loaded, or is unusable for our purposes."""
