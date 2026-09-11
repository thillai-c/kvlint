"""Inputs handed to lint rules.

Lives in its own module so `rules/` can import it without importing `engine`,
which imports `rules`. Plain dataclasses rather than pydantic models: these never
cross a serialization boundary, they only travel from the engine into a rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvlint.models import Request, TokenizedRequest


@dataclass(frozen=True)
class RuleContext:
    """One divergence, with both sides of it in text form.

    Rules compare `current_after` against `prior_after`: the text that follows the
    exact token where these two prompts stopped agreeing. Everything before the
    divergence is by definition identical, so the cause is at the head of those
    two strings.
    """

    request: Request
    tokenized: TokenizedRequest
    prior: Request | None
    prior_tokenized: TokenizedRequest | None

    divergence_token_idx: int
    divergence_char_pos: int

    before: str
    """Shared text immediately preceding the divergence. Context, not evidence."""

    current_after: str
    prior_after: str

    current_window: str
    """Text spanning both sides of the divergence: `before` plus `current_after`.

    Rules read this rather than `current_after` alone, because a changing value
    is usually detected part-way through and its opening characters sit on the
    shared side of the divergence.
    """

    prior_window: str

    block_size: int
    in_system_prompt: bool
    """Whether the divergence falls inside the system message. A dynamic value up
    there is far more expensive than the same value in the last user turn."""

    @property
    def tokens_after_divergence(self) -> int:
        return len(self.tokenized.token_ids) - self.divergence_token_idx


@dataclass(frozen=True)
class CorpusContext:
    """The whole log, for rules that cannot be judged one divergence at a time."""

    requests: list[Request]
    tokenized: list[TokenizedRequest]
    divergences: dict[str, int | None]
    shared_prefix_tokens: int
    """Longest token prefix common to every request in the log."""

    block_size: int


@dataclass(frozen=True)
class CorpusHit:
    """A corpus rule firing."""

    evidence: str
    affected_request_ids: list[str]
    est_tokens_lost: int
