"""Few-shot exemplars that change between requests in the same template family.

Exemplars are usually long, and they sit above the real input, so rotating or
reshuffling them throws away the most valuable part of the cache. Corpus-level
because a single divergence cannot tell you that a section is an exemplar block.
"""

from __future__ import annotations

import re

from kvlint.lint.context import CorpusContext, CorpusHit

RULE_ID = "volatile_fewshot"
SEVERITY = "medium"
AUTO_FIXABLE = False
SUGGESTION = (
    "Pin the exemplar set and its order. Sampling or shuffling examples per "
    "request gives every request a different prefix."
)

EXEMPLAR_MARKER = re.compile(
    r"^\s*(?:Example\s*\d*|Q|A|Input|Output|User|Assistant)\s*[:.\)]",
    re.MULTILINE | re.IGNORECASE,
)

# One or two markers is ordinary prose. A real exemplar block repeats.
MIN_MARKERS = 3


def _system_text(messages: list[str]) -> str:
    return "\n".join(messages)


def detect(ctx: CorpusContext) -> CorpusHit | None:
    systems: dict[str, str] = {}
    for request in ctx.requests:
        text = _system_text([m.content for m in request.messages if m.role == "system"])
        if text:
            systems[request.request_id] = text

    if len(systems) < 2 or len(set(systems.values())) < 2:
        return None  # every system prompt identical, nothing volatile

    # Only call it a few-shot problem if the varying text actually looks like
    # a set of exemplars, rather than any other kind of system-prompt drift.
    marker_counts = [len(EXEMPLAR_MARKER.findall(text)) for text in systems.values()]
    if min(marker_counts) < MIN_MARKERS:
        return None

    affected = sorted(systems)
    lost = sum(
        t.prompt_tokens if hasattr(t, "prompt_tokens") else len(t.token_ids)
        for t in ctx.tokenized
        if t.request_id in systems
    )
    return CorpusHit(
        evidence=(
            f"{len(set(systems.values()))} distinct exemplar blocks across "
            f"{len(systems)} requests, each averaging "
            f"{sum(marker_counts) // len(marker_counts)} examples"
        ),
        affected_request_ids=affected,
        est_tokens_lost=lost,
    )
