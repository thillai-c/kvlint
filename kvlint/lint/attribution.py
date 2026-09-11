"""Where each request stopped matching everything that came before it.

Implements plan decision D2: the lint rules read a token-exact divergence rather
than vLLM's block-rounded one. A 16-token rounding error is enough to hide the
timestamp that caused the miss, and `block_straddle` can only be detected by
comparing the exact index against the block-aligned one.

The structure that answers this is a radix tree over every prior request, which
is precisely the SGLang simulator at infinite budget. Reusing it rather than
building a second tree means the lint divergence and the reported SGLang hit
rate can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvlint.models import TokenizedRequest
from kvlint.sim.sglang_radix import SGLangRadixSimulator


@dataclass(frozen=True)
class Attribution:
    """One request's relationship to the best prefix available before it."""

    request_id: str
    divergence_token_idx: int | None
    """None when nothing preceded this request, or when it matched completely."""

    nearest_prior_request_id: str | None
    prompt_tokens: int

    @property
    def tokens_lost(self) -> int:
        """Tokens after the divergence, all of which had to be recomputed.

        An upper bound on the cost of this divergence: some of those tokens were
        genuinely new content that could never have been cached.
        """
        if self.divergence_token_idx is None:
            return 0
        return self.prompt_tokens - self.divergence_token_idx


def attribute(requests: list[TokenizedRequest]) -> list[Attribution]:
    """Token-exact divergence for every request, in arrival order."""
    # Infinite budget and page_size 1: we want to know where prompts actually
    # diverge, not what a particular memory configuration would have evicted.
    simulator = SGLangRadixSimulator(kv_budget_tokens=None, page_size=1)

    attributions: list[Attribution] = []
    for request in requests:
        result = simulator.feed(request)
        attributions.append(
            Attribution(
                request_id=result.request_id,
                divergence_token_idx=result.divergence_token_idx,
                nearest_prior_request_id=result.nearest_prior_request_id,
                prompt_tokens=result.prompt_tokens,
            )
        )
    return attributions


def shared_prefix_length(requests: list[TokenizedRequest]) -> int:
    """Longest token prefix common to every request in the log.

    If this is shorter than one block, nothing in the workload is cacheable at
    all, which is what `short_prefix` reports.
    """
    if not requests:
        return 0

    prefix = requests[0].token_ids
    for request in requests[1:]:
        tokens = request.token_ids
        limit = min(len(prefix), len(tokens))
        i = 0
        while i < limit and prefix[i] == tokens[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return len(prefix)
