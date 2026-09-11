"""Static content that ends part-way through a block.

vLLM caches whole blocks only. If the stable prefix ends mid-block, the block it
ends in also holds per-request content, so its hash differs every time and the
stable tokens inside it are thrown away with the rest.

Only visible because attribution is token-exact: vLLM's own divergence index is
already rounded down to a block boundary, so the waste is invisible from there.
"""

from __future__ import annotations

from kvlint.lint.context import RuleContext

RULE_ID = "block_straddle"
SEVERITY = "low"
AUTO_FIXABLE = True
SUGGESTION = (
    "Pad the static section to a block boundary so the shared prefix ends where "
    "a block ends, and those tokens become cacheable."
)

# Below this, padding costs more prompt tokens than it saves in reuse.
MIN_WASTED_TOKENS = 4


def detect(ctx: RuleContext) -> str | None:
    wasted = ctx.divergence_token_idx % ctx.block_size
    if wasted < MIN_WASTED_TOKENS:
        return None
    aligned = ctx.divergence_token_idx - wasted
    return (
        f"shared prefix ends at token {ctx.divergence_token_idx}, mid-block; "
        f"{wasted} cacheable tokens are lost because the block also holds "
        f"per-request content (last usable boundary: {aligned})"
    )
