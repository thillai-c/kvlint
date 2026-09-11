"""Nothing shared across the log is long enough to cache.

A corpus-level rule: there is no divergence to point at, the problem is that the
requests never agree on enough leading tokens to fill a single block.
"""

from __future__ import annotations

from kvlint.lint.context import CorpusContext, CorpusHit

RULE_ID = "short_prefix"
SEVERITY = "high"
AUTO_FIXABLE = False
SUGGESTION = (
    "Give these requests a shared, stable system prompt. Until the common prefix "
    "exceeds one block, prefix caching cannot help this workload at all."
)


def detect(ctx: CorpusContext) -> CorpusHit | None:
    if len(ctx.tokenized) < 2:
        return None
    if ctx.shared_prefix_tokens >= ctx.block_size:
        return None

    return CorpusHit(
        evidence=(
            f"longest prefix shared by all {len(ctx.tokenized)} requests is "
            f"{ctx.shared_prefix_tokens} tokens, under the {ctx.block_size}-token "
            "block size, so no block is ever reusable"
        ),
        affected_request_ids=[t.request_id for t in ctx.tokenized],
        est_tokens_lost=sum(len(t.token_ids) for t in ctx.tokenized),
    )
