"""vLLM v1 automatic prefix caching, simulated.

Semantics read from vllm-project/vllm @ main (2026-09-10), specifically
`vllm/v1/core/kv_cache_utils.py` and `vllm/v1/core/block_pool.py`:

- `hash_block_tokens(fn, parent_block_hash, curr_block_token_ids, extra_keys)`
  returns `fn((parent_hash, tuple(token_ids), extra_keys))`, so block hashes form
  a chain and an identical block at a different offset is a different hash.
- `NONE_HASH` seeds the chain, derived from the fixed string "vllm-none-hash"
  when the hash function is cryptographic.
- `get_request_block_hasher` hashes only complete blocks; it stops when the next
  full block would exceed the token count, so a trailing partial block is never
  cached.
- `get_cached_block` returns None on the first missing block, so reuse is a
  prefix walk that stops at the first miss rather than a set intersection.
- The free block queue is LRU over blocks with `ref_cnt == 0`, and uncached
  blocks are evicted before cached ones.

We reproduce the *equality semantics*, not the bit pattern. Our digests never
appear in output; only hit counts do. `extra_keys` is always None here because
multimodal, LoRA, and cache salting are v0.1 non-goals (build plan section 1).
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from kvlint.models import PerRequestResult, SimResult, TokenizedRequest
from kvlint.sim.base import aggregate

DEFAULT_BLOCK_SIZE = 16

# Mirrors vLLM's NONE_HASH seed. Fixed, never random: PYTHONHASHSEED randomization
# would make results differ between runs, breaking the determinism requirement in
# build plan section 2.
NONE_HASH = hashlib.sha256(b"vllm-none-hash").digest()

# Token ids are well inside int32 for every model we target, and array().tobytes()
# is markedly faster than joining strings for the ~1.25M hashes a large log needs.
_TOKEN_TYPECODE = "i"


def block_digest(parent: bytes, tokens: Sequence[int]) -> bytes:
    """Chain one block onto its parent.

    The parent digest is fixed-width, so there is no delimiter ambiguity between
    it and the token bytes that follow.
    """
    hasher = hashlib.sha256(parent)
    hasher.update(array(_TOKEN_TYPECODE, tokens).tobytes())
    return hasher.digest()


def block_hashes(token_ids: Sequence[int], block_size: int) -> list[bytes]:
    """Chained digests for every *complete* block; the partial tail is dropped."""
    digests: list[bytes] = []
    parent = NONE_HASH
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = block_digest(parent, token_ids[start : start + block_size])
        digests.append(parent)
    return digests


class VLLMBlockSimulator:
    """Block-hash prefix cache with LRU eviction under a block budget."""

    engine = "vllm"

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_budget_blocks: int | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if kv_budget_blocks is not None and kv_budget_blocks < 1:
            raise ValueError("kv_budget_blocks must be at least 1 when set")

        self.block_size = block_size
        self.kv_budget_blocks = kv_budget_blocks

        # Insertion order is LRU order: least recently used first. Values are the
        # id of the request that first created the block, which is how a finding
        # can say "you diverged from request X".
        self._cache: OrderedDict[bytes, str] = OrderedDict()
        self._seen_any_request = False
        self.evictions = 0

    def config(self) -> dict[str, Any]:
        return {"block_size": self.block_size, "kv_budget_blocks": self.kv_budget_blocks}

    def feed(self, request: TokenizedRequest) -> PerRequestResult:
        return self.feed_tokens(request.request_id, request.token_ids)

    def feed_tokens(self, request_id: str, token_ids: Sequence[int]) -> PerRequestResult:
        """Simulate one request. Kept separate from `feed` so the perf benchmark
        can measure the hash step without paying for pydantic validation."""
        digests = block_hashes(token_ids, self.block_size)

        # 1. Prefix walk. vLLM stops at the first miss, so a later matching block
        #    is worthless: its parent hash differs once the chain has diverged.
        hits = 0
        for digest in digests:
            if digest not in self._cache:
                break
            hits += 1

        nearest_prior = self._cache[digests[hits - 1]] if hits else None

        # 2. Insert or touch. Touching keeps the original writer id: we want to
        #    know who created the block, not who last reused it.
        for digest in digests:
            if digest in self._cache:
                self._cache.move_to_end(digest)
            else:
                self._cache[digest] = request_id

        # 3. Evict. Every block of this request is now at the LRU tail, so once
        #    the oldest entry belongs to this request there is nothing older left
        #    to drop. A request larger than the whole budget therefore overshoots,
        #    which real vLLM would handle by preemption.
        #    ASSUMPTION: no concurrency, so "belongs to this request" stands in
        #    for vLLM's ref_cnt > 0.
        if self.kv_budget_blocks is not None and len(self._cache) > self.kv_budget_blocks:
            current = set(digests)
            while len(self._cache) > self.kv_budget_blocks:
                oldest = next(iter(self._cache))
                if oldest in current:
                    break
                del self._cache[oldest]
                self.evictions += 1

        # A first request cannot have diverged from anything, and a full hit has
        # no divergence point either.
        divergence: int | None = None
        if self._seen_any_request and hits < len(digests):
            divergence = hits * self.block_size

        self._seen_any_request = True

        return PerRequestResult(
            request_id=request_id,
            prompt_tokens=len(token_ids),
            cached_tokens=hits * self.block_size,
            divergence_token_idx=divergence,
            nearest_prior_request_id=nearest_prior,
        )


def simulate(
    requests: Sequence[TokenizedRequest],
    block_size: int = DEFAULT_BLOCK_SIZE,
    kv_budget_blocks: int | None = None,
) -> SimResult:
    """Run a log through the simulator, plus an infinite-budget reference pass."""
    budgeted = VLLMBlockSimulator(block_size, kv_budget_blocks)
    actual = [budgeted.feed(request) for request in requests]

    if kv_budget_blocks is None:
        ideal = actual
    else:
        unbounded = VLLMBlockSimulator(block_size, None)
        ideal = [unbounded.feed(request) for request in requests]

    return aggregate(
        engine="vllm",
        config={
            **budgeted.config(),
            "evictions": budgeted.evictions,
        },
        actual=actual,
        ideal=ideal,
        block_size=block_size,
    )
