"""SGLang RadixAttention prefix cache, simulated.

Semantics read from sgl-project/sglang @ main (2026-09-10),
`python/sglang/srt/mem_cache/radix_cache.py`:

- `match_prefix` walks a radix tree keyed by token ids and returns the longest
  common prefix, at token granularity rather than block granularity.
- When `page_size > 1` the query is page-aligned first
  (`key = key.page_aligned(self.page_size)`), so matches land on page multiples.
  The default is 1, which is true token granularity.

This is the finer-grained of the two simulators: vLLM can only see multiples of
16 tokens, while this sees the exact token where two prompts stop agreeing. That
makes it the attribution source for the lint rules in M4, not just a second
hit-rate number.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import Any

DEFAULT_PAGE_SIZE = 1


class RadixNode:
    """One edge in the radix tree, holding a run of tokens.

    `owner` is the request that first wrote this span, which is what lets a lint
    finding say "you diverged from request X" rather than just "you diverged".
    """

    __slots__ = ("alive", "children", "key", "last_access", "owner", "parent")

    def __init__(
        self,
        key: list[int],
        parent: RadixNode | None,
        owner: str,
        last_access: int,
    ) -> None:
        self.key = key
        self.parent = parent
        self.owner = owner
        self.last_access = last_access
        self.children: dict[int, RadixNode] = {}
        self.alive = True

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _common_prefix_len(node_key: list[int], tokens: Sequence[int], offset: int) -> int:
    """How many tokens of `node_key` match `tokens` starting at `offset`."""
    limit = min(len(node_key), len(tokens) - offset)
    i = 0
    while i < limit and node_key[i] == tokens[offset + i]:
        i += 1
    return i


class SGLangRadixSimulator:
    """Token-level radix prefix cache with LRU eviction over leaves."""

    engine = "sglang"

    def __init__(
        self,
        kv_budget_tokens: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        if kv_budget_tokens is not None and kv_budget_tokens < 1:
            raise ValueError("kv_budget_tokens must be at least 1 when set")

        self.kv_budget_tokens = kv_budget_tokens
        self.page_size = page_size

        self._root = RadixNode(key=[], parent=None, owner="", last_access=0)
        self._clock = 0
        self._total_tokens = 0
        self._seen_any_request = False
        # Lazy-deletion heap of candidate leaves. Entries go stale when a node is
        # touched or stops being a leaf, so pops are validated rather than trusted.
        self._leaves: list[tuple[int, int, RadixNode]] = []
        self._heap_seq = 0
        self.evictions = 0

    def config(self) -> dict[str, Any]:
        return {"kv_budget_tokens": self.kv_budget_tokens, "page_size": self.page_size}

    @property
    def cached_tokens_held(self) -> int:
        """Tokens currently resident in the tree. Used by eviction and tests."""
        return self._total_tokens

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _push_leaf(self, node: RadixNode) -> None:
        self._heap_seq += 1
        heapq.heappush(self._leaves, (node.last_access, self._heap_seq, node))

    # ---------------------------------------------------------------- matching

    def _match(self, tokens: Sequence[int]) -> tuple[int, str | None]:
        """Longest common prefix against everything inserted so far."""
        node = self._root
        matched = 0
        owner: str | None = None

        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break
            shared = _common_prefix_len(child.key, tokens, matched)
            if shared == 0:
                break
            matched += shared
            owner = child.owner
            if shared < len(child.key):
                # Diverged inside this edge, so the walk ends here.
                break
            node = child

        return matched, owner
