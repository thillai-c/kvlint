"""SGLang RadixAttention prefix cache, simulated.

Semantics read from sgl-project/sglang @ main (2026-09-10),
`python/sglang/srt/mem_cache/radix_cache.py`:

- `match_prefix` walks a radix tree keyed by token ids and returns the longest
  common prefix, at token granularity rather than block granularity.
- When `page_size > 1` the query is page-aligned first
  (`key = key.page_aligned(self.page_size)`), so matches land on page multiples.
  The default is 1, which is true token granularity.
- Eviction takes evictable leaves in `last_access_time` order, requiring
  `lock_ref == 0` and all children already evicted.

This is the finer-grained of the two simulators: vLLM can only see multiples of
16 tokens, while this sees the exact token where two prompts stop agreeing. That
makes it the attribution source for the lint rules in M4, not just a second
hit-rate number.

ASSUMPTION: real SGLang eviction has more nuance than pure LRU over leaves
(page tables, lock refs held by running requests). We model the documented
behaviour and note the gap, per build plan section 9.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import Any

from kvlint.models import PerRequestResult, SimResult, TokenizedRequest
from kvlint.sim.base import aggregate

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

    # ---------------------------------------------------------------- insertion

    def _split(self, node: RadixNode, at: int) -> None:
        """Break `node` into a head of length `at` and a tail holding the rest."""
        tail = RadixNode(
            key=node.key[at:],
            parent=node,
            owner=node.owner,
            last_access=node.last_access,
        )
        tail.children = node.children
        for grandchild in tail.children.values():
            grandchild.parent = tail

        node.children = {tail.key[0]: tail}
        node.key = node.key[:at]
        if tail.is_leaf:
            self._push_leaf(tail)

    def _insert(self, tokens: Sequence[int], request_id: str) -> None:
        node = self._root
        position = 0
        now = self._tick()

        while position < len(tokens):
            first = tokens[position]
            child = node.children.get(first)

            if child is None:
                leaf = RadixNode(
                    key=list(tokens[position:]),
                    parent=node,
                    owner=request_id,
                    last_access=now,
                )
                node.children[first] = leaf
                self._total_tokens += len(leaf.key)
                self._push_leaf(leaf)
                return

            shared = _common_prefix_len(child.key, tokens, position)
            if shared < len(child.key):
                self._split(child, shared)

            child.last_access = now
            node = child
            position += shared

        # The whole request was already present; refresh the node we landed on.
        node.last_access = now
        if node.is_leaf:
            self._push_leaf(node)

    # ---------------------------------------------------------------- eviction

    def _evict(self, protect_after: int) -> None:
        """Drop least recently used leaves until back inside the budget.

        `protect_after` is the clock value at the start of this request. Anything
        touched at or after it belongs to the request we just served, and real
        SGLang would hold a lock_ref on it, so we stop rather than evict it.
        """
        if self.kv_budget_tokens is None:
            return

        while self._total_tokens > self.kv_budget_tokens and self._leaves:
            stamp, _, node = heapq.heappop(self._leaves)

            # Stale entry: the node was touched, adopted children, or already went.
            if not node.alive or not node.is_leaf or node.last_access != stamp:
                continue
            if node.last_access >= protect_after:
                # Everything left belongs to the request we just served.
                self._push_leaf(node)
                return

            parent = node.parent
            if parent is None:  # pragma: no cover
                continue

            del parent.children[node.key[0]]
            self._total_tokens -= len(node.key)
            node.alive = False
            self.evictions += 1

            if parent is not self._root and parent.is_leaf:
                self._push_leaf(parent)

    # ---------------------------------------------------------------- feeding

    def feed(self, request: TokenizedRequest) -> PerRequestResult:
        return self.feed_tokens(request.request_id, request.token_ids)

    def feed_tokens(self, request_id: str, token_ids: Sequence[int]) -> PerRequestResult:
        """Simulate one request. Split from `feed` so benchmarks can skip pydantic."""
        matched, owner = self._match(token_ids)

        # Page alignment: SGLang cannot reuse a partial page.
        if self.page_size > 1:
            matched -= matched % self.page_size
            if matched == 0:
                owner = None

        protect_after = self._clock + 1
        self._insert(token_ids, request_id)
        self._evict(protect_after)

        divergence: int | None = None
        if self._seen_any_request and matched < len(token_ids):
            divergence = matched
        self._seen_any_request = True

        return PerRequestResult(
            request_id=request_id,
            prompt_tokens=len(token_ids),
            cached_tokens=matched,
            divergence_token_idx=divergence,
            nearest_prior_request_id=owner,
        )


def simulate(
    requests: Sequence[TokenizedRequest],
    kv_budget_tokens: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> SimResult:
    """Run a log through the simulator, plus an infinite-budget reference pass."""
    budgeted = SGLangRadixSimulator(kv_budget_tokens, page_size)
    actual = [budgeted.feed(request) for request in requests]

    if kv_budget_tokens is None:
        ideal = actual
    else:
        unbounded = SGLangRadixSimulator(None, page_size)
        ideal = [unbounded.feed(request) for request in requests]

    return aggregate(
        engine="sglang",
        config={**budgeted.config(), "evictions": budgeted.evictions},
        actual=actual,
        ideal=ideal,
        # Token granularity, so there is no block size to report.
        block_size=page_size if page_size > 1 else None,
    )
