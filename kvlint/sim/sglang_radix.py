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

from collections.abc import Sequence

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
