"""vLLM block-hash simulator tests (M2 acceptance).

Numbers here are hand-computed from the spec, not read back from the
implementation, so a regression shows up as a disagreement rather than a
quietly updated expectation.
"""

from __future__ import annotations

from kvlint.sim.vllm_blocks import NONE_HASH, block_digest, block_hashes

BLOCK = 16


def tokens(count: int, start: int = 0) -> list[int]:
    return list(range(start, start + count))


# ------------------------------------------------------------------ hashing


def test_partial_trailing_block_is_never_hashed() -> None:
    """vLLM only caches complete blocks, so 40 tokens is 2 blocks, not 3."""
    assert len(block_hashes(tokens(40), BLOCK)) == 2


def test_exact_multiple_produces_every_block() -> None:
    assert len(block_hashes(tokens(32), BLOCK)) == 2


def test_short_input_produces_no_blocks() -> None:
    assert block_hashes(tokens(15), BLOCK) == []


def test_hashes_chain_through_the_parent() -> None:
    """Identical block content at a different offset must hash differently.

    This is the property that makes prefix caching a prefix, rather than a
    content-addressed block store.
    """
    same_block = tokens(16, start=100)
    first = block_digest(NONE_HASH, same_block)
    second = block_digest(first, same_block)
    assert first != second


def test_first_block_chains_from_none_hash() -> None:
    assert block_hashes(tokens(16), BLOCK)[0] == block_digest(NONE_HASH, tokens(16))


def test_shared_prefix_produces_shared_hashes() -> None:
    a = block_hashes(tokens(64), BLOCK)
    b = block_hashes(tokens(32) + tokens(32, start=900), BLOCK)
    assert a[:2] == b[:2]
    assert a[2] != b[2]
