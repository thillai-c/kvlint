"""vLLM block-hash simulator tests (M2 acceptance).

Numbers here are hand-computed from the spec, not read back from the
implementation, so a regression shows up as a disagreement rather than a
quietly updated expectation.
"""

from __future__ import annotations

import pytest

from kvlint.models import TokenizedRequest
from kvlint.sim.vllm_blocks import (
    NONE_HASH,
    VLLMBlockSimulator,
    block_digest,
    block_hashes,
    simulate,
)

BLOCK = 16


def tokens(count: int, start: int = 0) -> list[int]:
    return list(range(start, start + count))


def request(request_id: str, token_ids: list[int]) -> TokenizedRequest:
    """A TokenizedRequest carrying only what the simulator reads."""
    return TokenizedRequest(
        request_id=request_id,
        rendered_text="x" * len(token_ids),
        token_ids=token_ids,
        char_offsets=[(i, i + 1) for i in range(len(token_ids))],
    )


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


# ------------------------------------------------------------------ spec cases


def test_identical_requests_hit_fully_after_the_first() -> None:
    """M2 acceptance: identical requests reach 100% after the first."""
    sim = VLLMBlockSimulator(BLOCK)
    payload = tokens(64)

    first = sim.feed(request("r1", payload))
    second = sim.feed(request("r2", payload))

    assert first.cached_tokens == 0
    assert second.cached_tokens == 64
    assert second.divergence_token_idx is None
    assert second.nearest_prior_request_id == "r1"


def test_shared_40_token_prefix_caches_exactly_32_tokens() -> None:
    """M2 acceptance, and the case that catches partial-block errors.

    40 shared tokens is 2 whole blocks plus 8 leftover. The leftover was never
    cached, so the second request can reuse only 32.
    """
    sim = VLLMBlockSimulator(BLOCK)
    shared = tokens(40)

    sim.feed(request("r1", shared + tokens(24, start=500)))
    second = sim.feed(request("r2", shared + tokens(24, start=900)))

    assert second.cached_tokens == 32
    assert second.divergence_token_idx == 32


def test_budget_of_two_blocks_evicts_across_three_requests() -> None:
    """M2 acceptance: eviction is observable under a tight budget."""
    sim = VLLMBlockSimulator(BLOCK, kv_budget_blocks=2)

    sim.feed(request("r1", tokens(32, start=0)))
    sim.feed(request("r2", tokens(32, start=1000)))
    sim.feed(request("r3", tokens(32, start=2000)))

    assert sim.evictions > 0
    # Budget honoured, never exceeded.
    assert len(sim._cache) <= 2

    # r1's blocks are gone, so replaying it now misses.
    replay = sim.feed(request("r1-again", tokens(32, start=0)))
    assert replay.cached_tokens == 0


def test_infinite_budget_never_evicts() -> None:
    sim = VLLMBlockSimulator(BLOCK, kv_budget_blocks=None)
    for i in range(5):
        sim.feed(request(f"r{i}", tokens(32, start=i * 1000)))
    assert sim.evictions == 0


# ------------------------------------------------------------------ divergence


def test_first_request_has_no_divergence() -> None:
    sim = VLLMBlockSimulator(BLOCK)
    assert sim.feed(request("r1", tokens(64))).divergence_token_idx is None


def test_divergence_is_block_aligned() -> None:
    """Reported at a block boundary because that is all vLLM can see.

    The token-exact position comes from the radix tree in M4.
    """
    sim = VLLMBlockSimulator(BLOCK)
    sim.feed(request("r1", tokens(64)))
    # Diverge one token into the third block.
    second = sim.feed(request("r2", tokens(33) + tokens(31, start=700)))
    assert second.divergence_token_idx == 32


def test_complete_miss_diverges_at_zero() -> None:
    sim = VLLMBlockSimulator(BLOCK)
    sim.feed(request("r1", tokens(32)))
    second = sim.feed(request("r2", tokens(32, start=5000)))
    assert second.divergence_token_idx == 0
    assert second.nearest_prior_request_id is None


def test_nearest_prior_is_the_block_creator_not_the_last_reuser() -> None:
    """A finding should name who established the prefix."""
    sim = VLLMBlockSimulator(BLOCK)
    sim.feed(request("creator", tokens(32)))
    sim.feed(request("reuser", tokens(32)))
    third = sim.feed(request("third", tokens(32) + tokens(16, start=800)))
    assert third.nearest_prior_request_id == "creator"


def test_prefix_walk_stops_at_the_first_miss() -> None:
    """A later matching block cannot rescue a diverged chain."""
    sim = VLLMBlockSimulator(BLOCK)
    sim.feed(request("r1", tokens(16, start=0) + tokens(16, start=100)))
    # Same second block content, different first block: the chain differs.
    second = sim.feed(request("r2", tokens(16, start=50) + tokens(16, start=100)))
    assert second.cached_tokens == 0


# ------------------------------------------------------------------ aggregate


def test_simulate_reports_hit_rate_over_all_tokens() -> None:
    payload = tokens(64)
    result = simulate([request("r1", payload), request("r2", payload)])
    # 128 prompt tokens total, 64 of them cached.
    assert result.hit_rate == pytest.approx(0.5)
    assert result.engine == "vllm"
    assert result.block_size == BLOCK


def test_partial_tail_counts_against_the_hit_rate() -> None:
    """Prompt tokens include the uncacheable tail, so 40/40 is never 100%."""
    payload = tokens(40)
    result = simulate([request("r1", payload), request("r2", payload)])
    assert result.per_request[1].cached_tokens == 32
    assert result.hit_rate == pytest.approx(32 / 80)


def test_ideal_hit_rate_exceeds_actual_under_pressure() -> None:
    """The gap is the cost of memory pressure, separate from bad prompts."""
    requests = [
        request("r1", tokens(32, start=0)),
        request("r2", tokens(32, start=1000)),
        request("r3", tokens(32, start=0)),
    ]
    squeezed = simulate(requests, BLOCK, kv_budget_blocks=2)
    roomy = simulate(requests, BLOCK, kv_budget_blocks=None)

    assert squeezed.hit_rate < squeezed.ideal_hit_rate
    assert squeezed.ideal_hit_rate == pytest.approx(roomy.hit_rate)


def test_evicted_hit_loss_is_attributed_to_the_request_that_lost() -> None:
    requests = [
        request("r1", tokens(32, start=0)),
        request("r2", tokens(32, start=1000)),
        request("r3", tokens(32, start=0)),
    ]
    result = simulate(requests, BLOCK, kv_budget_blocks=2)
    by_id = {r.request_id: r for r in result.per_request}
    assert by_id["r3"].evicted_hit_loss == 32
    assert by_id["r1"].evicted_hit_loss == 0


def test_infinite_budget_reports_no_loss() -> None:
    result = simulate([request("r1", tokens(32))], BLOCK, None)
    assert all(r.evicted_hit_loss == 0 for r in result.per_request)
    assert result.hit_rate == result.ideal_hit_rate


def test_empty_log_does_not_divide_by_zero() -> None:
    result = simulate([])
    assert result.hit_rate == 0.0
    assert result.per_request == []


# ------------------------------------------------------------------ determinism


def test_repeated_runs_are_byte_identical() -> None:
    """Build plan section 2: same input and config, byte-identical JSON."""
    requests = [request(f"r{i}", tokens(48, start=i * 7)) for i in range(10)]
    first = simulate(requests, BLOCK, kv_budget_blocks=4)
    second = simulate(requests, BLOCK, kv_budget_blocks=4)
    assert first.model_dump_json() == second.model_dump_json()


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_invalid_block_size(bad: int) -> None:
    with pytest.raises(ValueError, match="block_size"):
        VLLMBlockSimulator(bad)


def test_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="kv_budget_blocks"):
        VLLMBlockSimulator(BLOCK, kv_budget_blocks=0)


def test_block_size_is_configurable() -> None:
    sim = VLLMBlockSimulator(block_size=8)
    sim.feed(request("r1", tokens(40)))
    second = sim.feed(request("r2", tokens(40)))
    assert second.cached_tokens == 40  # 40 is an exact multiple of 8
