"""SGLang radix simulator tests (M3 acceptance).

The point of contrast with M2: vLLM rounds reuse down to whole 16-token blocks,
this matches exactly. Several tests here assert positions that vLLM could not
report, because that precision is what the lint rules in M4 depend on.
"""

from __future__ import annotations

import pytest

from kvlint.models import TokenizedRequest
from kvlint.sim.sglang_radix import SGLangRadixSimulator, simulate


def tokens(count: int, start: int = 0) -> list[int]:
    return list(range(start, start + count))


def request(request_id: str, token_ids: list[int]) -> TokenizedRequest:
    return TokenizedRequest(
        request_id=request_id,
        rendered_text="x" * len(token_ids),
        token_ids=token_ids,
        char_offsets=[(i, i + 1) for i in range(len(token_ids))],
    )


# ------------------------------------------------------------------ matching


def test_first_request_caches_nothing() -> None:
    sim = SGLangRadixSimulator()
    result = sim.feed(request("r1", tokens(50)))
    assert result.cached_tokens == 0
    assert result.divergence_token_idx is None


def test_identical_request_hits_completely() -> None:
    sim = SGLangRadixSimulator()
    payload = tokens(50)
    sim.feed(request("r1", payload))
    second = sim.feed(request("r2", payload))
    assert second.cached_tokens == 50
    assert second.divergence_token_idx is None


def test_match_is_token_exact_not_block_rounded() -> None:
    """M3 acceptance: no block rounding.

    A 40-token shared prefix caches all 40 here, where vLLM caches only 32.
    """
    sim = SGLangRadixSimulator()
    shared = tokens(40)
    sim.feed(request("r1", shared + tokens(10, start=500)))
    second = sim.feed(request("r2", shared + tokens(10, start=900)))
    assert second.cached_tokens == 40
    assert second.divergence_token_idx == 40


@pytest.mark.parametrize("shared_len", [1, 7, 15, 17, 31, 33, 63])
def test_off_by_one_prefixes_match_exactly(shared_len: int) -> None:
    """M3 acceptance: correct on off-by-one prefixes, at every offset."""
    sim = SGLangRadixSimulator()
    shared = tokens(shared_len)
    sim.feed(request("r1", [*shared, 9001, *tokens(20, start=100)]))
    second = sim.feed(request("r2", [*shared, 9002, *tokens(20, start=100)]))
    assert second.cached_tokens == shared_len
    assert second.divergence_token_idx == shared_len


def test_complete_miss_caches_nothing() -> None:
    sim = SGLangRadixSimulator()
    sim.feed(request("r1", tokens(30)))
    second = sim.feed(request("r2", tokens(30, start=5000)))
    assert second.cached_tokens == 0
    assert second.divergence_token_idx == 0
    assert second.nearest_prior_request_id is None


def test_shorter_request_matching_a_prefix_hits_fully() -> None:
    """A prefix of an existing entry is entirely cached."""
    sim = SGLangRadixSimulator()
    sim.feed(request("long", tokens(100)))
    shorter = sim.feed(request("short", tokens(40)))
    assert shorter.cached_tokens == 40
    assert shorter.divergence_token_idx is None


def test_growing_conversation_reuses_the_whole_history() -> None:
    """The multi-turn pattern ShareGPT ingest produces."""
    sim = SGLangRadixSimulator()
    turn1 = tokens(30)
    turn2 = turn1 + tokens(20, start=300)
    turn3 = turn2 + tokens(20, start=600)

    sim.feed(request("t1", turn1))
    assert sim.feed(request("t2", turn2)).cached_tokens == 30
    assert sim.feed(request("t3", turn3)).cached_tokens == 50


def test_three_way_branch_from_a_shared_prefix() -> None:
    """Node splitting: three prompts sharing a trunk, diverging separately."""
    sim = SGLangRadixSimulator()
    trunk = tokens(25)
    sim.feed(request("a", trunk + tokens(10, start=100)))
    sim.feed(request("b", trunk + tokens(10, start=200)))
    third = sim.feed(request("c", trunk + tokens(10, start=300)))
    assert third.cached_tokens == 25
    assert third.divergence_token_idx == 25


def test_nearest_prior_names_the_span_owner() -> None:
    sim = SGLangRadixSimulator()
    sim.feed(request("creator", tokens(40)))
    second = sim.feed(request("follower", tokens(40) + tokens(5, start=700)))
    assert second.nearest_prior_request_id == "creator"


# ------------------------------------------------------------------ page size


def test_page_size_one_is_token_exact() -> None:
    sim = SGLangRadixSimulator(page_size=1)
    sim.feed(request("r1", tokens(37)))
    assert sim.feed(request("r2", [*tokens(37), 999])).cached_tokens == 37


def test_page_size_rounds_the_match_down() -> None:
    """With page_size 16, a 40-token match yields 32, like vLLM."""
    sim = SGLangRadixSimulator(page_size=16)
    shared = tokens(40)
    sim.feed(request("r1", shared + tokens(10, start=500)))
    second = sim.feed(request("r2", shared + tokens(10, start=900)))
    assert second.cached_tokens == 32


def test_page_size_larger_than_the_match_yields_nothing() -> None:
    sim = SGLangRadixSimulator(page_size=64)
    sim.feed(request("r1", tokens(40) + tokens(10, start=500)))
    second = sim.feed(request("r2", tokens(40) + tokens(10, start=900)))
    assert second.cached_tokens == 0
    assert second.nearest_prior_request_id is None


# ------------------------------------------------------------------ eviction


def test_eviction_respects_the_token_budget() -> None:
    """M3 acceptance: the tree never exceeds its budget."""
    sim = SGLangRadixSimulator(kv_budget_tokens=100)
    for i in range(10):
        sim.feed(request(f"r{i}", tokens(50, start=i * 1000)))
        assert sim.cached_tokens_held <= 100


def test_eviction_drops_the_least_recently_used_entry() -> None:
    sim = SGLangRadixSimulator(kv_budget_tokens=60)
    sim.feed(request("old", tokens(30, start=0)))
    sim.feed(request("mid", tokens(30, start=1000)))
    # Re-touch `old` so `mid` becomes the least recently used.
    sim.feed(request("old-again", tokens(30, start=0)))
    sim.feed(request("new", tokens(30, start=2000)))

    assert sim.feed(request("probe-old", tokens(30, start=0))).cached_tokens == 30
    assert sim.feed(request("probe-mid", tokens(30, start=1000))).cached_tokens == 0


def test_infinite_budget_never_evicts() -> None:
    sim = SGLangRadixSimulator(kv_budget_tokens=None)
    for i in range(10):
        sim.feed(request(f"r{i}", tokens(50, start=i * 1000)))
    assert sim.evictions == 0
    assert sim.cached_tokens_held == 500


def test_current_request_survives_its_own_eviction_pass() -> None:
    """A request bigger than the budget overshoots rather than evicting itself."""
    sim = SGLangRadixSimulator(kv_budget_tokens=10)
    sim.feed(request("big", tokens(100)))
    assert sim.feed(request("big-again", tokens(100))).cached_tokens == 100


# ------------------------------------------------------------------ aggregate


def test_simulate_reports_hit_rate_and_engine() -> None:
    payload = tokens(50)
    result = simulate([request("r1", payload), request("r2", payload)])
    assert result.engine == "sglang"
    assert result.hit_rate == pytest.approx(0.5)
    assert result.block_size is None


def test_page_size_is_reported_as_block_size() -> None:
    result = simulate([request("r1", tokens(32))], page_size=16)
    assert result.block_size == 16


def test_ideal_hit_rate_exceeds_actual_under_pressure() -> None:
    requests = [
        request("r1", tokens(50, start=0)),
        request("r2", tokens(50, start=1000)),
        request("r3", tokens(50, start=0)),
    ]
    squeezed = simulate(requests, kv_budget_tokens=60)
    assert squeezed.hit_rate < squeezed.ideal_hit_rate

    by_id = {r.request_id: r for r in squeezed.per_request}
    assert by_id["r3"].evicted_hit_loss > 0


def test_empty_log_does_not_divide_by_zero() -> None:
    result = simulate([])
    assert result.hit_rate == 0.0
    assert result.per_request == []


def test_repeated_runs_are_byte_identical() -> None:
    requests = [request(f"r{i}", tokens(60, start=i * 7)) for i in range(10)]
    first = simulate(requests, kv_budget_tokens=200)
    second = simulate(requests, kv_budget_tokens=200)
    assert first.model_dump_json() == second.model_dump_json()


# ------------------------------------------------------------------ cross-engine


def test_sglang_never_caches_less_than_vllm() -> None:
    """Token granularity is a superset of block granularity.

    If this ever inverts, one of the two simulators is wrong.
    """
    from kvlint.sim import vllm_blocks

    requests = [
        request("r1", tokens(100)),
        request("r2", tokens(73) + tokens(27, start=500)),
        request("r3", tokens(45) + tokens(55, start=800)),
        request("r4", tokens(100)),
    ]
    vllm = vllm_blocks.simulate(requests)
    sglang = simulate(requests)
    assert sglang.hit_rate >= vllm.hit_rate


# ------------------------------------------------------------------ validation


def test_rejects_invalid_page_size() -> None:
    with pytest.raises(ValueError, match="page_size"):
        SGLangRadixSimulator(page_size=0)


def test_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="kv_budget_tokens"):
        SGLangRadixSimulator(kv_budget_tokens=0)
