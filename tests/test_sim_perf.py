"""Simulator throughput (M2 acceptance: 10k x 2k tokens under 30s).

Marked `perf` and deselected by default: it is a real measurement, not a unit
test, and it allocates ~20M token ids. Run it with `pytest -m perf`.
"""

from __future__ import annotations

import time

import pytest

from kvlint.sim.sglang_radix import SGLangRadixSimulator
from kvlint.sim.vllm_blocks import VLLMBlockSimulator

REQUESTS = 10_000
TOKENS_PER_REQUEST = 2_000
BUDGET_SECONDS = 30.0


@pytest.mark.perf
def test_hash_step_meets_the_throughput_budget() -> None:
    # A realistic workload rather than random noise: a long shared system prompt
    # followed by a per-request tail. That exercises the hit path, which is the
    # expensive one, since a full miss stops the prefix walk immediately.
    shared_prefix = list(range(1_500))

    simulator = VLLMBlockSimulator(block_size=16)

    start = time.perf_counter()
    for i in range(REQUESTS):
        tail_base = 10_000 + (i * TOKENS_PER_REQUEST)
        token_ids = shared_prefix + list(range(tail_base, tail_base + 500))
        simulator.feed_tokens(f"r{i}", token_ids)
    elapsed = time.perf_counter() - start

    blocks = REQUESTS * TOKENS_PER_REQUEST // 16
    print(
        f"\n{REQUESTS} requests x {TOKENS_PER_REQUEST} tokens "
        f"({blocks:,} blocks) in {elapsed:.2f}s "
        f"({blocks / elapsed:,.0f} blocks/s)"
    )
    assert elapsed < BUDGET_SECONDS, f"took {elapsed:.2f}s, budget is {BUDGET_SECONDS}s"


@pytest.mark.perf
def test_radix_insert_meets_the_end_to_end_budget() -> None:
    """The binding constraint, not the block hashing.

    A pure-Python radix tree over 20M tokens does far more work than 1.25M
    hashes, so this is the number that decides whether build plan section 2's
    60s end-to-end budget holds.
    """
    shared_prefix = list(range(1_500))

    simulator = SGLangRadixSimulator()

    start = time.perf_counter()
    for i in range(REQUESTS):
        tail_base = 10_000 + (i * TOKENS_PER_REQUEST)
        token_ids = shared_prefix + list(range(tail_base, tail_base + 500))
        simulator.feed_tokens(f"r{i}", token_ids)
    elapsed = time.perf_counter() - start

    total_tokens = REQUESTS * TOKENS_PER_REQUEST
    print(
        f"\n{REQUESTS} requests x {TOKENS_PER_REQUEST} tokens "
        f"({total_tokens:,} tokens) in {elapsed:.2f}s "
        f"({total_tokens / elapsed:,.0f} tokens/s)"
    )
    assert elapsed < BUDGET_SECONDS, f"took {elapsed:.2f}s, budget is {BUDGET_SECONDS}s"
