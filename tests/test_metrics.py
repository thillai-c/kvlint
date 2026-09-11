"""Derived metrics tests."""

from __future__ import annotations

import pytest

from kvlint.models import PerRequestResult, SimResult
from kvlint.sim.metrics import (
    divergence_histogram,
    eviction_loss_fraction,
    full_hit_count,
    miss_count,
    summarize,
)


def result(*records: PerRequestResult) -> SimResult:
    return SimResult(
        engine="vllm",
        config={},
        per_request=list(records),
        hit_rate=0.5,
        ideal_hit_rate=0.75,
        block_size=16,
    )


def record(
    request_id: str,
    prompt: int,
    cached: int,
    divergence: int | None = None,
    loss: int = 0,
) -> PerRequestResult:
    return PerRequestResult(
        request_id=request_id,
        prompt_tokens=prompt,
        cached_tokens=cached,
        divergence_token_idx=divergence,
        evicted_hit_loss=loss,
    )


def test_miss_count_counts_diverged_requests() -> None:
    assert miss_count(result(record("a", 100, 100), record("b", 100, 32, 32))) == 1


def test_full_hit_count_counts_complete_reuse() -> None:
    assert full_hit_count(result(record("a", 100, 100), record("b", 100, 32, 32))) == 1


def test_eviction_loss_is_a_fraction_of_all_prompt_tokens() -> None:
    metrics = result(record("a", 100, 50, 50, loss=25), record("b", 100, 100))
    assert eviction_loss_fraction(metrics) == pytest.approx(25 / 200)


def test_eviction_loss_handles_an_empty_log() -> None:
    assert eviction_loss_fraction(result()) == 0.0


def test_divergence_histogram_buckets_by_position() -> None:
    metrics = result(
        record("a", 100, 0, 0),
        record("b", 100, 16, 16),
        record("c", 100, 64, 64),
        record("d", 100, 100),
    )
    assert divergence_histogram(metrics, bucket_size=64) == {0: 2, 64: 1}


def test_divergence_histogram_is_sorted_for_deterministic_output() -> None:
    metrics = result(record("a", 300, 256, 256), record("b", 300, 0, 0))
    assert list(divergence_histogram(metrics, bucket_size=64)) == [0, 256]


def test_divergence_histogram_rejects_a_zero_bucket() -> None:
    with pytest.raises(ValueError, match="bucket_size"):
        divergence_histogram(result(), bucket_size=0)


def test_summarize_exposes_every_table_field() -> None:
    row = summarize(result(record("a", 100, 50, 50, loss=10)))
    assert row["engine"] == "vllm"
    assert row["requests"] == 1
    assert row["prompt_tokens"] == 100
    assert row["cached_tokens"] == 50
    assert row["block_size"] == 16
