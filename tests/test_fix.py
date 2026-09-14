"""Rewriter, estimate and writer tests (M5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kvlint.errors import KvlintError
from kvlint.fix.estimate import Rates, deltas, mean_ttft_ms, total_cost
from kvlint.fix.rewriter import (
    CONTEXT_HEADER,
    canonicalize,
    fix_requests,
    normalize_text,
    volatile_shapes,
)
from kvlint.fix.writer import write
from kvlint.ingest import load as load_log
from kvlint.models import Message, PerRequestResult, Request, SimResult

STABLE = "You are a support assistant.\nAnswer concisely and cite your source."


def chat(request_id: str, system: str, user: str = "Where is my order?") -> Request:
    return Request(
        request_id=request_id,
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
    )


def system_of(request: Request) -> str:
    return next(m.content for m in request.messages if m.role == "system")


# ------------------------------------------------------------------ normalization


def test_normalize_converts_crlf_to_lf() -> None:
    assert normalize_text("a\r\nb") == "a\nb"


def test_normalize_strips_trailing_spaces() -> None:
    assert normalize_text("a   \nb\t\n") == "a\nb\n"


def test_normalize_applies_nfc() -> None:
    decomposed = "café"
    assert normalize_text(decomposed) == "café"


def test_normalize_leaves_clean_text_alone() -> None:
    assert normalize_text("already clean\n") == "already clean\n"


# ------------------------------------------------------------------ JSON ordering


def test_canonicalize_sorts_keys_recursively() -> None:
    value = {"b": 1, "a": {"d": 2, "c": 3}}
    assert json.dumps(canonicalize(value)) == '{"a": {"c": 3, "d": 2}, "b": 1}'


def test_canonicalize_preserves_list_order() -> None:
    """Lists are ordered data; only dict keys are arbitrary."""
    assert canonicalize([3, 1, 2]) == [3, 1, 2]


def test_canonicalize_leaves_scalars_alone() -> None:
    assert canonicalize("x") == "x"


def test_tools_are_canonicalized() -> None:
    requests = [
        Request(
            request_id="a", tools=[{"b": 1, "a": 2}], messages=[Message(role="user", content="x")]
        )
    ]
    fixed = fix_requests(requests).requests[0]
    assert fixed.tools is not None
    assert list(fixed.tools[0]) == ["a", "b"]


def test_fenced_json_in_content_is_canonicalized() -> None:
    body = 'Schema:\n```json\n{"b": 1, "a": 2}\n```'
    requests = [chat("a", body), chat("b", body)]
    fixed = fix_requests(requests).requests[0]
    assert '"a": 2' in system_of(fixed)
    assert system_of(fixed).index('"a"') < system_of(fixed).index('"b"')


# ------------------------------------------------------------------ volatility


def test_volatile_shapes_finds_a_changing_field() -> None:
    requests = [
        chat("a", "Current time: 2026-09-11T08:00:00Z\n" + STABLE),
        chat("b", "Current time: 2026-09-11T08:01:00Z\n" + STABLE),
    ]
    assert volatile_shapes(requests)


def test_volatile_shapes_ignores_a_constant_field() -> None:
    """A fixed date breaks nothing, so rewriting it would be pure churn."""
    same = "Knowledge cutoff: 2024-06-01\n" + STABLE
    assert volatile_shapes([chat("a", same), chat("b", same)]) == set()


# ------------------------------------------------------------------ relocation


def test_volatile_line_moves_below_the_stable_text() -> None:
    requests = [chat("a", f"Current time: 2026-09-11T08:0{i}:00Z\n" + STABLE) for i in range(2)]
    fixed = fix_requests(requests).requests[0]
    system = system_of(fixed)
    assert system.startswith("You are a support assistant.")
    assert CONTEXT_HEADER in system
    assert system.index(CONTEXT_HEADER) > system.index("cite your source")


def test_relocation_preserves_the_content() -> None:
    """Moving is safe; deleting would change what the model sees."""
    requests = [chat("a", f"Current time: 2026-09-11T08:0{i}:00Z\n" + STABLE) for i in range(2)]
    fixed = fix_requests(requests).requests[0]
    assert "2026-09-11T08:00:00Z" in system_of(fixed)


def test_already_trailing_volatiles_are_left_alone() -> None:
    """Relocating a trailing line is a no-op that still costs header tokens.

    Regression guard: an earlier version lowered the hit rate by 2.3 pp here.
    """
    requests = [chat("a", STABLE + f"\nCurrent time: 2026-09-11T08:0{i}:00Z") for i in range(2)]
    report = fix_requests(requests)
    assert report.changed == 0
    assert CONTEXT_HEADER not in system_of(report.requests[0])


def test_user_start_target_moves_into_the_last_user_turn() -> None:
    requests = [chat("a", f"Current time: 2026-09-11T08:0{i}:00Z\n" + STABLE) for i in range(2)]
    fixed = fix_requests(requests, target="user_start").requests[0]
    user = next(m.content for m in fixed.messages if m.role == "user")
    assert CONTEXT_HEADER in user
    assert "2026-09-11" not in system_of(fixed)


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="target must be"):
        fix_requests([chat("a", STABLE)], target="sideways")


def test_report_counts_what_changed() -> None:
    requests = [chat(f"r{i}", f"Current time: 2026-09-11T08:0{i}:00Z\n" + STABLE) for i in range(3)]
    report = fix_requests(requests)
    assert report.changed == 3
    assert report.moved_lines == 3


def test_a_clean_log_is_left_untouched() -> None:
    requests = [chat(f"r{i}", STABLE, f"Question {i}") for i in range(4)]
    report = fix_requests(requests)
    assert report.changed == 0
    assert report.requests == requests


def test_request_without_a_system_message_still_keeps_content() -> None:
    requests = [
        Request(
            request_id=f"r{i}",
            messages=[Message(role="user", content=f"ref req_0000000{i}a now")],
        )
        for i in range(2)
    ]
    fixed = fix_requests(requests).requests[0]
    assert "req_00000000a" in "".join(m.content for m in fixed.messages)


# ------------------------------------------------------------------ estimates


def sim(prompt: int, cached: int, count: int = 1) -> SimResult:
    return SimResult(
        engine="vllm",
        config={},
        per_request=[
            PerRequestResult(request_id=f"r{i}", prompt_tokens=prompt, cached_tokens=cached)
            for i in range(count)
        ],
        hit_rate=cached / prompt if prompt else 0.0,
        ideal_hit_rate=cached / prompt if prompt else 0.0,
    )


def test_ttft_counts_only_uncached_tokens() -> None:
    rates = Rates(prefill_ms_per_1k_tokens=40.0)
    assert mean_ttft_ms(sim(2000, 1000), rates) == pytest.approx(40.0)


def test_ttft_of_a_full_hit_is_zero() -> None:
    assert mean_ttft_ms(sim(1000, 1000), Rates()) == 0.0


def test_ttft_of_an_empty_result_is_zero() -> None:
    assert mean_ttft_ms(sim(0, 0, count=0), Rates()) == 0.0


def test_cost_is_none_without_a_price() -> None:
    assert total_cost(sim(1000, 500), Rates()) is None


def test_cost_discounts_cached_tokens() -> None:
    rates = Rates(price_per_1m_input=1.0, cached_discount=0.9)
    # 500 uncached at full price plus 500 cached at 10 percent.
    assert total_cost(sim(1000, 500), rates) == pytest.approx(550 / 1_000_000)


def test_deltas_are_positive_when_the_fix_helps() -> None:
    rates = Rates(price_per_1m_input=1.0)
    ttft, cost = deltas(sim(1000, 0), sim(1000, 900), rates)
    assert ttft > 0
    assert cost is not None and cost > 0


def test_invalid_discount_is_rejected() -> None:
    with pytest.raises(ValueError, match="cached_discount"):
        Rates(cached_discount=1.5)


def test_negative_prefill_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="prefill_ms_per_1k_tokens"):
        Rates(prefill_ms_per_1k_tokens=-1)


# ------------------------------------------------------------------ round-trip


def test_openai_output_round_trips(tmp_path: Path) -> None:
    """M5 acceptance: the fixed log re-parses with the same ingester."""
    requests = [chat(f"r{i}", STABLE, f"Question {i}") for i in range(3)]
    out = tmp_path / "fixed.jsonl"
    write(out, requests, "openai")
    assert load_log(out, "openai") == requests


def test_plain_output_round_trips(tmp_path: Path) -> None:
    requests = [Request(request_id=f"r{i}", raw_prompt=f"prompt {i}") for i in range(3)]
    out = tmp_path / "fixed.jsonl"
    write(out, requests, "plain")
    assert load_log(out, "plain") == requests


def test_sharegpt_output_round_trips(tmp_path: Path, fixtures_dir: Path) -> None:
    """Collapsing per-turn requests back into conversations must be lossless."""
    original = load_log(fixtures_dir / "sharegpt.jsonl", "sharegpt")
    out = tmp_path / "fixed.jsonl"
    write(out, original, "sharegpt")
    assert load_log(out, "sharegpt") == original


def test_writing_plain_without_a_prompt_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(KvlintError, match="no raw prompt"):
        write(tmp_path / "x.jsonl", [chat("a", STABLE)], "plain")


def test_unknown_output_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KvlintError, match="cannot write format"):
        write(tmp_path / "x.jsonl", [chat("a", STABLE)], "vllm")


def test_output_uses_lf_on_every_platform(tmp_path: Path) -> None:
    """The whitespace rules treat CRLF as a defect, so output must not emit it."""
    out = tmp_path / "fixed.jsonl"
    write(out, [chat("a", STABLE)], "openai")
    assert b"\r\n" not in out.read_bytes()
