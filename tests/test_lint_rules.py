"""Per-rule unit tests: a positive and a negative case for each (M4 acceptance).

Rules are pure functions over a synthetic window, so these never touch a
tokenizer or a simulator. The negative cases carry the weight: a linter that
fires on clean input gets switched off and never switched back on.
"""

from __future__ import annotations

from kvlint.lint.context import CorpusContext, RuleContext
from kvlint.lint.rules import (
    block_straddle,
    dynamic_id,
    dynamic_time,
    key_order,
    short_prefix,
    user_in_system,
    volatile_fewshot,
    whitespace_drift,
)
from kvlint.models import Message, Request, TokenizedRequest


def ctx(
    current_after: str,
    prior_after: str = "",
    *,
    in_system_prompt: bool = True,
    divergence: int = 100,
    block_size: int = 16,
    tools: list[dict] | None = None,
    prior_tools: list[dict] | None = None,
) -> RuleContext:
    request = Request(request_id="cur", messages=[Message(role="user", content="x")], tools=tools)
    prior = Request(
        request_id="old", messages=[Message(role="user", content="x")], tools=prior_tools
    )
    tokenized = TokenizedRequest(
        request_id="cur", rendered_text=current_after, token_ids=[1], char_offsets=[(0, 1)]
    )
    return RuleContext(
        request=request,
        tokenized=tokenized,
        prior=prior,
        prior_tokenized=None,
        divergence_token_idx=divergence,
        divergence_char_pos=0,
        before="You are a helpful assistant. ",
        current_after=current_after,
        prior_after=prior_after,
        current_window=current_after,
        prior_window=prior_after,
        block_size=block_size,
        in_system_prompt=in_system_prompt,
    )


# ------------------------------------------------------------------ dynamic_time


def test_dynamic_time_fires_on_a_changing_iso_timestamp() -> None:
    assert dynamic_time.detect(
        ctx("2026-09-11T08:00:00Z. Answer briefly.", "2026-09-10T08:00:00Z. Answer briefly.")
    )


def test_dynamic_time_fires_on_a_changing_clock_time() -> None:
    assert dynamic_time.detect(ctx("at 14:03 today", "at 09:21 today"))


def test_dynamic_time_fires_on_a_changing_epoch() -> None:
    assert dynamic_time.detect(ctx("ts=1757577600 end", "ts=1757491200 end"))


def test_dynamic_time_ignores_an_unchanged_timestamp() -> None:
    """A fixed date in the template is not what broke the cache."""
    same = "Knowledge cutoff 2024-06-01. Answer briefly."
    assert dynamic_time.detect(ctx(same, same)) is None


def test_dynamic_time_ignores_a_date_only_one_side_mentions() -> None:
    """A user asking about a date is ordinary traffic, not a template bug."""
    assert dynamic_time.detect(ctx("what happened on 2026-01-05?", "how do refunds work?")) is None


def test_dynamic_time_ignores_ordinary_differing_text() -> None:
    assert dynamic_time.detect(ctx("Ticket 12: where is my order", "Ticket 9: cancel it")) is None


# ------------------------------------------------------------------ dynamic_id


def test_dynamic_id_fires_on_a_changing_uuid() -> None:
    assert dynamic_id.detect(
        ctx(
            "request 3f2504e0-4f89-11d3-9a0c-0305e82c3301 begins",
            "request 8a1b22c3-4f89-11d3-9a0c-0305e82c3399 begins",
        )
    )


def test_dynamic_id_fires_on_a_prefixed_correlation_id() -> None:
    assert dynamic_id.detect(ctx("trace_9fbc21aa handling", "trace_02de77bb handling"))


def test_dynamic_id_fires_on_a_changing_counter() -> None:
    assert dynamic_id.detect(ctx("seq 1000241 of", "seq 1000240 of"))


def test_dynamic_id_ignores_a_stable_id() -> None:
    same = "tenant acme-corp, region eu-west"
    assert dynamic_id.detect(ctx(same, same)) is None


def test_dynamic_id_ignores_short_numbers() -> None:
    """Order counts and prices must not look like correlation ids."""
    assert dynamic_id.detect(ctx("you ordered 3 items", "you ordered 12 items")) is None


# ------------------------------------------------------------------ whitespace_drift


def test_whitespace_drift_fires_on_crlf_versus_lf() -> None:
    found = whitespace_drift.detect(ctx("Rules:\r\nBe brief.", "Rules:\nBe brief."))
    assert found is not None
    assert "CRLF" in found


def test_whitespace_drift_fires_on_trailing_space() -> None:
    assert whitespace_drift.detect(ctx("Be concise. ", "Be concise."))


def test_whitespace_drift_fires_on_unicode_form() -> None:
    """Same glyphs, different normalization form, different token ids."""
    found = whitespace_drift.detect(ctx("café rules", "café rules"))
    assert found is not None
    assert "Unicode" in found


def test_whitespace_drift_ignores_a_real_content_change() -> None:
    assert whitespace_drift.detect(ctx("Be brief.", "Be verbose.")) is None


def test_whitespace_drift_ignores_identical_text() -> None:
    assert whitespace_drift.detect(ctx("Be brief.", "Be brief.")) is None


# ------------------------------------------------------------------ key_order


def test_key_order_fires_when_tools_differ_only_in_order() -> None:
    assert key_order.detect(
        ctx(
            "x",
            "y",
            tools=[{"name": "get_weather", "type": "function"}],
            prior_tools=[{"type": "function", "name": "get_weather"}],
        )
    )


def test_key_order_ignores_genuinely_different_tools() -> None:
    assert (
        key_order.detect(
            ctx("x", "y", tools=[{"name": "a"}], prior_tools=[{"name": "b"}]),
        )
        is None
    )


def test_key_order_ignores_identically_ordered_tools() -> None:
    same = [{"type": "function", "name": "get_weather"}]
    assert key_order.detect(ctx("x", "y", tools=same, prior_tools=list(same))) is None


def test_key_order_fires_on_embedded_json() -> None:
    assert key_order.detect(
        ctx('config {"alpha": 1, "beta": 2} end', 'config {"beta": 2, "alpha": 1} end')
    )


def test_key_order_ignores_json_with_different_values() -> None:
    assert key_order.detect(ctx('{"alpha": 1, "beta": 2}', '{"beta": 9, "alpha": 1}')) is None


# ------------------------------------------------------------------ user_in_system


def test_user_in_system_fires_on_a_changing_email() -> None:
    assert user_in_system.detect(ctx("user ada@example.com asks", "user bob@example.com asks"))


def test_user_in_system_fires_on_a_labelled_field() -> None:
    assert user_in_system.detect(ctx("locale: en-GB now", "locale: fr-FR now"))


def test_user_in_system_ignores_personalization_outside_the_system_prompt() -> None:
    """In the last user turn this is correct practice, not a finding."""
    assert (
        user_in_system.detect(ctx("ada@example.com", "bob@example.com", in_system_prompt=False))
        is None
    )


def test_user_in_system_ignores_a_stable_system_prompt() -> None:
    same = "support@acme.com is the escalation address"
    assert user_in_system.detect(ctx(same, same)) is None


# ------------------------------------------------------------------ block_straddle


def test_block_straddle_fires_when_the_prefix_ends_mid_block() -> None:
    found = block_straddle.detect(ctx("x", "y", divergence=100, block_size=16))
    assert found is not None
    assert "4 cacheable tokens are lost" in found


def test_block_straddle_ignores_a_block_aligned_prefix() -> None:
    assert block_straddle.detect(ctx("x", "y", divergence=96, block_size=16)) is None


def test_block_straddle_ignores_a_trivial_overhang() -> None:
    """Padding two tokens costs more than it saves."""
    assert block_straddle.detect(ctx("x", "y", divergence=98, block_size=16)) is None


# ------------------------------------------------------------------ corpus rules


def tokenized(request_id: str, token_ids: list[int]) -> TokenizedRequest:
    return TokenizedRequest(
        request_id=request_id,
        rendered_text="x" * len(token_ids),
        token_ids=token_ids,
        char_offsets=[(i, i + 1) for i in range(len(token_ids))],
    )


def corpus(
    requests: list[Request],
    tokens: list[TokenizedRequest],
    shared: int,
    block_size: int = 16,
) -> CorpusContext:
    return CorpusContext(
        requests=requests,
        tokenized=tokens,
        divergences={t.request_id: shared for t in tokens},
        shared_prefix_tokens=shared,
        block_size=block_size,
    )


def test_short_prefix_fires_when_nothing_fills_a_block() -> None:
    tokens = [tokenized("a", list(range(50))), tokenized("b", list(range(100, 150)))]
    requests = [Request(request_id="a"), Request(request_id="b")]
    hit = short_prefix.detect(corpus(requests, tokens, shared=3))
    assert hit is not None
    assert hit.affected_request_ids == ["a", "b"]


def test_short_prefix_ignores_a_prefix_that_fills_a_block() -> None:
    tokens = [tokenized("a", list(range(50))), tokenized("b", list(range(50)))]
    requests = [Request(request_id="a"), Request(request_id="b")]
    assert short_prefix.detect(corpus(requests, tokens, shared=32)) is None


def test_short_prefix_ignores_a_single_request_log() -> None:
    tokens = [tokenized("a", list(range(50)))]
    assert short_prefix.detect(corpus([Request(request_id="a")], tokens, shared=0)) is None


def fewshot_request(request_id: str, examples: str) -> Request:
    return Request(
        request_id=request_id,
        messages=[
            Message(role="system", content=f"You are terse.\n{examples}"),
            Message(role="user", content="go"),
        ],
    )


def test_volatile_fewshot_fires_when_exemplars_rotate() -> None:
    requests = [
        fewshot_request("a", "Q: one\nA: 1\nQ: two\nA: 2\nQ: three\nA: 3"),
        fewshot_request("b", "Q: four\nA: 4\nQ: five\nA: 5\nQ: six\nA: 6"),
    ]
    tokens = [tokenized("a", list(range(40))), tokenized("b", list(range(40)))]
    assert volatile_fewshot.detect(corpus(requests, tokens, shared=10)) is not None


def test_volatile_fewshot_ignores_pinned_exemplars() -> None:
    same = "Q: one\nA: 1\nQ: two\nA: 2\nQ: three\nA: 3"
    requests = [fewshot_request("a", same), fewshot_request("b", same)]
    tokens = [tokenized("a", list(range(40))), tokenized("b", list(range(40)))]
    assert volatile_fewshot.detect(corpus(requests, tokens, shared=40)) is None


def test_volatile_fewshot_ignores_prose_that_merely_differs() -> None:
    """System prompts that vary without being exemplar blocks are a different bug."""
    requests = [
        fewshot_request("a", "Be polite to the customer."),
        fewshot_request("b", "Be direct with the customer."),
    ]
    tokens = [tokenized("a", list(range(40))), tokenized("b", list(range(40)))]
    assert volatile_fewshot.detect(corpus(requests, tokens, shared=10)) is None
