"""Model round-trip and validation tests (M0 acceptance criterion)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kvlint.models import (
    BeforeAfter,
    LintFinding,
    Message,
    PerRequestResult,
    Report,
    Request,
    SimResult,
    TokenizedRequest,
)


def test_request_round_trips() -> None:
    req = Request(
        request_id="r1",
        session_id="s1",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        timestamp=1234.5,
    )
    assert Request.model_validate(json.loads(req.model_dump_json())) == req


def test_request_defaults_are_optional() -> None:
    """The plain ingester supplies only raw_prompt; everything else must default."""
    req = Request(request_id="r1", raw_prompt="hello")
    assert req.messages == []
    assert req.tools is None
    assert req.session_id is None


def test_tokenized_request_round_trips_offsets_as_tuples() -> None:
    tr = TokenizedRequest(
        request_id="r1",
        rendered_text="hi there",
        token_ids=[15339, 1070],
        char_offsets=[(0, 2), (2, 8)],
    )
    revived = TokenizedRequest.model_validate(json.loads(tr.model_dump_json()))
    assert revived == tr
    # JSON has no tuple type; pydantic must coerce the lists back.
    assert all(isinstance(o, tuple) for o in revived.char_offsets)


def test_full_report_round_trips() -> None:
    report = Report(
        input_summary={"n_requests": 2, "format": "openai"},
        sims=[
            SimResult(
                engine="vllm",
                config={"block_size": 16, "kv_budget_blocks": None},
                per_request=[
                    PerRequestResult(request_id="r1", prompt_tokens=100, cached_tokens=0),
                    PerRequestResult(
                        request_id="r2",
                        prompt_tokens=100,
                        cached_tokens=32,
                        divergence_token_idx=32,
                        nearest_prior_request_id="r1",
                        evicted_hit_loss=16,
                    ),
                ],
                hit_rate=0.16,
                ideal_hit_rate=0.24,
                block_size=16,
            )
        ],
        findings=[
            LintFinding(
                rule_id="dynamic_time",
                severity="high",
                affected_request_ids=["r2"],
                affected_fraction=0.5,
                est_tokens_lost=68,
                evidence="Current time: 2026-09-10T14:03:11Z",
                suggestion="Move the timestamp to the end of the system prompt.",
                auto_fixable=True,
            )
        ],
        before_after=BeforeAfter(
            hit_rate_before=0.16,
            hit_rate_after=0.61,
            per_engine={"vllm": {"before": 0.16, "after": 0.61}},
        ),
        version="0.0.1",
    )
    assert Report.model_validate(json.loads(report.model_dump_json())) == report


def test_report_serialization_is_deterministic() -> None:
    """Byte-identical JSON for identical input is a hard requirement (§2)."""
    report = Report(input_summary={"b": 1, "a": 2}, sims=[], findings=[], version="0.0.1")
    assert report.model_dump_json() == report.model_dump_json()


def test_unknown_fields_are_rejected() -> None:
    """Malformed logs should fail at ingest, not silently drop data."""
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": "hi", "typo_field": 1})


def test_invalid_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "wizard", "content": "hi"})


def test_invalid_engine_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SimResult.model_validate(
            {
                "engine": "tensorrt",
                "config": {},
                "per_request": [],
                "hit_rate": 0.0,
                "ideal_hit_rate": 0.0,
            }
        )
