"""Lint engine tests (M4 acceptance).

The two criteria that matter: `dynamic_time` fires on an injected timestamp for
almost every request, and nothing fires on a clean log.
"""

from __future__ import annotations

from typing import Any

import pytest

from kvlint.lint.attribution import attribute, shared_prefix_length
from kvlint.lint.engine import exceeds, run, worst_severity
from kvlint.models import LintFinding, Message, Request, TokenizedRequest
from kvlint.tokenize.renderer import tokenize_requests
from tests.conftest import TINY_TOKENIZER_DIR

MODEL = str(TINY_TOKENIZER_DIR)

# Long enough that the shared prefix comfortably exceeds one block.
BASE_SYSTEM = (
    "You are a support assistant for Acme Corp. "
    "Answer concisely and always cite the knowledge base article you used. "
    "Never speculate about delivery dates you cannot verify."
)


def chat(request_id: str, system: str, user: str) -> Request:
    return Request(
        request_id=request_id,
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
    )


def lint(requests: list[Request], block_size: int = 16) -> list[LintFinding]:
    return run(requests, tokenize_requests(MODEL, requests), block_size)


def by_rule(findings: list[LintFinding]) -> dict[str, LintFinding]:
    return {f.rule_id: f for f in findings}


# ------------------------------------------------------------------ attribution


def tok(request_id: str, token_ids: list[int]) -> TokenizedRequest:
    return TokenizedRequest(
        request_id=request_id,
        rendered_text="x" * len(token_ids),
        token_ids=token_ids,
        char_offsets=[(i, i + 1) for i in range(len(token_ids))],
    )


def test_attribution_is_token_exact() -> None:
    """Not rounded to a block, which is the whole point of decision D2."""
    results = attribute([tok("a", list(range(100))), tok("b", [*range(37), *range(500, 563)])])
    assert results[0].divergence_token_idx is None
    assert results[1].divergence_token_idx == 37
    assert results[1].nearest_prior_request_id == "a"


def test_attribution_reports_tokens_lost_after_the_divergence() -> None:
    results = attribute([tok("a", list(range(100))), tok("b", [*range(37), *range(500, 563)])])
    assert results[1].tokens_lost == 100 - 37


def test_shared_prefix_length_across_the_log() -> None:
    assert shared_prefix_length([tok("a", list(range(50))), tok("b", list(range(20)))]) == 20


def test_shared_prefix_length_of_an_empty_log() -> None:
    assert shared_prefix_length([]) == 0


# ------------------------------------------------------------------ the false-positive guard


def aligned_clean_log(count: int = 12) -> list[Request]:
    """A well-built log whose shared prefix ends exactly on a block boundary.

    Padding the system prompt to alignment is itself the fix `block_straddle`
    recommends, so a log that has applied it must produce no findings at all.
    """
    for pad in range(0, 64):
        requests = [
            chat(f"r{i}", BASE_SYSTEM + "." * pad, f"Where is order {chr(ord('A') + i)}?")
            for i in range(count)
        ]
        tokenized = tokenize_requests(MODEL, requests)
        divergences = [a.divergence_token_idx for a in attribute(tokenized)][1:]
        if divergences and all(d is not None and d % 16 == 0 for d in divergences):
            return requests
    raise AssertionError("could not build a block-aligned clean log")


def test_no_rule_fires_on_a_clean_log() -> None:
    """M4 acceptance, and the criterion that decides whether anyone keeps the tool."""
    findings = lint(aligned_clean_log())
    assert findings == [], [f.rule_id for f in findings]


def test_clean_log_really_does_diverge() -> None:
    """Guards the guard: a log with no divergences would pass trivially."""
    tokenized = tokenize_requests(MODEL, aligned_clean_log())
    assert any(a.divergence_token_idx is not None for a in attribute(tokenized))


# ------------------------------------------------------------------ injected anti-patterns


def test_dynamic_time_fires_on_nearly_every_request() -> None:
    """M4 acceptance: affected_fraction above 0.9."""
    requests = [
        chat(
            f"r{i}", f"{BASE_SYSTEM}\nCurrent time: 2026-09-11T08:{i:02d}:00Z", "Where is my order?"
        )
        for i in range(20)
    ]
    finding = by_rule(lint(requests))["dynamic_time"]
    assert finding.affected_fraction > 0.9
    assert finding.severity == "high"
    assert finding.auto_fixable


def test_dynamic_id_fires_on_rotating_request_ids() -> None:
    requests = [
        chat(f"r{i}", f"{BASE_SYSTEM}\nTrace: trace_{i:08d}ab", "Where is my order?")
        for i in range(20)
    ]
    assert "dynamic_id" in by_rule(lint(requests))


def test_user_in_system_fires_on_personalized_system_prompts() -> None:
    requests = [
        chat(f"r{i}", f"{BASE_SYSTEM}\nuser: person{i}@example.com", "Where is my order?")
        for i in range(20)
    ]
    assert "user_in_system" in by_rule(lint(requests))


def test_short_prefix_fires_when_requests_share_nothing() -> None:
    requests = [
        chat(f"r{i}", f"System {i} is entirely different here.", f"Ask {i}") for i in range(6)
    ]
    findings = by_rule(lint(requests))
    assert "short_prefix" in findings
    assert findings["short_prefix"].auto_fixable is False


def test_findings_report_the_requests_they_affect() -> None:
    requests = [
        chat(
            f"r{i}", f"{BASE_SYSTEM}\nCurrent time: 2026-09-11T08:{i:02d}:00Z", "Where is my order?"
        )
        for i in range(10)
    ]
    finding = by_rule(lint(requests))["dynamic_time"]
    assert set(finding.affected_request_ids) <= {f"r{i}" for i in range(10)}
    assert finding.est_tokens_lost > 0


# ------------------------------------------------------------------ ordering and gating


def test_findings_are_ordered_by_severity_then_cost() -> None:
    requests = [
        chat(
            f"r{i}", f"{BASE_SYSTEM}\nCurrent time: 2026-09-11T08:{i:02d}:00Z", "Where is my order?"
        )
        for i in range(20)
    ]
    findings = lint(requests)
    order = {"high": 0, "medium": 1, "low": 2}
    severities = [order[f.severity] for f in findings]
    assert severities == sorted(severities)


def test_worst_severity_of_an_empty_result() -> None:
    assert worst_severity([]) is None


def test_exceeds_matches_at_and_above_the_threshold() -> None:
    low = LintFinding(
        rule_id="block_straddle",
        severity="low",
        affected_request_ids=["a"],
        affected_fraction=1.0,
        est_tokens_lost=4,
        evidence="x",
        suggestion="y",
        auto_fixable=True,
    )
    assert exceeds([low], "low")
    assert not exceeds([low], "high")


# ------------------------------------------------------------------ determinism and edges


def test_repeated_runs_produce_identical_findings() -> None:
    requests = [
        chat(
            f"r{i}", f"{BASE_SYSTEM}\nCurrent time: 2026-09-11T08:{i:02d}:00Z", "Where is my order?"
        )
        for i in range(10)
    ]
    tokenized = tokenize_requests(MODEL, requests)
    first = [f.model_dump() for f in run(requests, tokenized)]
    second = [f.model_dump() for f in run(requests, tokenized)]
    assert first == second


def test_empty_log_produces_no_findings() -> None:
    assert run([], []) == []


def test_single_request_log_produces_no_findings() -> None:
    """Nothing to diverge from, so nothing can be attributed."""
    assert run(*_pair([chat("only", BASE_SYSTEM, "hello")])) == []


def _pair(requests: list[Request]) -> tuple[list[Request], list[Any]]:
    return requests, tokenize_requests(MODEL, requests)


def test_mismatched_input_lengths_are_rejected() -> None:
    requests = [chat("a", BASE_SYSTEM, "x")]
    with pytest.raises(ValueError, match="disagree in length"):
        run(requests, [])
