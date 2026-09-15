"""Live validation tests against a mocked server (M6).

No real engine here: `httpx.MockTransport` stands in for vLLM, with counters that
behave the way the real ones do. The real number comes from a GPU box and is
recorded in docs/VALIDATION.md, never fabricated here.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from kvlint.errors import KvlintError
from kvlint.live.prometheus import first_present, parse
from kvlint.live.replay import replay, reset_prefix_cache
from kvlint.live.sglang_metrics import scrape_hit_rate
from kvlint.live.validate import validate_sglang, validate_vllm
from kvlint.live.vllm_metrics import CacheCounters, scrape
from kvlint.models import Message, Request, TokenizedRequest

# ------------------------------------------------------------------ prometheus


def test_parse_reads_an_unlabelled_metric() -> None:
    assert parse("vllm:prefix_cache_hits_total 42.0") == {"vllm:prefix_cache_hits_total": 42.0}


def test_parse_sums_across_label_sets() -> None:
    """One line per model or engine core; we want the total."""
    text = (
        'vllm:prefix_cache_hits_total{model_name="a"} 10.0\n'
        'vllm:prefix_cache_hits_total{model_name="b"} 5.0\n'
    )
    assert parse(text)["vllm:prefix_cache_hits_total"] == 15.0


def test_parse_skips_comments_and_blank_lines() -> None:
    text = "# HELP vllm:x help\n# TYPE vllm:x counter\n\nvllm:x 1.0\n"
    assert parse(text) == {"vllm:x": 1.0}


def test_parse_ignores_unparseable_values() -> None:
    assert parse("good 1.0\nbad not_a_number\n") == {"good": 1.0}


def test_first_present_prefers_the_earlier_spelling() -> None:
    metrics = {"vllm:prefix_cache_hits": 1.0, "vllm:prefix_cache_hits_total": 2.0}
    assert first_present(metrics, ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")) == 2.0


def test_first_present_returns_none_when_absent() -> None:
    assert first_present({}, ("nope",)) is None


# ------------------------------------------------------------------ counters


def test_counter_difference_isolates_the_replay_window() -> None:
    """The reason vLLM validation is exact: counters can be windowed."""
    window = CacheCounters(queries=300, hits=200) - CacheCounters(queries=100, hits=50)
    assert window.queries == 200
    assert window.hits == 150
    assert window.hit_rate == pytest.approx(0.75)


def test_hit_rate_is_none_when_nothing_was_queried() -> None:
    """None rather than zero, so callers cannot read 'no data' as 'zero percent'."""
    assert CacheCounters(queries=0, hits=0).hit_rate is None


# ------------------------------------------------------------------ fake server


class FakeVLLM:
    """A vLLM-shaped server whose counters move the way the real ones do."""

    def __init__(self, hit_fraction: float, prompt_tokens: int = 100) -> None:
        self.hit_fraction = hit_fraction
        self.prompt_tokens = prompt_tokens
        self.queries = 1000.0  # pre-existing traffic the window must exclude
        self.hits = 900.0
        self.requests_seen = 0
        self.reset_calls = 0
        self.expose_metrics = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/metrics":
            if not self.expose_metrics:
                return httpx.Response(200, text="vllm:num_requests_running 0.0\n")
            return httpx.Response(
                200,
                text=(
                    "# TYPE vllm:prefix_cache_queries_total counter\n"
                    f'vllm:prefix_cache_queries_total{{model_name="m"}} {self.queries}\n'
                    f'vllm:prefix_cache_hits_total{{model_name="m"}} {self.hits}\n'
                ),
            )

        if path == "/reset_prefix_cache":
            self.reset_calls += 1
            return httpx.Response(200, json={"success": True})

        if path in ("/v1/chat/completions", "/v1/completions"):
            self.requests_seen += 1
            self.queries += self.prompt_tokens
            self.hits += self.prompt_tokens * self.hit_fraction
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "x"}}],
                    "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 1},
                },
            )

        return httpx.Response(404)


def client_for(server: Any) -> httpx.Client:
    return httpx.Client(base_url="http://fake", transport=httpx.MockTransport(server.handler))


def chat(request_id: str, user: str) -> Request:
    return Request(
        request_id=request_id,
        messages=[
            Message(role="system", content="You are terse."),
            Message(role="user", content=user),
        ],
    )


def tokens(request_id: str, count: int, start: int = 0) -> TokenizedRequest:
    ids = list(range(start, start + count))
    return TokenizedRequest(
        request_id=request_id,
        rendered_text="x" * count,
        token_ids=ids,
        char_offsets=[(i, i + 1) for i in range(count)],
    )


# ------------------------------------------------------------------ scraping


def test_scrape_reads_both_counters() -> None:
    server = FakeVLLM(hit_fraction=0.5)
    with client_for(server) as client:
        counters = scrape(client)
    assert counters.queries == 1000.0
    assert counters.hits == 900.0


def test_scrape_fails_clearly_without_prefix_cache_metrics() -> None:
    server = FakeVLLM(hit_fraction=0.5)
    server.expose_metrics = False
    with client_for(server) as client, pytest.raises(KvlintError, match="prefix-cache counters"):
        scrape(client)


# ------------------------------------------------------------------ replay


def test_replay_sends_every_request_once() -> None:
    server = FakeVLLM(hit_fraction=0.5)
    requests = [chat(f"r{i}", f"q{i}") for i in range(4)]
    with client_for(server) as client:
        result = replay(client, requests, [tokens(f"r{i}", 100) for i in range(4)], "m")
    assert server.requests_seen == 4
    assert len(result.checks) == 4


def test_replay_flags_a_token_count_mismatch() -> None:
    """A mismatch means our chat template differs from the server's."""
    server = FakeVLLM(hit_fraction=0.5, prompt_tokens=100)
    requests = [chat("r0", "q")]
    with client_for(server) as client:
        result = replay(client, requests, [tokens("r0", 97)], "m")
    assert len(result.mismatches) == 1
    assert result.mismatches[0].local == 97
    assert result.mismatches[0].server == 100


def test_replay_records_server_errors_without_aborting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler)) as client:
        result = replay(client, [chat("r0", "q")], [tokens("r0", 10)], "m")
    assert result.failures and "500" in result.failures[0]


def test_replay_uses_the_completions_endpoint_for_plain_prompts() -> None:
    """Plain prompts must not be wrapped in a chat template by the server."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 5}})

    with httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler)) as client:
        replay(client, [Request(request_id="p", raw_prompt="hello")], [tokens("p", 5)], "m")
    assert seen == ["/v1/completions"]


def test_replay_requests_a_single_token() -> None:
    """Generation is irrelevant; we only want the prefill accounted for."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"prompt_tokens": 5}})

    with httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler)) as client:
        replay(client, [chat("r", "q")], [tokens("r", 5)], "m")
    assert bodies[0]["max_tokens"] == 1


def test_reset_prefix_cache_reports_success() -> None:
    server = FakeVLLM(hit_fraction=0.0)
    with client_for(server) as client:
        assert reset_prefix_cache(client) is True
    assert server.reset_calls == 1


def test_reset_prefix_cache_reports_failure_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler)) as client:
        assert reset_prefix_cache(client) is False


# ------------------------------------------------------------------ validation


def test_validation_agrees_when_the_server_matches_the_simulation() -> None:
    """Two identical 96-token requests: vLLM caches 96 of 192, so 50 percent."""
    requests = [chat("r0", "q"), chat("r1", "q")]
    tokenized = [tokens("r0", 96), tokens("r1", 96)]

    server = FakeVLLM(hit_fraction=0.5, prompt_tokens=96)
    with client_for(server) as client:
        result = validate_vllm(client, requests, tokenized, "m")

    assert result.simulated_hit_rate == pytest.approx(0.5)
    assert result.real_hit_rate == pytest.approx(0.5)
    assert result.abs_error_pp == pytest.approx(0.0)
    assert result.within_tolerance
    assert result.tokens_agree


def test_validation_excludes_traffic_from_before_the_replay() -> None:
    """The counters start at 90 percent; the window must not inherit that."""
    requests = [chat("r0", "q"), chat("r1", "q")]
    tokenized = [tokens("r0", 96), tokens("r1", 96)]

    server = FakeVLLM(hit_fraction=0.5, prompt_tokens=96)
    server.queries, server.hits = 1_000_000.0, 900_000.0
    with client_for(server) as client:
        result = validate_vllm(client, requests, tokenized, "m")

    assert result.real_hit_rate == pytest.approx(0.5)


def test_validation_fails_tolerance_when_the_server_disagrees() -> None:
    requests = [chat("r0", "q"), chat("r1", "q")]
    tokenized = [tokens("r0", 96), tokens("r1", 96)]

    server = FakeVLLM(hit_fraction=0.9, prompt_tokens=96)
    with client_for(server) as client:
        result = validate_vllm(client, requests, tokenized, "m")

    assert result.within_tolerance is False
    assert result.abs_error_pp is not None and result.abs_error_pp > 3.0


def test_validation_resets_the_cache_by_default() -> None:
    server = FakeVLLM(hit_fraction=0.5, prompt_tokens=96)
    with client_for(server) as client:
        result = validate_vllm(client, [chat("r", "q")], [tokens("r", 96)], "m")
    assert server.reset_calls == 1
    assert result.cache_was_reset


def test_validation_can_skip_the_reset() -> None:
    server = FakeVLLM(hit_fraction=0.5, prompt_tokens=96)
    with client_for(server) as client:
        validate_vllm(client, [chat("r", "q")], [tokens("r", 96)], "m", reset_cache=False)
    assert server.reset_calls == 0


def test_validation_notes_when_the_window_had_no_queries() -> None:
    """An idle counter means no comparison, which must not read as zero percent."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=("vllm:prefix_cache_queries_total 5.0\nvllm:prefix_cache_hits_total 5.0\n"),
            )
        return httpx.Response(200, json={"usage": {"prompt_tokens": 96}})

    with httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler)) as client:
        result = validate_vllm(client, [chat("r", "q")], [tokens("r", 96)], "m")

    assert result.real_hit_rate is None
    assert result.abs_error_pp is None
    assert any("nothing to compare" in note for note in result.notes)


# ------------------------------------------------------------------ sglang


class FakeSGLang:
    def __init__(self, hit_rate: float) -> None:
        self.hit_rate = hit_rate

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text=f"sglang:cache_hit_rate {self.hit_rate}\n")
        if "reset" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 96}})


def test_sglang_gauge_is_read_as_a_fraction() -> None:
    with client_for(FakeSGLang(0.42)) as client:
        assert scrape_hit_rate(client) == pytest.approx(0.42)


def test_sglang_gauge_expressed_as_a_percentage_is_normalized() -> None:
    with client_for(FakeSGLang(42.0)) as client:
        assert scrape_hit_rate(client) == pytest.approx(0.42)


def test_sglang_validation_is_marked_indicative() -> None:
    """A gauge covers the server's whole history, so it cannot be exact."""
    with client_for(FakeSGLang(0.5)) as client:
        result = validate_sglang(client, [chat("r", "q")], [tokens("r", 96)], "m")

    assert result.exact is False
    assert any("gauge" in note for note in result.notes)


def test_sglang_missing_metric_fails_clearly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="sglang:num_running_reqs 0.0\n")

    with (
        httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(KvlintError, match="cache_hit_rate"),
    ):
        scrape_hit_rate(client)
