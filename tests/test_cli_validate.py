"""End-to-end `validate` tests against a mocked server."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from kvlint.cli import ExitCode, app
from tests.conftest import TINY_TOKENIZER_DIR

runner = CliRunner()
MODEL = str(TINY_TOKENIZER_DIR)


def write_log(path: Path, count: int = 4) -> Path:
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are a terse support assistant."},
                    {"role": "user", "content": f"Where is order {i}?"},
                ]
            }
        )
        for i in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the client factory so `validate` talks to a MockTransport."""
    state: dict[str, Any] = {"queries": 0.0, "hits": 0.0, "hit_fraction": 0.0, "tokens": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=(
                    f"vllm:prefix_cache_queries_total {state['queries']}\n"
                    f"vllm:prefix_cache_hits_total {state['hits']}\n"
                ),
            )
        if "reset" in request.url.path:
            return httpx.Response(200, json={"ok": True})

        body = json.loads(request.content)
        # Report the token count kvlint computed, unless a test overrides it.
        served = state["tokens"] if state["tokens"] is not None else state["next_tokens"](body)
        state["queries"] += served
        state["hits"] += served * state["hit_fraction"]
        return httpx.Response(200, json={"usage": {"prompt_tokens": served}})

    def build_client(base_url: str, timeout: float = 120.0) -> httpx.Client:
        return httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))

    monkeypatch.setattr("kvlint.live.replay.build_client", build_client)
    return state


def token_counts(log: Path) -> list[int]:
    from kvlint.ingest import load as load_log
    from kvlint.tokenize.renderer import tokenize_requests

    return [len(t.token_ids) for t in tokenize_requests(MODEL, load_log(log, "openai"))]


def test_validate_reports_agreement(tmp_path: Path, fake_server: dict[str, Any]) -> None:
    log = write_log(tmp_path / "log.jsonl")
    counts = iter(token_counts(log))
    fake_server["next_tokens"] = lambda body: next(counts)

    from kvlint.ingest import load as load_log
    from kvlint.sim import vllm_blocks
    from kvlint.tokenize.renderer import tokenize_requests

    simulated = vllm_blocks.simulate(tokenize_requests(MODEL, load_log(log, "openai"))).hit_rate
    fake_server["hit_fraction"] = simulated

    result = runner.invoke(app, ["validate", str(log), "--model", MODEL, "--server", "http://fake"])

    assert result.exit_code == ExitCode.OK, result.output
    flat = re.sub(r"\s+", " ", result.output)
    assert "Token counts match the server" in flat
    assert "Absolute error" in flat


def test_validate_exits_two_when_token_counts_disagree(
    tmp_path: Path, fake_server: dict[str, Any]
) -> None:
    """A rendering mismatch makes every hit rate meaningless, so it must fail."""
    log = write_log(tmp_path / "log.jsonl")
    fake_server["tokens"] = 999  # deliberately wrong
    fake_server["next_tokens"] = lambda body: 999

    result = runner.invoke(app, ["validate", str(log), "--model", MODEL, "--server", "http://fake"])

    assert result.exit_code == ExitCode.FINDINGS
    flat = re.sub(r"\s+", " ", result.output)
    assert "Token count mismatch" in flat
    assert "renders the chat template differently" in flat


def test_validate_exits_two_when_outside_tolerance(
    tmp_path: Path, fake_server: dict[str, Any]
) -> None:
    log = write_log(tmp_path / "log.jsonl")
    counts = iter(token_counts(log))
    fake_server["next_tokens"] = lambda body: next(counts)
    fake_server["hit_fraction"] = 0.99  # far from whatever we simulate

    result = runner.invoke(app, ["validate", str(log), "--model", MODEL, "--server", "http://fake"])
    assert result.exit_code == ExitCode.FINDINGS


def test_validate_reports_an_unreachable_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = write_log(tmp_path / "log.jsonl")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "kvlint.live.replay.build_client",
        lambda base_url, timeout=120.0: httpx.Client(
            base_url="http://fake", transport=httpx.MockTransport(handler)
        ),
    )

    result = runner.invoke(app, ["validate", str(log), "--model", MODEL, "--server", "http://fake"])
    assert result.exit_code == ExitCode.ERROR
    assert "could not reach" in result.output


def test_validate_rejects_a_missing_log(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["validate", str(tmp_path / "nope.jsonl"), "--model", MODEL, "--server", "http://x"],
    )
    assert result.exit_code == ExitCode.ERROR
    assert "file not found" in result.output
