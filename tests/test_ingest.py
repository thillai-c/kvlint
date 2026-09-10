"""Ingest tests (M1 acceptance: each format parses its fixture)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kvlint.errors import IngestError
from kvlint.ingest import load, openai_jsonl, plain, sharegpt

# ------------------------------------------------------------------ openai


def test_openai_parses_fixture(fixtures_dir: Path) -> None:
    requests = openai_jsonl.load(fixtures_dir / "openai.jsonl")
    assert len(requests) == 3
    assert requests[0].messages[0].role == "system"
    assert requests[0].messages[1].content == "What is the capital of France?"


def test_openai_ignores_unknown_top_level_keys(fixtures_dir: Path) -> None:
    """Real logs are request-body dumps carrying `model`, `temperature`, etc."""
    requests = openai_jsonl.load(fixtures_dir / "openai.jsonl")
    assert requests[0].request_id == "req-1"


def test_openai_honours_supplied_ids_and_timestamp(fixtures_dir: Path) -> None:
    request = openai_jsonl.load(fixtures_dir / "openai.jsonl")[1]
    assert request.request_id == "custom-id"
    assert request.session_id == "s-1"
    assert request.timestamp == 1725974400.5


def test_openai_keeps_tools(fixtures_dir: Path) -> None:
    request = openai_jsonl.load(fixtures_dir / "openai.jsonl")[2]
    assert request.tools is not None
    assert request.tools[0]["function"]["name"] == "get_weather"


def test_openai_null_content_becomes_empty_string(fixtures_dir: Path) -> None:
    """A tool-call-only assistant turn still occupies template space."""
    request = openai_jsonl.load(fixtures_dir / "openai.jsonl")[2]
    assert request.messages[2].role == "assistant"
    assert request.messages[2].content == ""


def test_openai_rejects_multimodal_content(tmp_path: Path) -> None:
    log = tmp_path / "mm.jsonl"
    log.write_text(
        '{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(IngestError, match="multimodal"):
        openai_jsonl.load(log)


def test_openai_rejects_unknown_role(tmp_path: Path) -> None:
    log = tmp_path / "bad.jsonl"
    log.write_text('{"messages":[{"role":"wizard","content":"hi"}]}\n', encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported role"):
        openai_jsonl.load(log)


def test_openai_requires_messages(tmp_path: Path) -> None:
    log = tmp_path / "bad.jsonl"
    log.write_text('{"model":"x"}\n', encoding="utf-8")
    with pytest.raises(IngestError, match="missing 'messages'"):
        openai_jsonl.load(log)


# ------------------------------------------------------------------ sharegpt


def test_sharegpt_expands_cumulative_turns(fixtures_dir: Path) -> None:
    """Two user turns in one conversation must become two requests."""
    requests = sharegpt.load(fixtures_dir / "sharegpt.jsonl")
    conv_a = [r for r in requests if r.session_id == "conv-a"]
    assert len(conv_a) == 2

    # First request: system + first user turn.
    assert [m.role for m in conv_a[0].messages] == ["system", "user"]
    # Second: everything before and including the second user turn.
    assert [m.role for m in conv_a[1].messages] == ["system", "user", "assistant", "user"]


def test_sharegpt_each_request_is_a_prefix_of_the_next(fixtures_dir: Path) -> None:
    """This growing-prefix shape is the pattern prefix caching exists to exploit."""
    conv_a = [r for r in sharegpt.load(fixtures_dir / "sharegpt.jsonl") if r.session_id == "conv-a"]
    earlier, later = conv_a[0].messages, conv_a[1].messages
    assert later[: len(earlier)] == earlier


def test_sharegpt_shares_session_id_and_numbers_turns(fixtures_dir: Path) -> None:
    conv_a = [r for r in sharegpt.load(fixtures_dir / "sharegpt.jsonl") if r.session_id == "conv-a"]
    assert [r.request_id for r in conv_a] == ["conv-a-t1", "conv-a-t2"]


def test_sharegpt_synthesizes_session_id_when_absent(fixtures_dir: Path) -> None:
    requests = sharegpt.load(fixtures_dir / "sharegpt.jsonl")
    assert requests[-1].session_id == "conv-2"


def test_sharegpt_maps_speaker_aliases(tmp_path: Path) -> None:
    log = tmp_path / "aliases.jsonl"
    log.write_text(
        '{"id":"c","conversations":[{"from":"chatgpt","value":"a"},{"from":"USER","value":"b"}]}\n',
        encoding="utf-8",
    )
    request = sharegpt.load(log)[0]
    assert [m.role for m in request.messages] == ["assistant", "user"]


def test_sharegpt_trailing_assistant_turn_emits_no_request(tmp_path: Path) -> None:
    """A conversation ending in an assistant reply has no extra request to make."""
    log = tmp_path / "trailing.jsonl"
    log.write_text(
        '{"id":"c","conversations":[{"from":"human","value":"q"},{"from":"gpt","value":"a"}]}\n',
        encoding="utf-8",
    )
    assert len(sharegpt.load(log)) == 1


def test_sharegpt_rejects_unknown_speaker(tmp_path: Path) -> None:
    log = tmp_path / "bad.jsonl"
    log.write_text('{"conversations":[{"from":"alien","value":"x"}]}\n', encoding="utf-8")
    with pytest.raises(IngestError, match="unknown speaker"):
        sharegpt.load(log)


def test_sharegpt_requires_a_conversation_list(tmp_path: Path) -> None:
    log = tmp_path / "bad.jsonl"
    log.write_text('{"id":"c"}\n', encoding="utf-8")
    with pytest.raises(IngestError, match="no conversation list"):
        sharegpt.load(log)


# ------------------------------------------------------------------ plain


def test_plain_parses_fixture(fixtures_dir: Path) -> None:
    requests = plain.load(fixtures_dir / "plain.jsonl")
    assert len(requests) == 2
    assert requests[0].raw_prompt == "The quick brown fox"
    assert requests[0].messages == []
    assert requests[1].request_id == "p2"


def test_plain_requires_prompt(tmp_path: Path) -> None:
    log = tmp_path / "bad.jsonl"
    log.write_text('{"text":"oops"}\n', encoding="utf-8")
    with pytest.raises(IngestError, match="missing 'prompt'"):
        plain.load(log)


# ------------------------------------------------------------------ shared


def test_dispatch_by_format_name(fixtures_dir: Path) -> None:
    assert len(load(fixtures_dir / "plain.jsonl", "plain")) == 2
    assert len(load(fixtures_dir / "openai.jsonl", "openai")) == 3


def test_dispatch_rejects_unknown_format(fixtures_dir: Path) -> None:
    with pytest.raises(IngestError, match="unknown format"):
        load(fixtures_dir / "plain.jsonl", "vllm")


def test_error_names_the_offending_line(fixtures_dir: Path) -> None:
    """A user with a 100k-line log needs to know which line broke."""
    with pytest.raises(IngestError) as exc:
        openai_jsonl.load(fixtures_dir / "malformed.jsonl")
    assert exc.value.line_no == 2
    assert "malformed.jsonl:2" in str(exc.value)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    log = tmp_path / "gaps.jsonl"
    log.write_text('{"prompt":"a"}\n\n\n{"prompt":"b"}\n', encoding="utf-8")
    assert len(plain.load(log)) == 2


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="file not found"):
        plain.load(tmp_path / "nope.jsonl")


def test_request_ids_are_unique_within_a_log(fixtures_dir: Path) -> None:
    for name, fmt in [
        ("openai.jsonl", "openai"),
        ("sharegpt.jsonl", "sharegpt"),
        ("plain.jsonl", "plain"),
    ]:
        requests = load(fixtures_dir / name, fmt)
        ids = [r.request_id for r in requests]
        assert len(ids) == len(set(ids)), f"duplicate request_id in {name}"
