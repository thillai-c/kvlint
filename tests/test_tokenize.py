"""Render and tokenize tests (M1 acceptance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kvlint.errors import TokenizerError
from kvlint.models import Message, Request
from kvlint.tokenize.renderer import render, tokenize_request
from kvlint.tokenize.tokenizer import encode, load_tokenizer
from tests.conftest import QWEN_MODEL, TINY_TOKENIZER_DIR

# ------------------------------------------------------------------ tokenizer


def test_tokenizer_is_fast(tiny_tokenizer: Any) -> None:
    """Offsets are impossible without a fast tokenizer, so this is load-bearing."""
    assert tiny_tokenizer.is_fast


def test_tokenizer_is_cached() -> None:
    """Loading is the slowest step; `analyze` must not repeat it per request."""
    assert load_tokenizer(str(TINY_TOKENIZER_DIR)) is load_tokenizer(str(TINY_TOKENIZER_DIR))


def test_unknown_model_raises_a_clear_error() -> None:
    with pytest.raises(TokenizerError, match="could not load tokenizer"):
        load_tokenizer("./definitely-not-a-real-tokenizer-path")


def test_encode_returns_one_offset_per_token(tiny_tokenizer: Any) -> None:
    ids, offsets = encode(tiny_tokenizer, "hello world")
    assert len(ids) == len(offsets)


def test_offsets_map_back_to_the_source_text(tiny_tokenizer: Any) -> None:
    """Lint findings quote text around a divergence; this is how they do it."""
    text = "The quick brown fox"
    _, offsets = encode(tiny_tokenizer, text)
    assert "".join(text[a:b] for a, b in offsets) == text


def test_offsets_are_non_decreasing(tiny_tokenizer: Any) -> None:
    _, offsets = encode(tiny_tokenizer, "abc def")
    assert all(offsets[i][1] <= offsets[i + 1][0] for i in range(len(offsets) - 1))


def test_tiny_tokenizer_is_one_token_per_byte(tiny_tokenizer: Any) -> None:
    """The property M2/M3 rely on to make cache assertions checkable by hand."""
    ids, _ = encode(tiny_tokenizer, "abcdefghij")
    assert len(ids) == 10


# ------------------------------------------------------------------ renderer


def test_plain_prompt_bypasses_the_chat_template(tiny_tokenizer: Any) -> None:
    request = Request(request_id="r", raw_prompt="raw text, verbatim")
    assert render(tiny_tokenizer, request) == "raw text, verbatim"


def test_render_applies_the_chat_template(tiny_tokenizer: Any) -> None:
    request = Request(request_id="r", messages=[Message(role="user", content="hi")])
    assert render(tiny_tokenizer, request) == (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )


def test_render_adds_the_generation_prompt(tiny_tokenizer: Any) -> None:
    """Engines prefill through the assistant header, so we must too."""
    request = Request(request_id="r", messages=[Message(role="user", content="hi")])
    assert render(tiny_tokenizer, request).endswith("<|im_start|>assistant\n")


def test_render_includes_tools(tiny_tokenizer: Any) -> None:
    request = Request(
        request_id="r",
        messages=[Message(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert "get_weather" in render(tiny_tokenizer, request)


def test_render_is_deterministic(tiny_tokenizer: Any) -> None:
    request = Request(request_id="r", messages=[Message(role="user", content="hi")])
    assert render(tiny_tokenizer, request) == render(tiny_tokenizer, request)


def test_empty_request_raises(tiny_tokenizer: Any) -> None:
    with pytest.raises(TokenizerError, match="neither messages nor a prompt"):
        render(tiny_tokenizer, Request(request_id="r"))


def test_a_longer_conversation_renders_as_a_prefix_of_itself(tiny_tokenizer: Any) -> None:
    """The property the whole tool depends on: turn N+1 extends turn N.

    True only because the generation prompt is the sole trailing content, so
    appending turns leaves everything before it byte-identical.
    """
    first = Request(request_id="a", messages=[Message(role="user", content="one")])
    second = Request(
        request_id="b",
        messages=[
            Message(role="user", content="one"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="two"),
        ],
    )
    rendered_first = render(tiny_tokenizer, first)
    rendered_second = render(tiny_tokenizer, second)
    shared = rendered_first[: -len("<|im_start|>assistant\n")]
    assert rendered_second.startswith(shared)


# ------------------------------------------------------------------ tokenize_request


def test_tokenize_request_populates_every_field(tiny_tokenizer: Any) -> None:
    request = Request(
        request_id="r1", session_id="s1", messages=[Message(role="user", content="hi")]
    )
    tokenized = tokenize_request(tiny_tokenizer, request)
    assert tokenized.request_id == "r1"
    assert tokenized.session_id == "s1"
    assert tokenized.token_ids
    assert len(tokenized.char_offsets) == len(tokenized.token_ids)
    assert tokenized.rendered_text.startswith("<|im_start|>")


def test_special_tokens_are_single_tokens(tiny_tokenizer: Any) -> None:
    """ChatML markers must not shatter into bytes, or block boundaries drift."""
    request = Request(request_id="r", messages=[Message(role="user", content="hi")])
    tokenized = tokenize_request(tiny_tokenizer, request)
    rendered = tokenized.rendered_text
    # One token for each marker, rather than len("<|im_start|>") tokens.
    assert len(tokenized.token_ids) < len(rendered)


# ------------------------------------------------------------------ real tokenizer


@pytest.mark.network
def test_qwen_render_matches_golden(fixtures_dir: Path) -> None:
    """Guards against a transformers upgrade silently changing the template."""
    tokenizer = load_tokenizer(QWEN_MODEL)
    request = Request(
        request_id="r",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the capital of France?"),
        ],
    )
    golden = (fixtures_dir.parent / "golden" / "qwen25_chat.txt").read_text(encoding="utf-8")
    assert render(tokenizer, request) == golden


@pytest.mark.network
def test_qwen_token_count_matches_transformers_directly() -> None:
    """Our encode() must agree with a plain transformers call, or M6 will fail."""
    tokenizer = load_tokenizer(QWEN_MODEL)
    request = Request(
        request_id="r", messages=[Message(role="user", content="What is the capital of France?")]
    )
    rendered = render(tokenizer, request)
    ours, _ = encode(tokenizer, rendered)
    theirs = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assert ours == theirs


@pytest.mark.network
def test_qwen_offsets_reconstruct_the_text() -> None:
    tokenizer = load_tokenizer(QWEN_MODEL)
    text = "The quick brown fox jumps over the lazy dog."
    _, offsets = encode(tokenizer, text)
    assert "".join(text[a:b] for a, b in offsets) == text
