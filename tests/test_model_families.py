"""Rendering fidelity across model families (network-marked).

Model *size* does not affect prefix caching: Qwen2.5-0.5B and Qwen2.5-72B share
one tokenizer and one chat template, and block hashing only ever sees token ids.
What varies is the **family**, because that changes the template and the special
tokens.

The specific hazard is `add_special_tokens=False` in `tokenize.tokenizer.encode`.
It is correct when the chat template writes BOS and role markers into the text
itself, which is the usual arrangement. A family whose template omits BOS and
relies on the tokenizer to add it would leave us one token short, every prompt,
and the resulting hit rates would be quietly wrong.

These tests compare our render-then-encode pipeline against the tokenizer's own
`apply_chat_template(tokenize=True)`, which is the reference path. They are
marked `network` because each needs a tokenizer download, and they exist so a
`transformers` upgrade cannot silently break a family nobody revisits.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from kvlint.models import Message, Request
from kvlint.tokenize.renderer import render
from kvlint.tokenize.tokenizer import encode, load_tokenizer

pytestmark = pytest.mark.network

# One representative per template style. All are ungated, and the tokenizers are
# small; the model weights are never downloaded.
FAMILIES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]

# Templates end a prompt with a short generation prompt (an assistant header,
# or an end-of-text marker). Everything before it must carry over unchanged.
GENERATION_PROMPT_TOLERANCE = 8

CONVERSATION = [
    Message(role="system", content="You are a careful assistant."),
    Message(role="user", content="What is the capital of France?"),
]


def _template_ids(tokenizer: Any, messages: list[Message]) -> list[int]:
    """Token ids straight from the tokenizer's own template path.

    `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, which extends
    `UserDict` rather than `dict`, so an `isinstance(..., dict)` guard silently
    misses it and `len()` returns the number of keys. Subscript instead.
    """
    out = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        tokenize=True,
    )
    with contextlib.suppress(TypeError, KeyError):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


@pytest.mark.parametrize("model", FAMILIES)
def test_our_pipeline_matches_the_template_path(model: str) -> None:
    """Exact token ids, not just counts. A different id is a different block."""
    tokenizer = load_tokenizer(model)
    request = Request(request_id="r", messages=CONVERSATION)

    ours, _ = encode(tokenizer, render(tokenizer, request))
    assert ours == _template_ids(tokenizer, CONVERSATION)


@pytest.mark.parametrize("model", FAMILIES)
def test_no_special_token_is_added_twice(model: str) -> None:
    """Guards the `add_special_tokens=False` assumption directly.

    If the tokenizer would add its own BOS on top of a template that already
    contains one, this renders more tokens than the reference path.
    """
    tokenizer = load_tokenizer(model)
    request = Request(request_id="r", messages=CONVERSATION)

    ours, _ = encode(tokenizer, render(tokenizer, request))
    assert len(ours) == len(_template_ids(tokenizer, CONVERSATION))


@pytest.mark.parametrize("model", FAMILIES)
def test_offsets_reconstruct_the_rendered_text(model: str) -> None:
    """Lint evidence quotes text by offset, so this must hold per family."""
    tokenizer = load_tokenizer(model)
    rendered = render(tokenizer, Request(request_id="r", messages=CONVERSATION))

    _, offsets = encode(tokenizer, rendered)
    assert "".join(rendered[a:b] for a, b in offsets) == rendered


@pytest.mark.parametrize("model", FAMILIES)
def test_a_longer_conversation_extends_the_shorter_one(model: str) -> None:
    """The growing-prefix property the whole tool depends on.

    If a template rewrites earlier turns when a later one is appended, multi-turn
    caching would not work on that family and our ShareGPT numbers would be
    fiction.
    """
    tokenizer = load_tokenizer(model)
    longer = [
        *CONVERSATION,
        Message(role="assistant", content="Paris."),
        Message(role="user", content="And of Spain?"),
    ]

    short_ids, _ = encode(
        tokenizer, render(tokenizer, Request(request_id="a", messages=CONVERSATION))
    )
    long_ids, _ = encode(tokenizer, render(tokenizer, Request(request_id="b", messages=longer)))

    shared = 0
    for a, b in zip(short_ids, long_ids, strict=False):
        if a != b:
            break
        shared += 1

    # Everything except the trailing generation prompt must carry over. Compared
    # with a tolerance rather than exactly, because templates differ in what they
    # put at the end: Qwen closes with the assistant header, while Phi-3 appends
    # <|endoftext|> when not generating. Either way the history itself is stable,
    # and a template that rewrote earlier turns would score near zero here.
    assert shared >= len(short_ids) - GENERATION_PROMPT_TOLERANCE, (
        f"only {shared} of {len(short_ids)} tokens carried over, so appending a "
        "turn rewrote the history and multi-turn caching would not work"
    )


@pytest.mark.parametrize("model", FAMILIES)
def test_rendering_is_deterministic(model: str) -> None:
    tokenizer = load_tokenizer(model)
    request = Request(request_id="r", messages=CONVERSATION)
    assert render(tokenizer, request) == render(tokenizer, request)


def test_model_size_does_not_change_the_tokenizer() -> None:
    """Why validating on a 0.5B model generalizes to the whole family.

    Prefix caching is a function of token ids and block boundaries, so two models
    sharing a tokenizer and template cache identically regardless of parameter
    count. This is the basis for the claim in docs/VALIDATION.md.
    """
    small = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    large = load_tokenizer("Qwen/Qwen2.5-7B-Instruct")

    assert len(small) == len(large)
    assert small.chat_template == large.chat_template

    request = Request(request_id="r", messages=CONVERSATION)
    assert encode(small, render(small, request))[0] == encode(large, render(large, request))[0]
