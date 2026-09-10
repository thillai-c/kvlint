"""Chat-template rendering.

The rendered string must be byte-identical to what the serving engine builds from
the same request, because the simulators hash tokens derived from it. A stray
space here becomes a phantom cache miss in the report.
"""

from __future__ import annotations

from typing import Any

from kvlint.errors import TokenizerError
from kvlint.models import Request, TokenizedRequest
from kvlint.tokenize.tokenizer import encode, load_tokenizer


def render(tokenizer: Any, request: Request) -> str:
    """Render a request to the exact string the engine would prefill.

    Plain-format requests bypass templating entirely: their text is already the
    final prompt.
    """
    if request.raw_prompt is not None:
        return request.raw_prompt

    if not request.messages:
        raise TokenizerError(f"request {request.request_id!r} has neither messages nor a prompt")

    try:
        rendered = tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            tools=request.tools,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as exc:
        raise TokenizerError(
            f"chat template failed for request {request.request_id!r}: {exc}"
        ) from exc

    if not isinstance(rendered, str):  # pragma: no cover
        raise TokenizerError("apply_chat_template(tokenize=False) did not return a string")
    return rendered


def tokenize_request(tokenizer: Any, request: Request) -> TokenizedRequest:
    """Render and tokenize one request."""
    rendered = render(tokenizer, request)
    token_ids, offsets = encode(tokenizer, rendered)
    return TokenizedRequest(
        request_id=request.request_id,
        session_id=request.session_id,
        rendered_text=rendered,
        token_ids=token_ids,
        char_offsets=offsets,
    )


def tokenize_requests(model: str, requests: list[Request]) -> list[TokenizedRequest]:
    """Render and tokenize a whole log, preserving arrival order."""
    tokenizer = load_tokenizer(model)
    return [tokenize_request(tokenizer, request) for request in requests]
