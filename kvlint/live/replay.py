"""Replay a log against a running server.

Sequential, `max_tokens=1`, in the original arrival order. Sequential matters: the
simulators model one request at a time, so firing them concurrently would let the
server interleave prefills and produce a hit rate our model never claimed to
predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kvlint.errors import KvlintError
from kvlint.models import Request, TokenizedRequest

DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass
class TokenCheck:
    """Our token count against the server's, for one request."""

    request_id: str
    local: int
    server: int

    @property
    def matches(self) -> bool:
        return self.local == self.server


@dataclass
class ReplayResult:
    checks: list[TokenCheck] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def mismatches(self) -> list[TokenCheck]:
        return [c for c in self.checks if not c.matches]


def build_client(base_url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Any:
    """An httpx client, with a clear error when the optional extra is missing."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise KvlintError(
            "live validation needs httpx. Install it with: pip install 'kvlint[live]'"
        ) from exc
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)


def _payload(request: Request, model: str) -> tuple[str, dict[str, Any]]:
    """Endpoint and body for one request.

    Plain prompts go to the completions endpoint so the server does not wrap them
    in a chat template. That keeps the bytes the server sees identical to the
    bytes we tokenized, which is the whole basis of the comparison.
    """
    common = {"model": model, "max_tokens": 1, "temperature": 0.0, "stream": False}

    if request.raw_prompt is not None:
        return "/v1/completions", {**common, "prompt": request.raw_prompt}

    body: dict[str, Any] = {
        **common,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    if request.tools:
        body["tools"] = request.tools
    return "/v1/chat/completions", body


def replay(
    client: Any,
    requests: list[Request],
    tokenized: list[TokenizedRequest],
    model: str,
) -> ReplayResult:
    """Send every request once, recording the server's own token counts."""
    result = ReplayResult()

    for request, tokens in zip(requests, tokenized, strict=True):
        endpoint, body = _payload(request, model)
        response = client.post(endpoint, json=body)

        if response.status_code >= 400:
            result.failures.append(
                f"{request.request_id}: HTTP {response.status_code} {response.text[:200]}"
            )
            continue

        usage = response.json().get("usage") or {}
        server_tokens = usage.get("prompt_tokens")
        if server_tokens is None:
            result.failures.append(f"{request.request_id}: response had no usage.prompt_tokens")
            continue

        result.checks.append(
            TokenCheck(
                request_id=request.request_id,
                local=len(tokens.token_ids),
                server=int(server_tokens),
            )
        )

    return result
