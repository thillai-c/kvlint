"""Serialize fixed requests back to the format they were read from.

Round-tripping matters (build plan section M5): a fixed log that its own
ingester cannot read is not a fix, it is a new problem. So each format gets a
writer that is the inverse of its reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvlint.errors import KvlintError
from kvlint.models import Request

# Inverse of the alias map in ingest/sharegpt.py.
_ROLE_TO_SPEAKER = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
    "tool": "tool",
}


def _openai_lines(requests: list[Request]) -> list[str]:
    lines = []
    for request in requests:
        payload: dict[str, Any] = {
            "request_id": request.request_id,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.session_id is not None:
            payload["session_id"] = request.session_id
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.timestamp is not None:
            payload["timestamp"] = request.timestamp
        lines.append(json.dumps(payload, ensure_ascii=False))
    return lines


def _plain_lines(requests: list[Request]) -> list[str]:
    lines = []
    for request in requests:
        if request.raw_prompt is None:
            raise KvlintError(
                f"request {request.request_id!r} has no raw prompt, so it cannot be "
                "written back as plain format"
            )
        lines.append(
            json.dumps(
                {"request_id": request.request_id, "prompt": request.raw_prompt},
                ensure_ascii=False,
            )
        )
    return lines


def _sharegpt_lines(requests: list[Request]) -> list[str]:
    """Collapse per-turn requests back into whole conversations.

    Ingest expanded each conversation into one cumulative request per user turn,
    so the longest request in a session already holds the full history. Taking
    that one and dropping the rest reverses the expansion exactly.
    """
    longest: dict[str, Request] = {}
    order: list[str] = []
    for request in requests:
        key = request.session_id or request.request_id
        if key not in longest:
            order.append(key)
        if key not in longest or len(request.messages) > len(longest[key].messages):
            longest[key] = request

    lines = []
    for key in order:
        request = longest[key]
        lines.append(
            json.dumps(
                {
                    "id": key,
                    "conversations": [
                        {"from": _ROLE_TO_SPEAKER[m.role], "value": m.content}
                        for m in request.messages
                    ],
                },
                ensure_ascii=False,
            )
        )
    return lines


WRITERS = {
    "openai": _openai_lines,
    "plain": _plain_lines,
    "sharegpt": _sharegpt_lines,
}


def write(path: Path, requests: list[Request], fmt: str) -> int:
    """Write `requests` to `path` in `fmt`. Returns the number of lines written."""
    try:
        writer = WRITERS[fmt]
    except KeyError:
        known = ", ".join(sorted(WRITERS))
        raise KvlintError(f"cannot write format {fmt!r} (expected one of: {known})") from None

    lines = writer(requests)
    # Explicit newline so the file is byte-identical on Windows and Linux, which
    # matters because the whitespace rules treat CRLF as a defect.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)
