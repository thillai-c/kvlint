"""OpenAI chat-completions JSONL.

One request per line:

    {"messages": [{"role": "system", "content": "..."}], "tools": [...], "model": "..."}

Extra top-level keys (``model``, ``temperature``, provider bookkeeping) are ignored:
real logs are dumps of the request body and carry far more than we need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kvlint.errors import IngestError
from kvlint.ingest._jsonl import iter_json_lines, optional_float, optional_str
from kvlint.models import Message, Request

_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _parse_messages(path: Path, line_no: int, raw: Any) -> list[Message]:
    if not isinstance(raw, list) or not raw:
        raise IngestError(str(path), line_no, "'messages' must be a non-empty list")

    messages: list[Message] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise IngestError(str(path), line_no, f"messages[{i}] must be an object")

        role = item.get("role")
        if role not in _VALID_ROLES:
            raise IngestError(str(path), line_no, f"messages[{i}] has unsupported role {role!r}")

        content = item.get("content")
        if content is None:
            # An assistant turn that only made a tool call has null content. It
            # still occupies template space, so keep it as an empty string rather
            # than dropping it and shifting every downstream token offset.
            content = ""
        if not isinstance(content, str):
            # Multimodal content arrives as a list of parts. Out of scope for v0.1
            # (build plan §1 non-goals), and silently flattening it would produce a
            # rendered string that does not match what the engine sees.
            raise IngestError(
                str(path),
                line_no,
                f"messages[{i}] has non-string content ({type(content).__name__}); "
                "multimodal input is not supported in v0.1",
            )
        messages.append(Message(role=role, content=content))
    return messages


def _parse_tools(path: Path, line_no: int, raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise IngestError(str(path), line_no, "'tools' must be a list if present")
    for i, tool in enumerate(raw):
        if not isinstance(tool, dict):
            raise IngestError(str(path), line_no, f"tools[{i}] must be an object")
    return list(raw)


def load(path: Path) -> list[Request]:
    """Parse an OpenAI-format JSONL log into requests, preserving file order."""
    requests: list[Request] = []
    for line_no, obj in iter_json_lines(path):
        if "messages" not in obj:
            raise IngestError(str(path), line_no, "missing 'messages'")
        requests.append(
            Request(
                request_id=optional_str(path, line_no, obj, "request_id") or f"req-{line_no}",
                session_id=optional_str(path, line_no, obj, "session_id"),
                messages=_parse_messages(path, line_no, obj["messages"]),
                tools=_parse_tools(path, line_no, obj.get("tools")),
                timestamp=optional_float(path, line_no, obj, "timestamp"),
            )
        )
    return requests
