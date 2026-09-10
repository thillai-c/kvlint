"""ShareGPT conversations, expanded into per-turn requests.

One conversation per line:

    {"id": "conv-1", "conversations": [{"from": "human", "value": "..."},
                                       {"from": "gpt",   "value": "..."}]}

A conversation is not a request. A chat client serving that conversation issues one
request per user turn, each carrying the entire history so far, which is exactly the
access pattern prefix caching exists to exploit. So an N-user-turn conversation
becomes N requests with a shared ``session_id``, each a strict prefix of the next.

Getting this wrong would fabricate the headline number: feeding whole conversations
as single requests would show almost no reuse, while expanding them correctly shows
the growing-prefix pattern a real deployment has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kvlint.errors import IngestError
from kvlint.ingest._jsonl import iter_json_lines, optional_str
from kvlint.models import Message, Request

# ShareGPT and its many forks disagree on these labels; accept the common spellings.
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "function_call": "tool",
    "observation": "tool",
}

_CONVERSATION_KEYS = ("conversations", "conversation", "messages", "items")


def _find_turns(path: Path, line_no: int, obj: dict[str, Any]) -> list[Any]:
    for key in _CONVERSATION_KEYS:
        turns = obj.get(key)
        if isinstance(turns, list):
            return turns
    raise IngestError(
        str(path),
        line_no,
        f"no conversation list found (looked for {', '.join(map(repr, _CONVERSATION_KEYS))})",
    )


def _parse_turn(path: Path, line_no: int, index: int, turn: Any) -> Message:
    if not isinstance(turn, dict):
        raise IngestError(str(path), line_no, f"turn {index} must be an object")

    speaker = turn.get("from", turn.get("role"))
    if not isinstance(speaker, str):
        raise IngestError(str(path), line_no, f"turn {index} has no 'from' or 'role'")

    role = _ROLE_MAP.get(speaker.strip().lower())
    if role is None:
        raise IngestError(str(path), line_no, f"turn {index} has unknown speaker {speaker!r}")

    value = turn.get("value", turn.get("content"))
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise IngestError(str(path), line_no, f"turn {index} has non-string content")

    return Message(role=role, content=value)  # type: ignore[arg-type]


def load(path: Path) -> list[Request]:
    """Expand ShareGPT conversations into cumulative per-turn requests."""
    requests: list[Request] = []

    for line_no, obj in iter_json_lines(path):
        session_id = optional_str(path, line_no, obj, "id") or f"conv-{line_no}"
        turns = _find_turns(path, line_no, obj)

        history: list[Message] = []
        user_turn = 0
        for index, raw_turn in enumerate(turns):
            message = _parse_turn(path, line_no, index, raw_turn)
            history.append(message)

            # A request is issued when the user speaks, carrying everything before it.
            if message.role != "user":
                continue
            user_turn += 1
            requests.append(
                Request(
                    request_id=f"{session_id}-t{user_turn}",
                    session_id=session_id,
                    messages=list(history),
                )
            )

    return requests
