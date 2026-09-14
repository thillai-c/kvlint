"""Rewrite a log to remove cache-breaking patterns.

Three transformations, all at the **message level** (build plan section 5.5).
Never at token level: the engine sees a rendered string, and editing tokens would
produce something no template could have generated.

1. Normalize whitespace: CRLF to LF, strip trailing spaces, NFC.
2. Canonicalize JSON in tools and fenced blocks with sorted keys.
3. Move volatile lines out of the shared prefix and into a trailing context
   block, so the stable instructions above them stay identical.

The moves preserve content. A timestamp the model was relying on is still in the
prompt, just after the stable text rather than before it. Deleting it would raise
the hit rate further and change what the model sees, which is not a trade this
tool gets to make on the user's behalf.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from kvlint.lint.rules.dynamic_id import ID_PATTERN
from kvlint.lint.rules.dynamic_time import TIME_PATTERN
from kvlint.lint.rules.user_in_system import USER_FIELD_PATTERN
from kvlint.models import Message, Request

CONTEXT_HEADER = "[context]"

# Patterns whose values change between requests and therefore poison the prefix.
MOVABLE_PATTERNS = (TIME_PATTERN, ID_PATTERN, USER_FIELD_PATTERN)

_PLACEHOLDER = "\x00"
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_FENCED_JSON = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class FixReport:
    """What the rewriter actually changed, for honest reporting."""

    requests: list[Request]
    changed_request_ids: list[str] = field(default_factory=list)
    normalized_whitespace: int = 0
    canonicalized_json: int = 0
    moved_lines: int = 0

    @property
    def changed(self) -> int:
        return len(self.changed_request_ids)


# ---------------------------------------------------------------- normalization


def normalize_text(text: str) -> str:
    """CRLF to LF, no trailing spaces, NFC. The `whitespace_drift` fix."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("", text)
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------- JSON ordering


def canonicalize(value: Any) -> Any:
    """Recursively rebuild dicts with sorted keys.

    Python preserves insertion order, so a schema assembled from a set or a merge
    serializes differently between processes while being semantically identical.
    Sorting makes the rendered text stable. The `key_order` fix.
    """
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def _canonicalize_fenced_json(text: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        rendered = json.dumps(canonicalize(parsed), indent=2)
        if rendered != match.group(1):
            changed = True
        return f"```json\n{rendered}\n```"

    return _FENCED_JSON.sub(replace, text), changed


# ---------------------------------------------------------------- moving volatiles


def _shape(line: str) -> str:
    """The line with every dynamic value blanked out.

    Two lines share a shape when they are the same template holding different
    values, which is exactly the signature of a cache-breaking field.
    """
    for pattern in MOVABLE_PATTERNS:
        line = pattern.sub(_PLACEHOLDER, line)
    return line


def volatile_shapes(requests: list[Request]) -> set[str]:
    """Line shapes that appear with more than one value across the log.

    Only these get moved. A date that is identical in every request is not
    breaking anything, and rewriting it would be churn for no gain.
    """
    seen: dict[str, set[str]] = {}
    for request in requests:
        for message in request.messages:
            if message.role != "system":
                continue
            for line in message.content.split("\n"):
                shape = _shape(line)
                if shape == line:
                    continue  # no dynamic value in this line
                seen.setdefault(shape, set()).add(line)
    return {shape for shape, values in seen.items() if len(values) > 1}


def _split_system(content: str, volatile: set[str]) -> tuple[str, list[str]]:
    """Separate a system message into stable lines and lines to relocate."""
    lines = content.split("\n")
    kept: list[str] = []
    moved: list[str] = []
    for line in lines:
        if _shape(line) in volatile:
            moved.append(line)
        else:
            kept.append(line)

    return "\n".join(kept).rstrip(), moved


# ---------------------------------------------------------------- entry point


def fix_requests(
    requests: list[Request],
    target: str = "system_end",
) -> FixReport:
    """Apply every auto-fixable transformation, returning a new list.

    `target` decides where relocated lines land: the end of the system message,
    or the start of the last user message. The second moves them out of the
    shared prefix entirely, which caches better but puts them further from the
    instructions that reference them.
    """
    if target not in {"system_end", "user_start"}:
        raise ValueError("target must be 'system_end' or 'user_start'")

    # Normalize first: shapes are compared as text, so CRLF drift would make the
    # same line look like two different shapes and defeat the volatility check.
    staged = [
        request.model_copy(
            update={
                "messages": [
                    message.model_copy(update={"content": normalize_text(message.content)})
                    for message in request.messages
                ]
            }
        )
        for request in requests
    ]

    volatile = volatile_shapes(staged)
    report = FixReport(requests=[])

    for original, request in zip(requests, staged, strict=True):
        messages = list(request.messages)
        moved: list[str] = []
        json_changed = False

        for index, message in enumerate(messages):
            content, fenced_changed = _canonicalize_fenced_json(message.content)
            json_changed = json_changed or fenced_changed

            if message.role == "system" and volatile:
                content, relocated = _split_system(content, volatile)
                moved.extend(relocated)

            messages[index] = message.model_copy(update={"content": content})

        if moved:
            messages = _reattach(messages, moved, target)

        tools = request.tools
        if tools is not None:
            canonical = canonicalize(tools)
            if json.dumps(canonical) != json.dumps(tools):
                json_changed = True
            tools = canonical

        fixed = request.model_copy(update={"messages": messages, "tools": tools})
        report.requests.append(fixed)

        if fixed != original:
            report.changed_request_ids.append(fixed.request_id)
        if any(m.content != o.content for m, o in zip(messages, original.messages, strict=False)):
            report.normalized_whitespace += 1
        report.moved_lines += len(moved)
        report.canonicalized_json += int(json_changed)

    return report


def _reattach(messages: list[Message], moved: list[str], target: str) -> list[Message]:
    """Put relocated lines back into the prompt, after the stable text."""
    block = f"\n\n{CONTEXT_HEADER}\n" + "\n".join(moved)

    if target == "system_end":
        for index, message in enumerate(messages):
            if message.role == "system":
                messages[index] = message.model_copy(update={"content": message.content + block})
                return messages

    # Fall through to the last user turn, which is also the `user_start` target.
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            messages[index] = messages[index].model_copy(
                update={"content": block.lstrip("\n") + "\n\n" + messages[index].content}
            )
            return messages

    # No system and no user message: keep the content rather than dropping it.
    messages.append(Message(role="user", content=block.lstrip("\n")))
    return messages
