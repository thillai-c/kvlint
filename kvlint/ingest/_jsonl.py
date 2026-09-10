"""Shared JSONL scanning.

All three formats are line-delimited JSON, so the read/parse/report-the-line-number
loop lives here once.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kvlint.errors import IngestError


def iter_json_lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_no, obj)`` for each non-blank line, 1-indexed.

    Blank lines are skipped rather than treated as errors: real logs get
    concatenated, truncated, and hand-edited, and a trailing newline is not a
    reason to refuse a 100k-line file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise IngestError(str(path), 0, "file not found") from exc
    except UnicodeDecodeError as exc:
        raise IngestError(str(path), 0, f"not valid UTF-8: {exc}") from exc

    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IngestError(str(path), line_no, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise IngestError(
                str(path), line_no, f"expected a JSON object, got {type(obj).__name__}"
            )
        yield line_no, obj


def require_str(path: Path, line_no: int, obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise IngestError(str(path), line_no, f"{key!r} must be a string")
    return value


def optional_str(path: Path, line_no: int, obj: dict[str, Any], key: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str | int):
        raise IngestError(str(path), line_no, f"{key!r} must be a string if present")
    return str(value)


def optional_float(path: Path, line_no: int, obj: dict[str, Any], key: str) -> float | None:
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IngestError(str(path), line_no, f"{key!r} must be a number if present")
    return float(value)
