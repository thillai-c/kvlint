"""Plain-prompt JSONL.

One prompt per line:

    {"prompt": "..."}

The text is used verbatim. No chat template is applied, because the whole point of
this format is that the caller already has the exact string the engine will see.
This is also the format `vllm analyze-prefix-cache` consumes, which makes it the
natural input for cross-checking against it later.
"""

from __future__ import annotations

from pathlib import Path

from kvlint.errors import IngestError
from kvlint.ingest._jsonl import iter_json_lines, optional_float, optional_str, require_str
from kvlint.models import Request


def load(path: Path) -> list[Request]:
    """Parse a plain-prompt JSONL log into requests, preserving file order."""
    requests: list[Request] = []
    for line_no, obj in iter_json_lines(path):
        if "prompt" not in obj:
            raise IngestError(str(path), line_no, "missing 'prompt'")
        requests.append(
            Request(
                request_id=optional_str(path, line_no, obj, "request_id") or f"req-{line_no}",
                session_id=optional_str(path, line_no, obj, "session_id"),
                raw_prompt=require_str(path, line_no, obj, "prompt"),
                timestamp=optional_float(path, line_no, obj, "timestamp"),
            )
        )
    return requests
