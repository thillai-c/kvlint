"""Deterministic JSON output.

Build plan section 2: the same input and config must produce a byte-identical
report. That rules out wall-clock timestamps, absolute paths that vary by
machine, and any set or dict whose iteration order is not pinned.
"""

from __future__ import annotations

from pathlib import Path

from kvlint.models import Report


def dumps(report: Report) -> str:
    """Serialize a report to stable JSON text."""
    # pydantic preserves field declaration order and we never serialize a set, so
    # ordering is already deterministic. `indent=2` keeps it reviewable in a diff,
    # which matters when the report is checked into CI.
    return report.model_dump_json(indent=2) + "\n"


def write(path: Path, report: Report) -> None:
    """Write a report to `path`.

    Explicit LF so the bytes match on Windows and Linux. A report that differs
    only by line endings would defeat the byte-identical guarantee the moment
    someone compared one produced on a laptop with one produced in CI.
    """
    path.write_text(dumps(report), encoding="utf-8", newline="\n")
