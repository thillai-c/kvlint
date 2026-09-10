"""Log ingesters: normalize each supported log format into `list[Request]`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kvlint.errors import IngestError
from kvlint.ingest import openai_jsonl, plain, sharegpt
from kvlint.models import Request

LOADERS: dict[str, Callable[[Path], list[Request]]] = {
    "openai": openai_jsonl.load,
    "sharegpt": sharegpt.load,
    "plain": plain.load,
}


def load(path: Path, fmt: str) -> list[Request]:
    """Load `path` using the named format."""
    try:
        loader = LOADERS[fmt]
    except KeyError:
        known = ", ".join(sorted(LOADERS))
        raise IngestError(
            str(path), 0, f"unknown format {fmt!r} (expected one of: {known})"
        ) from None
    return loader(path)


__all__ = ["LOADERS", "IngestError", "load", "openai_jsonl", "plain", "sharegpt"]
