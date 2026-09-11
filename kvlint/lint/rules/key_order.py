"""Tool schemas or JSON blobs serialized with an unstable key order.

Python dicts preserve insertion order, so a schema built from a set, a merge, or
anything iterated non-deterministically serializes differently between processes
while being semantically identical. Tool definitions sit at the very top of the
prompt, so this invalidates essentially everything.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kvlint.lint.context import RuleContext

RULE_ID = "key_order"
SEVERITY = "high"
AUTO_FIXABLE = True
SUGGESTION = "Serialize tools and any embedded JSON with json.dumps(..., sort_keys=True)."

_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _key_signature(value: Any) -> Any:
    """Keys and structure, ignoring order and ignoring leaf values."""
    if isinstance(value, dict):
        return sorted((key, _key_signature(sub)) for key, sub in value.items())
    if isinstance(value, list):
        return [_key_signature(item) for item in value]
    return None


def _tools_disagree(ctx: RuleContext) -> str | None:
    current, prior = ctx.request.tools, ctx.prior.tools if ctx.prior else None
    if not current or not prior:
        return None
    if _canonical(current) != _canonical(prior):
        return None  # genuinely different tools, not a serialization artifact
    if json.dumps(current) == json.dumps(prior):
        return None  # same order too, so this is not the cause
    return "tool schema serialized with an unstable key order"


def _windows_disagree(ctx: RuleContext) -> str | None:
    here = _OBJECT.search(ctx.current_window)
    there = _OBJECT.search(ctx.prior_window)
    if here is None or there is None or here.group(0) == there.group(0):
        return None
    try:
        parsed_here = json.loads(here.group(0))
        parsed_there = json.loads(there.group(0))
    except json.JSONDecodeError:
        return None
    if _key_signature(parsed_here) != _key_signature(parsed_there):
        return None
    if _canonical(parsed_here) != _canonical(parsed_there):
        return None
    return "embedded JSON serialized with an unstable key order"


def detect(ctx: RuleContext) -> str | None:
    return _tools_disagree(ctx) or _windows_disagree(ctx)
