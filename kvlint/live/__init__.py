"""Live validation against a running vLLM or SGLang server.

NOTE: the `replay` function is deliberately not re-exported here. Doing so would
bind the name `kvlint.live.replay` to the function and shadow the module of the
same name, so `from kvlint.live import replay` would hand you the wrong object.
Import it from `kvlint.live.replay` directly.
"""

from __future__ import annotations

from kvlint.live.replay import ReplayResult, TokenCheck, build_client, reset_prefix_cache
from kvlint.live.validate import (
    ACCEPTABLE_ERROR_PP,
    ValidationResult,
    validate_sglang,
    validate_vllm,
)

__all__ = [
    "ACCEPTABLE_ERROR_PP",
    "ReplayResult",
    "TokenCheck",
    "ValidationResult",
    "build_client",
    "reset_prefix_cache",
    "validate_sglang",
    "validate_vllm",
]
