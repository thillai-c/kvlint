"""Core data models.

Every value crossing a module boundary is a pydantic model (KVLINT_BUILD_PLAN.md §2).
Field semantics follow §4 of the build plan exactly; comments here record the
reasoning that the spec's inline comments compress.

Determinism note (§2: "same input + same config -> byte-identical JSON report"):
these models must never carry wall-clock timestamps, absolute paths, or anything
else that varies between runs on the same input. `Request.timestamp` is the one
exception and it comes *from the log being analyzed*, not from the current clock.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
Engine = Literal["vllm", "sglang"]
Severity = Literal["high", "medium", "low"]


class _Base(BaseModel):
    """Shared config: reject unknown fields so malformed logs fail loudly at ingest."""

    model_config = ConfigDict(extra="forbid")


class Message(_Base):
    """One chat message, as it would be passed to `apply_chat_template`."""

    role: Role
    content: str


class Request(_Base):
    """A single inference request, normalized from whatever log format it came in."""

    request_id: str
    session_id: str | None = None
    """Groups multi-turn conversations. ShareGPT ingest assigns one per conversation."""

    messages: list[Message] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    raw_prompt: str | None = None
    """Set only by the `plain` ingester, which bypasses chat templating entirely."""

    timestamp: float | None = None
    """Arrival time if the log has one. Simulation uses list order, not this value."""


class TokenizedRequest(_Base):
    """A request after chat-template rendering and tokenization."""

    request_id: str
    session_id: str | None = None
    rendered_text: str
    """The exact string fed to the engine. Byte-for-byte fidelity here is what makes
    the simulators trustworthy. See the M6 `usage.prompt_tokens` cross-check."""

    token_ids: list[int]
    char_offsets: list[tuple[int, int]]
    """Per-token (start, end) into `rendered_text`, so lint findings can quote the
    source text around a divergence. Requires a fast tokenizer."""


class PerRequestResult(_Base):
    """Simulation outcome for one request under one engine."""

    request_id: str
    prompt_tokens: int
    cached_tokens: int

    divergence_token_idx: int | None = None
    """First token that missed against the best prior prefix. None when there was no
    prior request, or when the request hit completely."""

    nearest_prior_request_id: str | None = None
    """The request owning the longest matching prefix, the 'what did we drift from'
    half of every lint finding."""

    evicted_hit_loss: int = 0
    """Tokens that would have hit under an infinite KV budget but did not here.
    Computed by differencing against the ideal-budget pass."""


class SimResult(_Base):
    """Aggregate simulation result for one engine over the whole log."""

    engine: Engine
    config: dict[str, Any]
    per_request: list[PerRequestResult]
    hit_rate: float
    """cached_tokens / prompt_tokens, summed across requests."""

    ideal_hit_rate: float
    """Same ratio with an infinite KV budget. The gap between the two is the cost of
    memory pressure, as distinct from the cost of bad prompt construction."""

    block_size: int | None = None
    """16 for vLLM; None for SGLang at token granularity (page_size 1)."""


class LintFinding(_Base):
    """One rule firing, aggregated across every request it affects."""

    rule_id: str
    severity: Severity
    affected_request_ids: list[str]
    affected_fraction: float
    est_tokens_lost: int
    """Upper bound, and reported as such. Findings may overlap on the same divergence,
    so these are NOT additive across rules."""

    evidence: str
    suggestion: str
    auto_fixable: bool


class BeforeAfter(_Base):
    """Result of re-simulating a log after applying auto-fixes (§5.5)."""

    hit_rate_before: float
    hit_rate_after: float
    per_engine: dict[str, dict[str, float]]
    est_ttft_delta_ms: float | None = None
    est_cost_delta: float | None = None
    """Both deltas are estimates derived from user-supplied rates, never measurements.
    Output must label them as such."""


class Report(_Base):
    """Top-level artifact. Serializing this is the deterministic JSON output."""

    input_summary: dict[str, Any]
    sims: list[SimResult]
    findings: list[LintFinding]
    before_after: BeforeAfter | None = None
    version: str
