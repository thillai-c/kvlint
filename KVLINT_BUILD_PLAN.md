# kvlint — Build Plan & Spec

> Paste this file into Claude Code as `PLAN.md` at the repo root. Work through milestones in order. Each milestone has acceptance criteria; do not start the next milestone until the current one passes.

---

## 1. What we are building

**kvlint** is a pip-installable Python CLI + library that analyzes real LLM request logs *offline* (no GPU, no model weights) and answers:

1. What prefix-cache hit rate would this workload get on **vLLM** (block-hash caching) and **SGLang** (RadixAttention)?
2. **Why** are we missing? Which specific prompt-template patterns break the cache (timestamps, UUIDs, key ordering, whitespace drift, etc.)?
3. What happens under **memory pressure** — how does hit rate degrade as the KV budget shrinks and blocks get evicted?
4. If we apply the suggested fixes, what's the **before/after** hit rate, estimated TTFT, and estimated cost?

Target user: teams self-hosting vLLM / SGLang / llama.cpp for chat, RAG, and agent workloads.

### What makes it worth building

- **Two engines, modelled separately.** vLLM's block-hash chain and SGLang's
  token-level radix tree, not one approximation of both. The gap between them is
  itself a finding: it is the cost of vLLM's block rounding.
- **Token-exact attribution.** Point at the token where prompts stop agreeing,
  not the 16-token block. Without that precision a finding cannot quote the
  value that caused the miss.
- **Eviction and memory pressure.** Show how the hit rate degrades as the KV
  budget shrinks, so a user can tell "buy more GPU" apart from "fix your
  template".
- **Auto-fix with replay.** Rewrite the log, re-simulate, and report a measured
  before and after. Suggesting a fix is easy; proving it worked is the hard part.
- **Validated against a real server**, with the procedure published so anyone can
  repeat it.

### Non-goals for v0.1

- No multimodal, LoRA, or cache-salt extra keys (document as roadmap).
- No scheduler/preemption/arrival-time modeling beyond request order + LRU eviction.
- No commercial-API usage-field parsing. OpenAI, Anthropic and Gemini cache
  with explicit breakpoints, minimum prefix lengths, and TTL expiry rather
  than an LRU block cache, so the simulators here would give a wrong answer.
- No GPU required anywhere in the core path.

---

## 2. Tech stack & conventions

- Python 3.11+, `uv` for env/deps, `hatchling` build backend.
- Deps: `typer`, `rich`, `pydantic>=2`, `transformers` (tokenizers + chat templates only), `tokenizers`, `plotly` (optional extra `[report]`), `httpx` (optional extra `[live]`).
- Dev: `pytest`, `pytest-cov`, `ruff`, `mypy` (strict on `kvlint/`), `pre-commit`.
- Type hints everywhere. Pydantic models for all data crossing module boundaries.
- Every module gets tests. Target ≥85% coverage on `sim/` and `lint/`.
- Deterministic output: same input + same config → byte-identical JSON report.
- CLI must be fast: 10k requests × 2k tokens should analyze in < 60s on a laptop CPU (excluding tokenization warmup).
- Commit style: conventional commits. One milestone ≈ one PR-sized chunk.

---

## 3. Repository layout

```
kvlint/
  __init__.py
  cli.py                  # typer app: analyze | lint | fix | validate | demo | report
  models.py               # Request, TokenizedRequest, SimResult, LintFinding, Report
  ingest/
    __init__.py
    openai_jsonl.py       # {"messages":[...], "tools":[...], "model":...} per line
    sharegpt.py           # ShareGPT conversations → per-turn requests
    plain.py              # {"prompt": "..."} per line
  tokenize/
    __init__.py
    renderer.py           # apply_chat_template exactly as the engine would; tools included
    tokenizer.py          # cached HF tokenizer loading; token_ids + offsets mapping
  sim/
    __init__.py
    base.py               # Simulator protocol: feed(request) -> PerRequestResult
    vllm_blocks.py        # 16-token block hash chain, full-block-only, LRU eviction w/ budget
    sglang_radix.py       # radix tree, token-level LCP, LRU eviction w/ budget
    metrics.py            # hit rate, cached-token fraction, divergence position
  lint/
    __init__.py
    engine.py             # runs rules over (request, divergence_pos, nearest_prior_prefix)
    rules/
      __init__.py
      dynamic_time.py     # timestamps / dates / times before the divergence
      dynamic_id.py       # UUIDs, hex ids, request/session ids, numeric counters
      whitespace_drift.py # trailing spaces, \r\n vs \n, double spaces, NFC/NFKC diffs
      key_order.py        # tools/JSON schema serialized with unstable key order
      user_in_system.py   # per-user fields (name, email, locale) inside system prompt
      short_prefix.py     # static prefix < 1 block (nothing cacheable)
      block_straddle.py   # static content ends mid-block, dynamic content begins in it
      volatile_fewshot.py # few-shot examples that change across requests
  fix/
    __init__.py
    rewriter.py           # apply fixes: move dynamic → end, canonicalize JSON, normalize ws
  report/
    __init__.py
    terminal.py           # rich tables
    json_out.py
    html.py               # plotly: hit-rate bars, divergence histogram, eviction curve
  live/
    __init__.py
    vllm_metrics.py       # replay against a running vLLM; scrape /metrics counters
    sglang_metrics.py     # same for SGLang (/metrics cache hit rate)
  demo/
    synthetic.py          # generates demo logs with injected anti-patterns
tests/
  ...
docs/
  README.md, METHODOLOGY.md, RULES.md
```

---

## 4. Core data models (`models.py`)

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Request(BaseModel):
    request_id: str
    session_id: str | None  # groups multi-turn conversations
    messages: list[Message]
    tools: list[dict] | None
    raw_prompt: str | None  # for plain format
    timestamp: float | None  # arrival order if present


class TokenizedRequest(BaseModel):
    request_id: str
    session_id: str | None
    rendered_text: str  # exact string fed to the engine
    token_ids: list[int]
    char_offsets: list[tuple[int, int]]  # per token, for mapping back to text


class PerRequestResult(BaseModel):
    request_id: str
    prompt_tokens: int
    cached_tokens: int  # tokens served from cache
    divergence_token_idx: int | None  # first token that missed vs best prior prefix
    nearest_prior_request_id: str | None
    evicted_hit_loss: int  # tokens that WOULD have hit without eviction


class SimResult(BaseModel):
    engine: Literal["vllm", "sglang"]
    config: dict
    per_request: list[PerRequestResult]
    hit_rate: float  # cached / total prompt tokens
    ideal_hit_rate: float  # same, with infinite KV budget
    block_size: int | None


class LintFinding(BaseModel):
    rule_id: str
    severity: Literal["high", "medium", "low"]
    affected_request_ids: list[str]
    affected_fraction: float
    est_tokens_lost: int
    evidence: str  # short excerpt around divergence (redacted if needed)
    suggestion: str
    auto_fixable: bool


class Report(BaseModel):
    input_summary: dict
    sims: list[SimResult]
    findings: list[LintFinding]
    before_after: dict | None
    version: str
```

---

## 5. Algorithm specs

### 5.1 vLLM block-hash simulator (`sim/vllm_blocks.py`)

Mirror vLLM v1 automatic prefix caching semantics (pin and document the vLLM version you read, e.g. read `vllm/v1/core/kv_cache_utils.py` on `main`):

- Block size default 16 (configurable).
- Split `token_ids` into full blocks; **drop the trailing partial block** (never cached).
- `hash(block_i) = H(parent_hash, tuple(block_tokens))`, with `parent_hash = NONE_HASH` for block 0. Use `hashlib.sha256` over a stable serialization for simulation purposes (we don't need bit-compat with vLLM, only identical *equality semantics*).
- Cache = dict `hash → Block(last_used, ref)` with a **block budget** `kv_budget_blocks` (None = infinite).
- For each request in arrival order:
  1. Walk blocks; count consecutive leading hits. Stop at first miss (vLLM only reuses a prefix).
  2. `cached_tokens = hits * block_size`.
  3. Insert all blocks of this request into cache (touch existing).
  4. If over budget, evict LRU blocks that are not part of the current request.
- Record `divergence_token_idx = hits * block_size` when there was at least one prior request and hits < full blocks; else None.
- Also run once with infinite budget to compute `ideal_hit_rate` and `evicted_hit_loss` per request.

### 5.2 SGLang radix simulator (`sim/sglang_radix.py`)

- Radix tree keyed by token ids; nodes store token spans and `last_access`.
- Match = longest common prefix at **token granularity** (no block rounding).
- Insert request tokens; split nodes as needed.
- Eviction: LRU over leaf nodes until under `kv_budget_tokens`. Keep it simple and documented; note that real SGLang eviction has more nuance.

### 5.3 Divergence → lint pipeline

For each request with a `divergence_token_idx`:
- Recover `nearest_prior_request_id` (the request whose prefix matched longest).
- Compute a text diff window: ±64 tokens around the divergence point in both requests, mapped back to text via `char_offsets`.
- Pass `(current_text_window, prior_text_window, absolute_char_pos, request, prior_request)` to every rule.

### 5.4 Lint rules (`lint/rules/*`) — each rule: `detect(ctx) -> LintFinding | None`

| rule_id | detects | suggestion | auto_fix |
|---|---|---|---|
| `dynamic_time` | ISO/RFC dates, HH:MM(:SS), epoch ints, weekday names differing at divergence | move to end of system prompt or into last user message | yes |
| `dynamic_id` | UUID v4, 16–64 hex, `req_`/`sess_`/`trace_` prefixed tokens, monotonically increasing ints | strip or move to end | yes |
| `whitespace_drift` | windows differ only in whitespace/newline/NFC normalization | normalize in template builder | yes |
| `key_order` | both windows parse as JSON with equal key sets but different order | `json.dumps(sort_keys=True)` for tools/schemas | yes |
| `user_in_system` | name/email/locale-like fields in system message before divergence | move personalization to first user turn | yes |
| `short_prefix` | longest shared prefix across whole log < block_size | lengthen/stabilize system prompt; note nothing is cacheable | no |
| `block_straddle` | static content ends within a block that also contains dynamic content | pad static section to block boundary (report tokens wasted) | yes (pad) |
| `volatile_fewshot` | few-shot exemplar section differs across requests in same "template family" | pin exemplars | no |

Rules must be pure functions with unit tests using synthetic windows. Aggregate findings per rule across requests; compute `affected_fraction` and `est_tokens_lost = sum(prompt_tokens - divergence_idx)` for affected requests (upper bound; say so).

### 5.5 Fixer (`fix/rewriter.py`)

Apply only `auto_fixable` findings, at the **message level** (not token level):
- Extract matched dynamic spans and append them to the end of the system message as a `\n\n[context]\n...` block, or to the start of the last user message (configurable).
- Canonicalize JSON in `tools` and any fenced JSON in messages.
- Normalize whitespace (`\r\n→\n`, strip trailing spaces, NFC).
Then re-tokenize and re-simulate. Emit `before_after = {hit_rate_before, hit_rate_after, per_engine, est_ttft_delta, est_cost_delta}`.

Cost/TTFT estimates: `--price-per-1m-input`, `--cached-discount` (default 0.9), `--prefill-ms-per-1k-tokens` (default 40, user-overridable). Label as estimates in all output.

### 5.6 Live validation (`live/`)

`kvlint validate --server http://localhost:8000 --engine vllm --model <id> logs.jsonl`
- Read `/metrics`, capture `vllm:prefix_cache_hits_total` and `vllm:prefix_cache_queries_total` (adapt names to the running version; fall back to parsing the periodic log line if metrics absent).
- Replay requests sequentially with `max_tokens=1`, same arrival order.
- Read counters again; compute real hit rate; compare to simulated; print absolute error.
- Also compare `usage.prompt_tokens` from the server to our token count per request — mismatch = chat-template rendering bug; surface it loudly.
- Same for SGLang using its `/metrics` cache-hit gauge.

---

## 6. CLI surface

```
kvlint analyze  LOGS --model MODEL [--format openai|sharegpt|plain] [--engine vllm,sglang]
                    [--block-size 16] [--kv-budget-blocks N] [--json out.json] [--html out.html]
kvlint lint     LOGS --model MODEL            # analyze + findings table
kvlint fix      LOGS --model MODEL [--apply --out fixed.jsonl]   # before/after
kvlint validate LOGS --model MODEL --server URL --engine vllm|sglang
kvlint demo     [--out demo.jsonl]            # generate synthetic log + run lint + fix
kvlint report   RESULT.json --html out.html
```

Exit codes: 0 ok, 1 error, 2 = findings with severity high (for CI use, `--fail-on high`).

---

## 7. Milestones & acceptance criteria

### M0 — Scaffold
- `uv init`, pyproject, ruff/mypy/pytest config, pre-commit, GitHub Actions (lint + test on 3.11/3.12).
- `kvlint --help` works. `models.py` complete.
- ✅ CI green on empty test suite.

### M1 — Ingest + tokenize
- Three ingesters → `list[Request]`. ShareGPT expands multi-turn into cumulative per-turn requests with shared `session_id`.
- Renderer uses `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)`; then tokenize with offsets.
- ✅ Tests: each format parses fixtures; rendered text for a Qwen2.5 template matches a golden file; token count matches `transformers` direct call.

### M2 — vLLM simulator
- Implement 5.1 incl. eviction and ideal run.
- ✅ Hand-computed unit tests: identical requests → 100% after first; shared 40-token prefix → exactly 32 cached tokens; budget of 2 blocks with 3 distinct requests → evictions observed; deterministic output.
- ✅ Perf: 10k synthetic requests × 2k tokens < 30s (hash step only).

### M3 — SGLang simulator + metrics
- Implement 5.2. Shared `Simulator` protocol.
- ✅ Tests: token-level LCP correct on off-by-one prefixes; eviction respects budget.
- ✅ `analyze` prints side-by-side vLLM vs SGLang table.

### M4 — Lint engine + 8 rules
- Implement 5.3 + all rules in 5.4 with synthetic-window tests (positive + negative case each).
- ✅ On `demo` log with an injected timestamp, `dynamic_time` fires with affected_fraction > 0.9.
- ✅ No rule fires on a clean synthetic log (false-positive guard).

### M5 — Fixer + before/after
- Implement 5.5. `fix` prints before/after table for both engines.
- ✅ Demo log: hit rate improves by ≥ 40 percentage points after fix.
- ✅ Fixed output re-parses with the same ingester (round-trip).

### M6 — Live validation
- Implement 5.6 for vLLM (SGLang best-effort).
- Run against local `vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching` on CPU or small GPU.
- ✅ Simulated vs real hit rate within 3 pp on demo log; token counts match server `usage`. Document the vLLM version tested.

### M7 — Reports + demo + docs
- HTML report: before/after bars, divergence-position histogram, hit-rate-vs-KV-budget curve.
- `kvlint demo` runs end-to-end with zero args and prints the headline number.
- README: chart first, one-line install (`uvx kvlint demo`), 3-line quickstart,
  a "how it works" and a "when it will not help" section, METHODOLOGY.md
  (what's simulated, what's estimated, limits), RULES.md.
- ✅ Fresh clone → `uv run kvlint demo` works on macOS + Linux.

### M8 — Release
- Name `kvlint` verified free on PyPI on 2026-09-09 (fallbacks: `cachelint`, `prefixaudit`, `missmap`). Reserve it early by publishing a 0.0.1 placeholder at M0. MIT license. Tag v0.1.0. Trusted-publishing workflow.
- ✅ `pip install kvlint && kvlint demo` works in a clean venv.

---

## 8. Definition of done (v0.1.0)

- All milestones' acceptance criteria met; CI green; coverage ≥ 85% on `sim/` and `lint/`.
- Demo produces: "Hit rate X% → Y% after fixing N issues (vLLM), A% → B% (SGLang); simulator matched real vLLM within Z pp."
- README shows that chart above the fold.

---

## 9. Working agreements for Claude Code

- Read this file first every session. Start by stating which milestone is in progress.
- Before writing the vLLM simulator, fetch and read `vllm/v1/core/kv_cache_utils.py` from GitHub and note the version/commit in `sim/vllm_blocks.py` docstring.
- Write tests before or alongside implementation; run `uv run pytest -q` and `uv run ruff check .` before declaring a milestone done.
- Prefer small, reviewable commits with conventional-commit messages.
- When uncertain about engine behavior, implement the documented behavior, add a `# ASSUMPTION:` comment, and list it in METHODOLOGY.md.
- Never fabricate benchmark numbers in docs; only paste numbers produced by running the code.
- Keep the CLI output terse by default; `--verbose` for detail.

---

## 10. LinkedIn launch checklist (after v0.1.0)

- One chart image: before/after hit rate, both engines.
- One sentence with the number. One sentence on what the tool does. Repo link in first comment.
- Tag vLLM and SGLang. Tagline: "a linter for your prefix cache."
- Second post a week later: "validated against a real vLLM server within Z pp — here's how" (METHODOLOGY.md).
