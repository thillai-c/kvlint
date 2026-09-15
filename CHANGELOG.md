# Changelog

Notable changes to kvlint. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/).

## [0.1.0]

First public release.

### Added

**Two engine simulators.** vLLM's block-hash chain (16-token blocks, chained
hashes, partial tail never cached, LRU eviction under a block budget) and
SGLang's token-level radix tree with page alignment. Both run offline, with no
GPU and no model weights.

**Eight lint rules** with cause attribution, each with a positive and a negative
test: `dynamic_time`, `dynamic_id`, `user_in_system`, `whitespace_drift`,
`key_order`, `short_prefix`, `volatile_fewshot`, `block_straddle`. Rules read a
token-exact divergence point rather than a block-rounded one, which is what lets
a finding quote the value that caused the miss.

**Auto-fix with before/after replay.** Rewrites the log at message level,
re-tokenizes, re-simulates, and reports the measured change. Content is moved,
never deleted, so the model still sees every value it saw before.

**Three input formats.** OpenAI chat JSONL, ShareGPT (expanded into one
cumulative request per user turn), and plain prompts.

**Live validation.** `kvlint validate` replays a log against a running vLLM or
SGLang server and compares the simulated hit rate against the server's own
counters, plus a per-request token-count cross-check.

**Reports.** Terminal tables, deterministic JSON (byte-identical for the same
input and config, so it can be diffed in CI), and a standalone HTML report with
before/after bars, a divergence histogram, and a hit-rate-against-KV-budget
curve.

**CI gate.** `kvlint lint --fail-on high` exits 2 when findings at or above a
severity threshold fire.

### Validated

The vLLM simulator was checked against a live vLLM 0.29.0 server on
Qwen2.5-0.5B-Instruct with 50 requests:

| | Value |
|---|---|
| kvlint simulation | 84.0% |
| vLLM server | 84.0% |
| Absolute error | 0.00 pp |
| Token counts | matched, 50 of 50 |

Eviction under memory
pressure, multi-turn logs, tool schemas, and SGLang are **not** yet validated
against a real server, and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) records
that.

### Known limitations

- No scheduler modelling: no batching, preemption, chunked prefill, or arrival
  times.
- `est_tokens_lost` is an upper bound and rules can overlap, so the column does
  not sum.
- No multimodal, LoRA, or cache salting. Multimodal input is rejected at ingest
  rather than silently mis-analyzed.
- Commercial APIs are out of scope: they cache with explicit breakpoints,
  minimum prefix lengths, and TTL expiry rather than an LRU block cache.
- TTFT and cost figures are estimates derived from user-supplied rates, not
  measurements, and are labelled as such wherever they appear.
