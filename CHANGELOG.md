# Changelog

Notable changes to kvlint. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/).

## [0.1.1]

Documentation and packaging. No behaviour change to the simulators or rules.

### Fixed

- **Source distributions no longer carry working notes.** The sdist defaulted to
  everything git tracked, so an internal planning document shipped inside
  0.1.0. Sdist contents are now listed explicitly.
- **`kvlint validate` warns prominently when the prefix cache was not reset.**
  vLLM only registers `/reset_prefix_cache` when started with
  `VLLM_SERVER_DEV_MODE=1`, so the automatic reset silently does nothing
  otherwise. A warm cache inflates the measured hit rate: during validation it
  produced a 4.8 pp error, larger than the 3 pp tolerance, which reads as a
  simulator fault rather than contamination. The warning now appears before the
  numbers instead of after them.
- **Correct author and repository metadata.** The PyPI project links pointed at
  a repository path that did not exist.
- **README rendering.** Absolute URLs so links and badges resolve on PyPI as
  well as GitHub, and the example output is now captured from a real run rather
  than written by hand.

### Added

- `kvlint validate --kv-budget-blocks` and `--kv-budget-tokens`, so the
  simulation can be run against the server's actual KV capacity. Without this
  the simulator always assumed an unbounded cache, which made it impossible to
  validate eviction against a server small enough to evict.

### Validated

Live validation extended from 50 requests to **10,300**, across bounded KV
caches, multi-turn conversations, tool schemas and a second model family. Exact
agreement on every workload. See the 0.1.0 notes below and
[docs/METHODOLOGY.md](https://github.com/thillai-c/kvlint/blob/main/docs/METHODOLOGY.md).

---

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

Checked against a live vLLM 0.29.0 server across **10,300 requests**, with exact
agreement on every workload:

| Workload | Requests | kvlint | vLLM | Error |
|---|---|---|---|---|
| Multi-tenant, bounded KV cache | 9,500 | matched | matched | 0.00 pp |
| Multi-turn conversations | 300 | 74.3% | 74.3% | 0.00 pp |
| Tool schemas | 200 | 95.9% | 95.9% | 0.00 pp |
| Second model family (TinyLlama) | 300 | 74.7% | 74.7% | 0.00 pp |

Token counts matched on all 10,300 requests. The eviction model was exercised
against a deliberately undersized 8,774-block cache, with the cost of eviction
rising from 1.8 to 15.3 percentage points across the four workloads.

Detail in [docs/METHODOLOGY.md](https://github.com/thillai-c/kvlint/blob/main/docs/METHODOLOGY.md).

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
