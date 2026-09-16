# kvlint

**A linter for your prefix cache.**

[![PyPI](https://img.shields.io/pypi/v/kvlint)](https://pypi.org/project/kvlint/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/kvlint/)
[![CI](https://github.com/thillai-c/kvlint/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thillai-c/kvlint/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

kvlint reads your LLM request logs offline, tells you which line of which
prompt is destroying your prefix cache, and rewrites it.

```
Hit rate 8.0% to 65.3% after fixing 4 issues (vLLM), 11.0% to 68.0% (SGLang).
```

That is `kvlint demo` on a synthetic support workload. Reproduce it in one
command, no install required:

```bash
uvx kvlint demo
```

---

## What it does

kvlint reads real LLM request logs **offline**, with no GPU and no model weights,
and answers four questions:

1. What prefix-cache hit rate would this workload get on **vLLM** and **SGLang**?
2. **Why** is it missing? Which line of which prompt is breaking the cache?
3. How does it degrade as the **KV budget shrinks** and blocks get evicted?
4. What is the **before and after** if the fixes are applied?

Built for teams self-hosting vLLM or SGLang for chat, RAG, and agent workloads.

## Quickstart

```bash
pip install kvlint

kvlint analyze requests.jsonl --model Qwen/Qwen2.5-7B-Instruct
kvlint lint    requests.jsonl --model Qwen/Qwen2.5-7B-Instruct
kvlint fix     requests.jsonl --model Qwen/Qwen2.5-7B-Instruct --apply --out fixed.jsonl
```

Input can be OpenAI chat JSONL, ShareGPT conversations, or plain prompts
(`--format openai|sharegpt|plain`). ShareGPT conversations are expanded into one
cumulative request per user turn, which is how a chat client actually calls the
server.

## What a finding looks like

```
$ kvlint lint requests.jsonl --model Qwen/Qwen2.5-0.5B-Instruct

Linted 40 requests from requests.jsonl (vLLM hit rate 8.0%, SGLang 10.9%)

Findings
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Severity ┃ Rule           ┃ Affected ┃ Tokens lost ┃ Fix  ┃ Evidence                ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ high     │ dynamic_id     │    97.5% │       6,772 │ auto │ identifier in system    │
│          │                │          │             │      │ prompt changed:         │
│          │                │          │             │      │ 'req_56448162f3' vs     │
│          │                │          │             │      │ 'req_51706749f3'        │
│ high     │ dynamic_time   │    97.5% │       6,772 │ auto │ time value in system    │
│          │                │          │             │      │ prompt changed:         │
│          │                │          │             │      │ '2026-09-11T08:01:00Z'  │
│          │                │          │             │      │ vs                      │
│          │                │          │             │      │ '2026-09-11T08:00:00Z'  │
│ high     │ user_in_system │    97.5% │       6,772 │ auto │ per-user field in       │
│          │                │          │             │      │ system prompt changed   │
│ low      │ block_straddle │    97.5% │       6,772 │ auto │ shared prefix ends at   │
│          │                │          │             │      │ token 22, mid-block     │
└──────────┴────────────────┴──────────┴─────────────┴──────┴─────────────────────────┘

Tokens lost is an upper bound and rules can overlap, so the column does not sum.
```

Eight rules, each with a positive and a negative test. Full reference in
[RULES.md](https://github.com/thillai-c/kvlint/blob/main/docs/RULES.md).

## Validated against a real server

Not against itself. Against a live vLLM 0.29.0 server, 10,300 requests, exact
agreement on every run:

| Workload | Requests | kvlint | vLLM | Error |
|---|---|---|---|---|
| Multi-tenant, bounded KV cache | 9,500 | matched | matched | **0.00 pp** |
| Multi-turn conversations | 300 | 74.3% | 74.3% | **0.00 pp** |
| Tool schemas | 200 | 95.9% | 95.9% | **0.00 pp** |
| Second model family (TinyLlama) | 300 | 74.7% | 74.7% | **0.00 pp** |

**Token counts matched on all 10,300 requests**, which matters more than the hit
rate: it proves kvlint renders the chat template to the same bytes the server
prefills. `kvlint validate` checks both on every run and exits non-zero if
either fails.

The eviction model was tested against a deliberately undersized 8,774-block
cache, with workloads scaled so the cost of eviction rose from 1.8 to 15.3
percentage points. Agreement held across the whole range.

Full detail in
[METHODOLOGY.md](https://github.com/thillai-c/kvlint/blob/main/docs/METHODOLOGY.md).

## Use it in CI

```bash
kvlint lint requests.jsonl --model $MODEL --fail-on high
```

Exit codes: `0` clean, `1` error, `2` findings at or above the threshold.

## Commands

| Command | Does |
|---|---|
| `analyze` | Hit rate for both engines, with an optional KV-budget sweep |
| `lint` | Adds the findings table and the `--fail-on` gate |
| `fix` | Rewrites the log, replays it, reports before and after |
| `validate` | Replays against a live server and compares to its own counters |
| `demo` | Generates a workload with known problems and runs the lot |
| `report` | Renders a saved JSON report as HTML |

`--json` writes a deterministic report: same input and config gives byte-identical
output, so it can be diffed in CI. `--html` needs `pip install 'kvlint[report]'`.

## How it works

- **Two engine models, not one approximation.** vLLM's block-hash chain and
  SGLang's token-level radix tree are simulated separately and reported side by
  side. The gap between them is the cost of vLLM's block rounding.
- **Token-exact attribution.** Findings point at the precise token where two
  prompts stop agreeing, not a 16-token block. That precision is what lets a
  finding quote the timestamp that caused the miss.
- **Auto-fix with replay.** kvlint rewrites the log, re-tokenizes, re-simulates,
  and reports the measured before and after, so the fix is proven rather than
  suggested. It moves content, never deletes it: the model still sees every value
  it saw before.
- **Checked against a real server.** 0.00 pp error across 10,300 requests on
  vLLM 0.29.0, covering bounded caches, multi-turn, tool schemas and two model
  families.

## Documentation

- [METHODOLOGY.md](https://github.com/thillai-c/kvlint/blob/main/docs/METHODOLOGY.md): what is measured, simulated, estimated,
  and where it is wrong
- [RULES.md](https://github.com/thillai-c/kvlint/blob/main/docs/RULES.md): the eight rules, what fires them and what does not
- [CHANGELOG.md](https://github.com/thillai-c/kvlint/blob/main/CHANGELOG.md): what shipped in each release

## Development

```bash
git clone https://github.com/thillai-c/kvlint.git && cd kvlint
uv sync --all-extras
uv run pytest -q                 # offline suite
uv run pytest -m network         # real tokenizers, four model families
uv run pytest -m perf            # throughput benchmarks
```

Python 3.11+. Ruff, mypy strict, and the full suite run in CI on Linux, macOS,
and Windows.

## License

Apache-2.0
