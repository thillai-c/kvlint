# Methodology

What kvlint measures, what it models, what it estimates, and where it is wrong.

The short version: **hit rates are simulated, TTFT and cost are estimated, and
the vLLM simulator has been checked against a real server once.** Everything
below explains which is which, so you can decide how much weight to put on any
particular number.

---

## Measured, simulated, estimated

| Output | Kind | Source |
|---|---|---|
| Token counts | **Measured** | The model's own tokenizer |
| Divergence position | **Measured** | Exact token where two prompts differ |
| Hit rate | **Simulated** | Our model of the engine's cache |
| Eviction loss | **Simulated** | Difference against an unbounded-budget pass |
| TTFT delta | **Estimated** | Your `--prefill-ms-per-1k-tokens` |
| Cost delta | **Estimated** | Your `--price-per-1m-input` and `--cached-discount` |
| Real hit rate (`validate`) | **Measured** | The server's own Prometheus counters |

TTFT and cost are the weakest of these by a wide margin. They are linear
extrapolations from a rate you supply, and they ignore batching, scheduling,
chunked prefill, and hardware entirely. kvlint labels them as estimates wherever
they appear. Treat them as order-of-magnitude, not as a forecast.

---

## How the vLLM simulator works

Semantics read from `vllm-project/vllm @ main` (2026-09-10), specifically
`vllm/v1/core/kv_cache_utils.py` and `vllm/v1/core/block_pool.py`.

1. Split token ids into blocks of 16. **The trailing partial block is dropped**,
   because vLLM never caches it.
2. Hash each block as `H(parent_hash, block_tokens)`, forming a chain. Identical
   block content at a different offset therefore produces a different hash.
3. Walk the chain against the cache and **stop at the first miss**. A later
   matching block cannot rescue a diverged prefix.
4. Insert every block, refresh LRU order, and evict least-recently-used blocks
   when over budget.

We reproduce vLLM's *equality semantics*, not its bit patterns. Our digests never
leave the process; only hit counts do.

### A consequence worth internalising

Because the partial tail is never cached, a workload of **identical** 40-token
prompts tops out at 80% on vLLM, not 100%. The 8 leftover tokens are recomputed
every time. This is correct behaviour, not a bug in the model, and it is why
`block_straddle` exists as a rule.

---

## How the SGLang simulator works

Semantics read from `sgl-project/sglang @ main` (2026-09-10),
`python/sglang/srt/mem_cache/radix_cache.py`.

A radix tree keyed by token ids. Matching is token-exact, so SGLang reports a
higher hit rate than vLLM on the same log, and the gap between them is the cost
of vLLM's block rounding. With `page_size > 1` matches are rounded down to a page
multiple; the default is 1.

Eviction is LRU over evictable leaves.

---

## Attribution: why the lint rules are token-exact

vLLM can only tell you a prompt diverged somewhere in block N. That is a 16-token
window, which is wide enough to hide the timestamp that caused the miss.

So the lint rules read from a separate token-exact source: the SGLang radix tree
run at infinite budget. This is also what makes `block_straddle` detectable, since
that rule compares the exact divergence index against the block-aligned one.

Per-engine hit rates are still reported separately and unchanged.

---

## Assumptions

Each of these is a place where we chose a defensible behaviour in the absence of
a definitive answer. They are marked `# ASSUMPTION:` in the source.

**Tokenization uses `add_special_tokens=False`.** The chat template already
writes BOS and role markers into the text as literal strings, so letting the
tokenizer add its own would double them. Verified correct against four model
families in `tests/test_model_families.py`, and against a real vLLM server, where
token counts matched on all 50 requests.

**Eviction protects the current request.** kvlint simulates one request at a
time, so "blocks belonging to the request being served" stands in for vLLM's
`ref_cnt > 0`. A request larger than the entire budget therefore overshoots it,
where real vLLM would preempt.

**SGLang eviction is plain LRU over leaves.** Real SGLang has more nuance around
page tables and lock references held by running requests.

**Requests are served sequentially, in file order.** No arrival times, no
batching, no interleaving. `Request.timestamp` is parsed but not used for
scheduling.

**The fixer moves content, never deletes it.** Deleting a timestamp would raise
the hit rate further and change what the model sees. That is not a trade kvlint
makes on your behalf.

---

## Validation

Run on 2026-09-14 against vLLM 0.29.0 on an RTX 4070 Laptop under WSL2, with
Qwen2.5-0.5B-Instruct and 50 requests:

| | Value |
|---|---|
| kvlint simulation | 84.0% |
| vLLM server | 84.0% |
| Absolute error | **0.00 pp** |
| Token counts | matched, 50 of 50 |

An exact match is the *expected* outcome for a correct simulator rather than a
lucky one. Prefix caching is deterministic: with a cold cache, sequential
requests and no memory pressure, the cached-token count is fully determined by
the block-hash chain. A gap would have meant a real defect.

Reproduce it by replaying a log against a local `vllm serve` with
`kvlint validate`, which reports the absolute error and cross-checks token
counts on every run.

### What that run does not cover

The conditions that made the match exact also make it a soft test. Still
unvalidated against a real server:

- **Eviction under memory pressure.** The server had a large KV cache and never
  evicted, so the LRU model was never exercised. This remains simulation-only.
- **Multi-turn growth**, where each request extends the previous one.
- **Tool schemas**, since no request in the log carried `tools`.
- **SGLang**, not yet run. It is also structurally weaker to validate: SGLang
  exposes `sglang:cache_hit_rate` as a **gauge** covering the server's whole
  history, while vLLM exposes **counters** that can be differenced across the
  replay window to isolate exactly our traffic.

### Does a 0.5B model generalize?

Yes, and for a reason that is easy to check rather than take on trust. Prefix
caching is a function of token ids and block boundaries; nothing in it touches
parameter count. Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct have identical
vocabularies and byte-identical chat templates, asserted in
`test_model_size_does_not_change_the_tokenizer`.

What varies between models is the **family**, because that changes the tokenizer
and the template. Rendering is checked across four families in CI.

---

## Known limitations

1. **No scheduler modelling.** No batching, preemption, chunked prefill, or
   arrival times. Under real memory pressure vLLM may preempt and recompute,
   which kvlint will not predict.
2. **`est_tokens_lost` is an upper bound and rules overlap.** Some tokens after a
   divergence were genuinely new and could never have been cached, and two rules
   can fire on the same divergence. The column does not sum, and the output says
   so.
3. **No multimodal, LoRA, or cache salting.** vLLM mixes these into the block
   hash through `extra_keys`; kvlint always passes `None`. Multimodal input is
   rejected at ingest rather than silently mis-analyzed.
4. **Commercial APIs are out of scope.** OpenAI, Anthropic and Gemini use
   different caching mechanisms (explicit breakpoints, minimum prefix lengths,
   TTL expiry rather than LRU). Pointing kvlint at those logs would produce a
   confidently wrong number, so it does not try.
5. **The fixer is heuristic.** It relocates lines matching known dynamic patterns
   and leaves everything else alone. It will miss patterns it has no rule for.
