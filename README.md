# kvlint

**A linter for your prefix cache.**

`kvlint` analyzes real LLM request logs *offline* (no GPU, no model weights) and tells you:

1. What prefix-cache hit rate your workload would get on **vLLM** (block-hash caching) and **SGLang** (RadixAttention).
2. **Why** you're missing: which prompt-template patterns break the cache (timestamps, UUIDs, unstable key ordering, whitespace drift).
3. How hit rate degrades under **memory pressure** as the KV budget shrinks.
4. The **before/after** once fixes are applied.

Built for teams self-hosting vLLM / SGLang.

> **Status: pre-alpha, under active development.** The CLI surface exists; command
> implementations are landing milestone by milestone. Not yet usable.

## License

Apache-2.0
