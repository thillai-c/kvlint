"""vLLM v1 automatic prefix caching, simulated.

Semantics read from vllm-project/vllm @ main (2026-09-10), specifically
`vllm/v1/core/kv_cache_utils.py` and `vllm/v1/core/block_pool.py`:

- `hash_block_tokens(fn, parent_block_hash, curr_block_token_ids, extra_keys)`
  returns `fn((parent_hash, tuple(token_ids), extra_keys))`, so block hashes form
  a chain and an identical block at a different offset is a different hash.
- `NONE_HASH` seeds the chain, derived from the fixed string "vllm-none-hash"
  when the hash function is cryptographic.

We reproduce the *equality semantics*, not the bit pattern. Our digests never
appear in output; only hit counts do. `extra_keys` is always None here because
multimodal, LoRA, and cache salting are v0.1 non-goals (build plan section 1).
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence

DEFAULT_BLOCK_SIZE = 16

# Mirrors vLLM's NONE_HASH seed. Fixed, never random: PYTHONHASHSEED randomization
# would make results differ between runs, breaking the determinism requirement in
# build plan section 2.
NONE_HASH = hashlib.sha256(b"vllm-none-hash").digest()

# Token ids are well inside int32 for every model we target, and array().tobytes()
# is markedly faster than joining strings for the ~1.25M hashes a large log needs.
_TOKEN_TYPECODE = "i"


def block_digest(parent: bytes, tokens: Sequence[int]) -> bytes:
    """Chain one block onto its parent.

    The parent digest is fixed-width, so there is no delimiter ambiguity between
    it and the token bytes that follow.
    """
    hasher = hashlib.sha256(parent)
    hasher.update(array(_TOKEN_TYPECODE, tokens).tobytes())
    return hasher.digest()
