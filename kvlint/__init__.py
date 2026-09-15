"""kvlint: a linter for your prefix cache.

Analyzes real LLM request logs offline (no GPU, no model weights) to estimate
prefix-cache hit rate on vLLM and SGLang, explain why the cache misses, and
show the before/after of applying fixes.

See KVLINT_BUILD_PLAN.md for the full specification.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
