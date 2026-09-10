"""Generate the vendored tiny tokenizer used by the offline test suite.

Run: `uv run python scripts/build_tiny_tokenizer.py`

Why a hand-built tokenizer instead of a real one: the default test suite must run
with no network, so a HuggingFace outage can never turn CI red (plan decision D3).
This builds a byte-level model with no merges, so every byte is exactly one token.
That makes token counts trivially predictable in tests, which is what you want when
asserting things like "a shared 40-token prefix caches exactly 32 tokens".

The output is committed under tests/fixtures/tiny_tokenizer/. Regenerate only if
the chat template needs to change; the golden files depend on it.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny_tokenizer"

# ChatML, matching the shape Qwen2.5 uses, so behaviour here is representative of
# the real template the network-marked tests exercise.
# No whitespace-control dashes: `{%-` would strip the newlines that separate turns,
# and those newlines are part of the exact string the engine prefills.
CHAT_TEMPLATE = (
    "{% if tools %}"
    "<|im_start|>system\n{{ tools | tojson }}<|im_end|>\n"
    "{% endif %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


def build() -> PreTrainedTokenizerFast:
    # ByteLevel maps all 256 byte values onto printable characters, so every
    # possible input is representable and there is no unknown-token path.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: index for index, char in enumerate(alphabet)}

    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    # use_regex=False: no word-splitting, so one byte is one token, always.
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        chat_template=CHAT_TEMPLATE,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    fast.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return fast


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = build()
    tokenizer.save_pretrained(str(OUT_DIR))

    # Sanity check the properties the tests rely on.
    reloaded = PreTrainedTokenizerFast.from_pretrained(str(OUT_DIR))
    assert reloaded.is_fast, "must be a fast tokenizer for offset mapping"

    encoding = reloaded("hello", add_special_tokens=False, return_offsets_mapping=True)
    assert len(encoding["input_ids"]) == 5, encoding["input_ids"]
    assert encoding["offset_mapping"][0] == (0, 1)

    rendered = reloaded.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    assert rendered == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", repr(rendered)

    files = sorted(p.name for p in OUT_DIR.iterdir())
    print(f"wrote {OUT_DIR}")
    print("  " + "\n  ".join(files))
    spec = json.loads((OUT_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    print("  vocab size:", len(spec["model"]["vocab"]))


if __name__ == "__main__":
    main()
