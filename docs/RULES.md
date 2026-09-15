# Rules

kvlint has eight rules. Six examine a single cache miss and the prompt it drifted
from; two look at the whole log, because their question cannot be answered one
miss at a time.

Every rule demands positive evidence that the **same slot holds a different
value** across two prompts. None fire on the mere presence of a suspicious
string. This is deliberate: a linter that fires on clean input gets switched off
and never switched back on, and there is a test asserting that no rule fires on
a well-built log.

| Rule | Severity | Auto-fix |
|---|---|---|
| [`dynamic_time`](#dynamic_time) | high | yes |
| [`dynamic_id`](#dynamic_id) | high | yes |
| [`user_in_system`](#user_in_system) | high | yes |
| [`whitespace_drift`](#whitespace_drift) | high | yes |
| [`key_order`](#key_order) | high | yes |
| [`short_prefix`](#short_prefix) | high | no |
| [`volatile_fewshot`](#volatile_fewshot) | medium | no |
| [`block_straddle`](#block_straddle) | low | yes |

---

## dynamic_time

**Detects** an ISO date, a clock time, an epoch integer, or a weekday or month
name that differs between two prompts at the same position.

**Why it is expensive** Usually it sits at the top of the system prompt, so every
token after it has to be recomputed. This is the single most common way to
destroy a prefix cache.

**Fix** The line is relocated below the stable instructions, so the instruction
block above it stays identical. The value itself is preserved, never deleted.

**Will not fire on** a fixed date such as a knowledge cutoff, or a date only one
of the two prompts mentions (a user asking about a date is ordinary traffic).

---

## dynamic_id

**Detects** a UUID, a hex digest, a prefixed correlation id (`req_`, `trace_`,
`session_`), or a counter of six digits or more, changing between prompts.

**Why it is expensive** Same reason as above, and these are usually added for
tracing without anyone considering the cache cost. Models rarely need them.

**Fix** Relocated below the stable text.

**Will not fire on** short numbers. Order counts, quantities and prices must not
look like correlation ids, so the threshold is six digits.

---

## user_in_system

**Detects** an email address, a labelled per-user field (`user:`, `locale:`,
`timezone:`), a locale code, or a phone number inside the **system** prompt,
differing between prompts.

**Why it is expensive** It gives every user their own cache entry. A shared
instruction block that could have been reused across your whole fleet is reused
by nobody.

**Fix** Moved out of the system prompt. `--target user_start` puts it in the last
user turn instead, which caches better still.

**Will not fire on** personalization in the user turn, which is correct practice
rather than a defect, or on a stable address such as a support contact.

---

## whitespace_drift

**Detects** two prompts that are identical once whitespace and Unicode
normalization form are removed, but differ in raw bytes: CRLF against LF,
trailing spaces, or NFC against NFD.

**Why it is expensive** It is the most frustrating class of miss, because a human
diffing the two prompts sees nothing at all while the tokenizer produces
different ids.

**Fix** CRLF to LF, trailing whitespace stripped, NFC applied.

**Will not fire on** any real content difference. If the windows differ by
anything other than whitespace, the rule stays silent.

---

## key_order

**Detects** tool schemas or embedded JSON that serialize to the same canonical
form but in a different key order.

**Why it is expensive** Tool definitions sit at the very top of the prompt, so an
unstable order invalidates essentially everything. Python dicts preserve
insertion order, so a schema built from a set, a merge, or anything iterated
non-deterministically will differ between processes while being semantically
identical.

**Fix** `json.dumps(..., sort_keys=True)` applied to `tools` and to fenced JSON
blocks in message content.

**Will not fire on** tools that genuinely differ, or JSON whose values differ.

---

## short_prefix

**Corpus-level.** Detects that the longest prefix shared by every request in the
log is shorter than one block.

**Why it is expensive** Nothing in the workload is cacheable at all. There is no
single divergence to point at; the requests simply never agree on enough leading
tokens to fill a block.

**Fix** Not automatic. Give these requests a shared, stable system prompt.
Until the common prefix exceeds one block, prefix caching cannot help.

---

## volatile_fewshot

**Corpus-level.** Detects system prompts that differ across requests where the
differing text looks like a set of exemplars, meaning at least three markers such
as `Example:`, `Q:`, `A:`, `Input:` or `Output:`.

**Why it is expensive** Exemplars are long and sit above the real input, so
rotating or reshuffling them throws away the most valuable part of the cache.

**Fix** Not automatic, because choosing which exemplars to pin is a modelling
decision rather than a mechanical one. Pin the set and its order.

**Will not fire on** system prompts that merely vary. Without the exemplar
markers it is a different problem, and a different rule's job.

---

## block_straddle

**Detects** a shared prefix that ends part-way through a block, wasting the
cacheable tokens in that block.

**Why it is cheap, and why it is still reported** vLLM caches whole blocks only.
If the stable prefix ends mid-block, that block also holds per-request content,
so its hash differs every time and the stable tokens inside it are discarded.
Severity is `low` because the loss is bounded by the block size.

Only visible because attribution is token-exact. vLLM's own divergence index is
already rounded down to a block boundary, so the waste is invisible from there.

**Fix** Pad the static section to a block boundary.

**Will not fire on** a block-aligned prefix, or an overhang below four tokens
where padding would cost more than it saves.

---

## Reading the findings table

**`affected_fraction`** is the share of requests where the rule fired.

**`est_tokens_lost`** is an **upper bound**, and totals **do not sum** across
rules. Two rules can fire on the same divergence, and some tokens after a
divergence were genuinely new content that could never have been cached. The
terminal output says this under the table.

**Rules keep firing after a fix, and that is correct.** Relocating a timestamp
does not stop it being a value that changes between requests. What the fix buys
is position: the miss now costs the tail of the prompt rather than all of it, and
`est_tokens_lost` drops accordingly. A fix that made the finding vanish would
have had to delete the timestamp.
