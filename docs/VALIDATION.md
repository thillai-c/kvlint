# Validating the simulator against a real server

Every hit rate kvlint prints elsewhere is a **model** of an engine. This procedure
is where the engine speaks for itself.

`kvlint validate` replays a log against a running server, reads the server's own
prefix-cache counters, and reports the absolute error against what kvlint
predicted. The acceptance bar is **3 percentage points**.

> **No number from this page belongs in the README until the run has actually
> happened.** The tolerance below is the target, not a result.

---

## Why WSL

vLLM does not run on Windows. It needs Linux, and in practice a CUDA GPU, though
a CPU-only build works for a 0.5B model if you are patient.

Everything except this page runs fine on Windows. Only the live replay needs WSL.

---

## 1. Set up WSL

From PowerShell **as Administrator**, once:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot if prompted, then open the Ubuntu terminal and set a username and password.

Check the GPU is visible inside WSL:

```bash
nvidia-smi
```

If that prints a table, CUDA passthrough works. You need a recent NVIDIA driver
**on Windows**; do not install a driver inside WSL, it ships its own passthrough
layer and a second driver breaks it.

If `nvidia-smi` errors, use the CPU fallback below. It is slower but still a
valid validation, because prefix caching is a scheduling and memory behaviour,
not a GPU one.

---

## 2. Clone and install kvlint

Inside WSL:

```bash
# Tools
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# The repo
git clone https://github.com/thillai-c/kvlint.git
cd kvlint

# Dependencies, including the [live] extra that validate needs
uv sync --all-extras

# Confirm it works
uv run kvlint --version
uv run pytest -q
```

Clone into the WSL filesystem (`~/kvlint`), not `/mnt/c/...`. Working across the
Windows filesystem boundary is several times slower and occasionally confuses
file watching.

---

## 3. Start a vLLM server

vLLM is **not** a kvlint dependency, so install it separately in its own
environment:

```bash
uv venv ~/vllm-env --python 3.12
source ~/vllm-env/bin/activate
uv pip install vllm
```

Then serve a small model. These flags are chosen to make the measurement clean,
not to make it fast:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --enforce-eager \
  --port 8000
```

Why each flag matters for validation:

- `--gpu-memory-utilization 0.90` buys a large KV cache. `validate` simulates
  with an **unbounded** budget, so if the server runs tight on memory and evicts,
  the real hit rate drops below the prediction and the error is eviction, not a
  simulator bug. A 0.5B model leaves plenty of room at 0.90.
- `--max-model-len 4096` is ample for the validation log and keeps more memory
  free for KV blocks.
- `--enforce-eager` skips CUDA graph capture, which cuts startup from minutes to
  seconds. It changes nothing about caching.
- Leave `--block-size` at its default of 16, which is what kvlint assumes. If you
  change it, pass the same value to `validate --block-size`.

The first run downloads roughly 1 GB of model weights.

CPU-only fallback, if `nvidia-smi` did not work:

```bash
VLLM_TARGET_DEVICE=cpu vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-prefix-caching \
  --port 8000 \
  --max-model-len 2048
```

Wait for `Application startup complete`, then confirm the counters exist:

```bash
curl -s localhost:8000/metrics | grep prefix_cache
```

You should see two lines:

```
vllm:prefix_cache_queries_total{...} 0.0
vllm:prefix_cache_hits_total{...} 0.0
```

**If those lines are missing, stop here.** Either prefix caching is off or this
vLLM version renames them. `kvlint validate` will say so rather than guess.

Record the version, it goes in the results table:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 4. Run the validation

In a second WSL terminal, back in the kvlint checkout:

```bash
cd ~/kvlint

# A log with a shared system prompt, so there is real reuse to measure
uv run python - <<'EOF'
import json, pathlib
system = ("You are a support assistant for Acme Corp. "
          "Answer concisely and cite the knowledge base article you used. "
          "Never speculate about delivery dates you cannot verify.")
lines = [
    json.dumps({"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Where is my order number {1000+i}?"},
    ]})
    for i in range(50)
]
pathlib.Path("validation-log.jsonl").write_text("\n".join(lines) + "\n")
print("wrote 50 requests")
EOF

uv run kvlint validate validation-log.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --server http://localhost:8000 \
  --engine vllm
```

---

## 5. Reading the result

```
Simulated vs real (vllm)
  kvlint simulation   61.3%
  vllm server         60.1%

Absolute error: 1.20 pp (tolerance 3 pp)

Token counts match the server for all 50 requests.
```

**Two independent checks, and the second matters more.**

**Token counts** must match exactly. A mismatch means kvlint renders the chat
template differently from the server, so every hit rate it has ever printed for
this model is built on the wrong string. `validate` exits 2 when they disagree
and says so loudly. Fix that before looking at anything else.

**Absolute error** is the headline. Within 3 pp, the simulator is doing its job.

Exit codes: `0` within tolerance and tokens agree, `2` outside tolerance or a
token mismatch, `1` could not reach the server or read the log.

---

## If the error is larger than 3 pp

Work through these in order. Each is a real cause, not a guess.

1. **Was the cache actually cold?** `validate` calls `/reset_prefix_cache` by
   default and reports when the server refuses. A warm cache credits hits the log
   did not earn. Restarting the server is the reliable reset.

2. **Was the server idle?** Other traffic during the replay lands in the same
   counters. The counter difference isolates the window, not the client, so a
   busy server pollutes the measurement.

3. **Does the block size match?** kvlint defaults to 16. If the server was
   started with a different `--block-size`, pass the same value to `validate`.

4. **Chunked prefill and preemption.** Under memory pressure vLLM may preempt and
   recompute, which kvlint does not model. Give the server enough
   `--gpu-memory-utilization` that it never preempts, or accept the gap and
   document it.

An honest documented delta with its cause beats a tuned fudge factor. If the gap
is real, it belongs in METHODOLOGY.md as a known limitation.

---

## SGLang, best effort

```bash
uv pip install "sglang[all]"
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --port 30000 --enable-metrics

uv run kvlint validate validation-log.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --server http://localhost:30000 \
  --engine sglang
```

**This comparison is weaker than the vLLM one, by construction.** SGLang exposes
`sglang:cache_hit_rate` as a **gauge**, not a counter pair. A gauge reports the
server's running average over its whole history, so it cannot be windowed to our
replay. vLLM's counters can be differenced, which isolates exactly our traffic.

So: treat a close SGLang match as encouraging, and a mismatch as inconclusive
unless the server was freshly started and otherwise idle. kvlint labels this
output as indicative rather than exact.

---

## Recording the result

Paste the real output here, with the environment that produced it. Empty until
the run happens.

| Date | vLLM version | Device | Model | Requests | Simulated | Real | Error | Tokens match |
|---|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | | |
