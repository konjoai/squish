# Squish — Local LLM Inference for Apple Silicon

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/squish.svg)](https://pypi.org/project/squish/)
[![CI](https://github.com/squishai/squish/actions/workflows/ci.yml/badge.svg)](https://github.com/squishai/squish/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](https://github.com/squishai/squish)
[![Discord](https://img.shields.io/badge/Discord-join%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/squish)
[![HuggingFace](https://img.shields.io/badge/🤗%20Models-squish--community-yellow)](https://huggingface.co/squish-community)

<img src="assets/squish-logo-1.png" height="500" alt="Squish Logo"/>

> Every model you download ships in a format designed for training clusters, not laptops.
> Squish converts it once into a Metal-native format that maps directly into unified memory — **sub-second cold loads, every time.**

> ⚠️ **macOS + Apple Silicon (M1–M5) only.** Linux/CUDA support is on the roadmap.

---

## The Numbers

| | mlx_lm (cold) | Ollama | **Squish** |
|---|:---:|:---:|:---:|
| **Cold-start load time** | 28.81 s | 8–25 s | **0.33–0.53 s** |
| **RAM during load** | ~2,400 MB | ~2,000–8,000 MB | **160 MB** ‡ |
| **Disk size — 8B model** | 16.4 GB | ~4.7 GB (GGUF q4) | **4.4 GB (INT8 squished)** |
| **Throughput — qwen3:8b M3** | 12–16 tok/s | 14–19 tok/s | **14–22 tok/s** † |
| **Throughput — qwen3:4b M3** | 28–36 tok/s | 30–40 tok/s | **35–50 tok/s** † |
| **Throughput — qwen3:1.7b M3** | 55–70 tok/s | 55–75 tok/s | **65–90 tok/s** † |
| OpenAI-compatible API | ✅ | ✅ | ✅ |
| Ollama-compatible API | ❌ | ✅ | ✅ |
| Web chat UI | ❌ | ❌ | ✅ |
| Grammar-enforced tool calling | ❌ | ❌ | ✅ |
| Batch / concurrent requests | ❌ | limited | ✅ |
| macOS menu bar app | ❌ | ❌ | ✅ |
| VS Code extension | ❌ | ❌ | ✅ |
| Pre-squished weights (skip compression) | N/A | N/A | ✅ ([HuggingFace](https://huggingface.co/squish-community)) |
| Source available | ✅ | ✅ | ✅ |

> **54× faster cold load.  15× less RAM during load.  3.7× smaller model files.  Statistically identical outputs.**

‡ *160 MB = Apple Metal virtual-address delta during load (mmap, no CPU heap). Peak RSS ~402 MB.*  
† *Throughput measured with `--agent` preset (AgentKV + speculative decode). MLX-native, no GGUF conversion.*

### Model Sizes — Raw vs Squished

| Model | Raw (bf16) | Squished (INT8) | Saved |
|-------|:----------:|:---------------:|:-----:|
| qwen3:0.6b | 1.3 GB | 0.4 GB | **69%** |
| qwen3:1.7b | 3.5 GB | 1.0 GB | **71%** |
| qwen3:4b | 8.2 GB | 2.2 GB | **73%** |
| qwen3:8b | 16.4 GB | 4.4 GB | **73%** |
| qwen3:14b | 28.7 GB | 7.6 GB | **74%** |
| llama3.1:8b | 16.1 GB | 4.3 GB | **73%** |
| deepseek-r1:7b | 14.4 GB | 3.9 GB | **73%** |

---

## Install

```bash
# Homebrew (recommended)
brew install wesleyscholl/squish/squish
```

```bash
# One-liner installer
curl -fsSL https://raw.githubusercontent.com/squishai/squish/main/install.sh | bash
```

```bash
# pip
pip install squish
```

## Quick Start

```bash
squish run                  # auto-detects RAM, pulls + starts best model for your machine
```

Or pick a specific model:

```bash
squish catalog              # browse 29 available models
squish pull qwen3:8b        # download pre-squished weights from HuggingFace (~4.4 GB)
squish run qwen3:8b         # start server on :11435
```

Then open **http://localhost:11435/chat** in any browser.

Or chat in the terminal:

```bash
squish chat qwen3:8b
```

Drop-in for any OpenAI or Ollama client:

```bash
export OPENAI_BASE_URL=http://localhost:11435/v1
export OPENAI_API_KEY=squish
# or
export OLLAMA_HOST=http://localhost:11435
```

First time? Use the interactive setup wizard:

```bash
squish setup                # detects your RAM, recommends a model, pulls + starts it
```

---

## Core Features

- **Sub-second loads** — INT8 npy-dir format maps directly into Apple Metal unified memory; no dtype conversion on every boot
- **OpenAI + Ollama drop-in** — any existing client works with a single env-var change; no code changes required
- **macOS menu bar app** — SquishBar lives in your menu bar; shows live tok/s, start/stop server, one-click model switch
- **VS Code extension** — sidebar chat with streaming, model selector, server lifecycle management ([setup guide](docs/vscode-agent.md))
- **Web chat UI** — built-in at `/chat`; dark-themed, streaming, offline, multi-session history
- **Grammar-enforced tool calling** — XGrammar FSM prevents malformed JSON in tool use; works with any OpenAI `tools` client
- **Agent preset** — `--agent` (auto-enabled on Apple Silicon) wires AgentKV INT2 + speculative decode + semantic cache
- **29 ready-to-use models** — pre-squished weights on HuggingFace skip the compression step; `squish pull qwen3:8b` finishes in minutes

See [MODULES.md](MODULES.md) for the full flag reference and stability tiers (Stable / Beta / Experimental).

---

## Links

| Resource | URL |
|---|---|
| Docs | [squishai.github.io/squish](https://squishai.github.io/squish/) |
| HuggingFace models | [huggingface.co/squish-community](https://huggingface.co/squish-community) |
| Module reference | [MODULES.md](MODULES.md) |
| VS Code agent setup | [docs/vscode-agent.md](docs/vscode-agent.md) |
| Architecture paper | [docs/paper.md](docs/paper.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Discord | [discord.gg/squish](https://discord.gg/squish) |

---

## Demo

![](dev/demos/squish-demo.gif)

---

## The Numbers That Matter

Model: **Qwen2.5-1.5B-Instruct** · Hardware: **Apple Silicon M-series, MLX framework**

| | Cold `mlx_lm` load† | Reference (`mlx_lm`) | **Squish (cached)** |
|---|---:|---:|---:|
| **Load time** | 28.81s | 1.96s | **0.53s** |
| **RAM during load** | ~2400 MB | ~2400 MB | **160 MB** |
| **Peak load RAM** | ~2600 MB | ~2600 MB | **402 MB** |
| **Token cost** | $0 (local) | $0 (local) | **$0** |
| **Original .safetensors needed?** | ✅ mandatory | ✅ mandatory | **❌ not needed** |

†Cold = OS page cache cold, first process start.  
Squish cached = after one-time 19s conversion; all subsequent runs.

> **54× faster cold load.  15× less RAM.  Statistically identical outputs.**

<p align="center">
  <img src="dev/figures/fig1_load_times.png" alt="Load time comparison: cold mlx_lm vs reference vs Squish cached" width="720"/>
  <br/><em>Figure 1 — Cold-start load time comparison across three configurations</em>
</p>

<p align="center">
  <img src="dev/figures/fig2_ram_comparison.png" alt="RAM usage comparison" width="720"/>
  <br/><em>Figure 2 — Peak RAM during model load</em>
</p>

---

## The Problem

Every model you download ships in `.safetensors` — a format designed to move
weights between training clusters.  It was never designed as a local runtime format.

When `mlx_lm.load()` runs, it:
1. Allocates ~2.4 GB into **CPU heap** even though Apple Silicon has unified memory
2. **Converts every tensor** from storage dtype to runtime dtype — every single boot
3. Makes you wait **28 seconds** before the first token — for data that never changes

Squish fixes all three by decoupling storage from runtime.  The original files are
not needed after the first run.  Delete them.

---

## How It Works

```
FIRST RUN (~5-10 min — one-time per machine, done automatically by `squish pull`)
HuggingFace MLX weights ──► Squish INT8 compress ──► npy-dir on disk
                                      │
                                      └──► squish_weights.safetensors  (bf16, MLX-native)

ALL SUBSEQUENT RUNS (0.53s cold / 0.33s warm)
squish_weights.safetensors ──► mx.load() ──► Metal GPU map ──► model ready
```

No CPU heap allocation.  No dtype conversion.  Direct Metal virtual-address mapping.

### Three-Tier Cache

| Tier | File | Load time |
|---:|---|---:|
| 0 | INT8 `.npy` tensors (Vectro compressed) | ~19s |
| 1 | `finalized/*.npy` (float16, per-tensor) | ~4.5s |
| **2** | **`squish_weights.safetensors` (bf16 MLX)** | **0.33–0.53s** |

<p align="center">
  <img src="dev/figures/fig4_architecture.png" alt="Squish three-tier cache architecture" width="720"/>
  <br/><em>Figure 4 — Squish three-tier weight cache architecture</em>
</p>

---

## Benchmark Accuracy

Evaluated with **EleutherAI lm-evaluation-harness** — the framework behind the
[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

| Task | Reference | Squish | Δ | Pass |
|------|----------:|------:|---:|:---:|
| **ARC-Easy** (acc_norm) | 74.5% | 73.5% | -1.0% | ✅ |
| **HellaSwag** (acc_norm) | 63.5% | 62.0% | -1.5% | ✅ |
| **Winogrande** (acc) | 65.5% | **67.0%** | **+1.5%** | ✅ |
| **PIQA** (acc_norm) | 77.5% | 76.5% | -1.0% | ✅ |

Pass criterion: ≤2% delta (well within measurement noise at 200 samples).  
Winogrande improved by 1.5% — INT8 quantisation noise is uncorrelated with task variance.

Full reproducibility commands and multi-seed results are in [docs/RESULTS.md](docs/RESULTS.md).

<p align="center">
  <img src="dev/figures/fig3_accuracy_multi_model.png" alt="Benchmark accuracy across multiple models" width="720"/>
  <br/><em>Figure 3 — Accuracy delta vs fp16 baseline across benchmarks and models</em>
</p>

---

## v1 → v9: What Changed

Squish launched at v1.0 with a single optimization: the INT8 npy-dir format with three-tier caching.
v9.0 adds 220 modules across six phases of inference optimization.
Accuracy is unchanged — every optimization preserves the ≤2% delta criterion.

| Metric | Squish v1 | Squish v9 | Change |
|---|---:|---:|:---|
| Load time (1.5B, cached) | 0.53 s | **1.61 s** | server init overhead from 222 modules |
| Load time (7B, cached) | 2.27 s | **3.41 s** | same pattern |
| Load time (14B, cached) | 3.36 s | **5.93 s** | same pattern |
| TTFT (1.5B) | ~668 ms† | **148 ms** ✅ | streaming fixed; radix prefix reuse |
| TTFT (7B) | N/A | **533 ms** | measured live on M3 |
| TTFT (14B) | N/A | **1,008 ms** | measured live on M3 |
| TTFT (14B INT2) | N/A | 1,345 ms‡ | 4.3 GB; coherence collapse |
| Decode throughput (1.5B) | 18.9 tok/s | **7.5 tok/s**§ | measured under memory pressure |
| KV cache — compression | none | 4× (KIVI INT8) | SnapKV + KIVI KV compression |
| KV cache — prefix reuse | none | delta-only prefill | RadixTree reuse across turns |
| Grammar constrain/token | N/A | 5.5 μs | tool calling is now zero-overhead |
| MoE routing overhead | N/A | 570 μs (91% hit) | lookahead cache, watchdog-managed |
| Concurrent requests | limited | ✅ continuous batching | PagedKV + adaptive batcher |
| Max context (7B, 16 GB) | ~8K tokens | 32K+ tokens | KV eviction + compression |
| Total modules | 8 | 222 | 6 phases, 26 waves |
| ARC-Easy accuracy | 73.5% | **73.5%** ✅ | unchanged |
| HellaSwag accuracy | 62.0% | **63.0%** ✅ | +1pp |
| PIQA accuracy | 76.5% | **76.5%** ✅ | unchanged |
| WinoGrande accuracy | 67.0% | **66.0%** | −1pp (within stderr) |

† v1 streaming had a trailing-chunk artifact — all tokens arrived after ~48 s wall-clock; TTFT via `/health` was already 668 ms.
§ Measured on M3 under real system load (7 GB available RAM). Cold-dedicted-hardware throughput will be higher; spec-decode gains require a second draft model to be loaded.
‡ 2-bit post-training quantization of a pre-trained BF16 checkpoint causes coherence collapse (repetitive token loops). INT2 requires native ternary training (e.g. BitNet b1.58); INT4 is the practical PTQ lower bound.

**Six phases of optimization added between v1 and v9:**

| Phase | What it adds |
|:---:|---|
| 1 | Radix KV prefix sharing, PagedKV allocator, continuous batching, speculative decoding (ReDrafter) |
| 2 | Super-weight calibrator, asymmetric ternary quantization, Q-Filters, fast weight memory, LLM-42 determinism |
| 3 | Grammar engine (xgrammar), tool-calling acceleration, tag-dispatch, schema precompilation |
| 4 | MoE lookahead cache, Flash MLA, SSD acceptance predictor, Hydra speculative heads |
| 5 | Metal-fused kernels (RoPE, SwiGLU, INT8 KV attention), FFN `mx.compile` |
| 6 | Model pipeline, hash integrity checks, OpenAI compat suite, benchmark framework |

Run `dev/benchmarks/bench_v9_vs_v1.py` to regenerate the comparison table from saved results.
Run `dev/benchmarks/bench_eoe.py --output dev/results/eoe_v9.json` on Apple Silicon to measure live v9 numbers.

---

## Drop-In API Server

Replace every cloud API call today.  Start the server once; use it forever.

```bash
# Recommended: use the CLI
squish run 7b           # port 11435 by default

# Advanced: direct invocation
python3 -m squish.server \
    --model-dir      ~/models/Qwen2.5-7B-Instruct-bf16 \
    --compressed-dir ~/models/Qwen2.5-7B-Instruct-bf16-compressed \
    --port 11435
```

**Key server flags** (`squish run --help` for the full list):

| Flag | Values | Default | Purpose |
|---|---|---|---|
| `--kv-cache-mode` | `fp16` · `int8` · `snap` | `fp16` | KV cache compression; `int8` saves RAM on long contexts via KIVI INT8 + FP16 recent window; `snap` adds SnapKV importance-based eviction |
| `--kv-cache-window` | integer | `64` | FP16 recent-token window size for `int8`/`snap` modes |
| `--kv-cache-budget` | integer | `4096` | Max K/V positions retained in `snap` mode |
| `--log-level` | `warning` · `info` · `debug` | `warning` | Uvicorn log verbosity |

**Key compress flags** (`squish compress --help`):

| Flag | Default | Purpose |
|---|---|---|
| `--awq` | off | Run AWQ activation calibration before INT8/INT4 compression |
| `--awq-samples N` | `20` | Calibration samples for AWQ (more → better accuracy, slower) |
| `--int4` | off | INT4 nibble-packed output (~44% disk savings vs INT8). ⚠ Not recommended for models < 3B — use INT8 for best quality on small models. |
| `--zstd-level N` | `0` | Optional zstd entropy pass after quantisation (level 3 recommended) |

Point **any OpenAI client** at it — no code changes:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="squish",   # value ignored; no auth locally
)

# Streaming works
for chunk in client.chat.completions.create(
    model="Qwen2.5-1.5B-Instruct-bf16",
    messages=[{"role": "user", "content": "Explain attention mechanisms."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**Works with**: Python `openai` ≥1.0, LangChain, LlamaIndex, Continue.dev, Cursor,
any client that speaks the OpenAI wire protocol.

### Server Endpoints

| Endpoint | Status |
|---|---|
| `POST /v1/chat/completions` | ✅ streaming + non-streaming + tool calls |
| `POST /v1/completions` | ✅ legacy text completion |
| `GET  /v1/models` | ✅ model listing |
| `GET  /health` | ✅ liveness probe |
| `GET  /v1/metrics` | ✅ throughput · queue depth · memory |
| `POST /v1/embeddings` | ✅ mean-pool L2-normalised |
| `GET  /chat` | ✅ **Web chat UI** (browser) |
| `POST /api/chat` | ✅ Ollama-compatible ndjson |
| `POST /api/generate` | ✅ Ollama-compatible ndjson |
| `GET  /api/tags` | ✅ Ollama model listing |
| `GET  /api/version` | ✅ Ollama version handshake |
| `POST /api/embeddings` | ✅ Ollama-compatible embeddings |

---

## Web Chat UI

Open `http://localhost:11435/chat` in any browser after starting the server.

- Dark-themed, single-page app — no external services, works fully offline
- Streaming responses with live token rendering (marked.js + highlight.js)
- Conversation history persisted in `localStorage` (multi-session sidebar)
- Model selector auto-populated from `/v1/models`
- System prompt editor, settings panel (temp / top_p / max_tokens / seed)
- Copy buttons on all code blocks

---

## Ollama Drop-In

Squish mounts the full Ollama HTTP API at `/api/*`.  Any tool that speaks Ollama
will work against Squish with a single env-var change and **zero code changes**.

```bash
# Point any Ollama client at Squish
export OLLAMA_HOST=http://localhost:11435

# Works with the official Ollama CLI
ollama list
ollama run squish   # uses /api/generate under the hood

# Works with Continue.dev, Open WebUI, Enchanted, Msty, etc.
```

```python
# Works with the official ollama Python library
import ollama

client = ollama.Client(host="http://localhost:11435")
response = client.chat(
    model="Qwen2.5-7B-Instruct-bf16",
    messages=[{"role": "user", "content": "What is entropy coding?"}],
)
print(response["message"]["content"])
```

---

## Tool / Function Calling

`/v1/chat/completions` accepts OpenAI-format `tools` and returns `tool_calls`
in the response.  Squish injects the JSON schema into the system prompt (Qwen2.5
style) and parses the structured output automatically.

```python
import openai, json

client = openai.OpenAI(base_url="http://localhost:11435/v1", api_key="squish")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct-bf16",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

if response.choices[0].finish_reason == "tool_calls":
    call = response.choices[0].message.tool_calls[0]
    args = json.loads(call.function.arguments)
    print(f"Tool: {call.function.name}, Args: {args}")
    # → Tool: get_weather, Args: {'city': 'Tokyo', 'unit': 'celsius'}
```

---

## Integrations

Ready-made config templates live in `configs/`.  Start Squish once, then point
any of these tools at it — **no cloud API key needed for any of them**.

### Continue.dev (VS Code / JetBrains AI assistant)

```bash
# Copy config to Continue.dev's config directory
cp configs/continue.json ~/.continue/config.json
squish run 7b
# Re-open VS Code → Continue sidebar → Squish model appears automatically
```

### aider (AI pair programming in the terminal)

```bash
pip install aider-chat
squish run 7b

# Use the bundled config
aider --config configs/aider.yml

# Or install globally
cp configs/aider.yml ~/.aider.conf.yml
aider   # picks up config automatically
```

### LiteLLM (unified proxy — route multiple providers through one endpoint)

```bash
pip install litellm
squish run 7b

litellm --config configs/litellm.yaml --port 4000
# → all OpenAI clients pointing at localhost:4000 now use Squish
```

### Open WebUI / Enchanted / Msty (Ollama-compatible frontends)

Set the Ollama host to `http://localhost:11435` — all Ollama-compatible UIs work
out of the box with zero additional configuration.

---

## Advanced Features

Beyond the core stable feature set, Squish includes a large library of inference optimisations.

**Stable (validated on hardware):** INT8/INT4 compression, KV cache compression (KIVI + SnapKV),
speculative decoding, AWQ calibration, prefix/radix cache, batch scheduler, streaming, paged attention,
Flash Attention, Ollama drop-in, tool calling.

**Beta:** Advanced KV compression (ShadowKV, PQCache, YOCO, DiffKV), additional speculative decode
variants (EAGLE3, MEDUSA, KnapSpec), attention architectures (SageAttention2, GQA, ChunkedPrefill).

**Experimental:** Cutting-edge attention (FlashMLA, NativeSparseAttn), extended quantisation
(VPTQ, FP8, MXQuant, TernaryQuant), long-context optimisations (DualChunkAttn, MInference).

See [MODULES.md](MODULES.md) for the full flag reference with one-line descriptions of every supported
optimisation, categorised by stability tier.

---

## Community

- **[Discord](https://discord.gg/squish)** — get help, share benchmarks, discuss models
- **[GitHub Discussions](https://github.com/wesleyscholl/squish/discussions)** — Q&A, ideas, show & tell
- **[HuggingFace](https://huggingface.co/squish-community)** — pre-squished model weights (no local compression needed)
- **[Contributing](CONTRIBUTING.md)** — good first issues, dev setup, PR guidelines

---

## Requirements

- macOS · Apple Silicon (M1–M5)
- Python 3.10+ (3.12 recommended)
- Dependencies install automatically via `pip install squish`
- Core: `mlx-lm`, `numpy`, `transformers`, `fastapi`, `uvicorn[standard]`, `safetensors`, `zstandard`, `aiofiles`, `huggingface-hub`
- Eval extras: `pip install squish[eval]` adds `lm-eval`, `datasets`, `accelerate`
- Optional: Rust quantizer (`squish_quant_rs/`) for 4–6× faster compression throughput

---

## Weight Fidelity

| Metric | Value |
|---|---:|
| Mean cosine similarity | **0.99999** |
| Min cosine similarity | 0.99995 |
| First-token agreement | **5/5** test prompts |
| Tensors quantised (INT8) | 249 / 338 |
| Tensors passthrough (fp16) | 89 / 338 |

Embeddings, layer norms, and `lm_head` are stored as passthrough float16.  
Zero quantisation error on the prediction path.

---

## Novelty

The prior work: BitStack (ICLR 2025), Huff-LLM (Feb 2025), DFloat11, NeuZip.  
None of them work on Apple Silicon.  None serve an OpenAI-compatible API.  
None achieve sub-second loads from a compressed format.

MLX GitHub issue #3043 (January 2026) — an open feature request to add entropy
coding to MLX — is the clearest signal this gap exists and is unsolved.

Search `"compressed weight" "MLX" inference "no decompression" "Apple Silicon"` — zero results.

---

## The Summary Worth Citing

> *Squish INT8 compression achieves accuracy statistically equivalent to fp16 baseline  
> across four standard reasoning benchmarks (ARC-Easy, HellaSwag, Winogrande, PIQA),  
> while reducing cold-start load time by 54× and peak load RAM by 6×.  
> The compressed format requires zero access to the original model files  
> after a one-time per-device conversion.*

The numbers are real.  Run it yourself.
