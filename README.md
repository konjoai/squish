# Squish

<img src="assets/squish-logo-1.png" height="320" alt="Squish Logo"/>

## Squeeze the Most Out of Your Models

**Local LLM inference for Apple Silicon. Faster end-to-end response on long contexts, less RAM, INT3 support.**

[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/squish-ai.svg)](https://pypi.org/project/squish-ai/)
[![CI](https://github.com/konjoai/squish/actions/workflows/ci.yml/badge.svg)](https://github.com/konjoai/squish/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](https://github.com/konjoai/squish)
[![HuggingFace](https://img.shields.io/badge/🤗%20Models-squish--community-yellow)](https://huggingface.co/squish-community)

---

## The Numbers (v9.32.0 / bench v5.1.1)

Measured 2026-06-02 on Apple M3 MacBook Pro, 16 GB unified memory.
Model: Qwen2.5-7B-Instruct. Quant: INT4 (squish) / Q4_K_M (Ollama).
Five-run medians. Raw artifacts in [`results/benchmarks_v5_1_1/`](results/benchmarks_v5_1_1/).

| Metric | Ollama 0.18.2 | **Squish (recommended)** |
|---|---:|---:|
| **E2E response @ 4000-token prompt** | 69.63 s | **12.78 s** &nbsp;_(5.4× faster)_ |
| **E2E response @ 75-token prompt** | 8.09 s | **5.50 s** &nbsp;_(1.5× faster)_ |
| **Peak RAM during inference** | ~5 GB | **3.36 GB** |
| **Disk size — INT4** | 4.36 GB | **4.00 GB** |
| **Disk size — INT3 (Qwen3)** | not supported | **3.56 GB** |
| **TTFT @ 75-token prompt** | **131 ms** | 279 ms &nbsp;_(honest loss)_ |

**Squish wins end-to-end response time at every prompt size measured**, with
the largest win on long contexts (5.4× at 4000 tokens), uses ~33% less RAM,
and supports INT3 for compatible model families.

**Ollama wins time-to-first-token at every prompt size**, and inter-token
jitter on long contexts. If first-byte latency matters more than full-response
latency, Ollama is the right tool.

Full table, methodology, and ablation: [`docs/RESULTS.md`](docs/RESULTS.md)
(v5.1.1 section).

---

## Why Squish

Squish is for the workload most local-LLM tools aren't tuned for: **the same
model called many times an hour from the terminal with shifting context** —
git-commit-message generation, code-review prompts, agent loops, multi-turn
chat, document Q&A.

On a 16 GB Mac, that workload collides with the rest of your work. Ollama
keeps ~5 GB resident and pays a long prefill cost on each new long prompt.
Squish is a persistent daemon: the model loads once when the daemon starts,
and a two-cache architecture (block-paged KV cache for shifting prefixes,
prompt KV cache for exact repeats) avoids re-prefilling work the daemon has
already done.

Designed for one developer on one machine. Not a production multi-tenant API.

---

## Install

> Prerequisite (macOS/Homebrew): Xcode Command Line Tools are required.
> Install them with `xcode-select --install`.
> If Homebrew reports "Command Line Tools are too outdated", update from
> System Settings -> General -> Software Update, or reinstall CLT.

```bash
# Homebrew (recommended on macOS)
brew tap konjoai/squish
brew trust konjoai/squish
brew install squish

# PyPI
pip install squish-ai

# From source
git clone https://github.com/konjoai/squish
cd squish
pip install -e .
```

> Note: The PyPI package is `squish-ai`. After installing, the Python module
> and CLI are both named `squish`:
>
> ```bash
> pip install squish-ai
> squish run --version
> python -c "import squish; print(squish.__version__)"
> ```

## Optional Performance Enhancements

**4x faster quantization** - install the Rust extension:

```bash
cd squish_quant_rs && python3 -m maturin build --release && pip install .
```

Requirements: macOS 13+, Apple Silicon (M1–M5), Python 3.10+.

---

## Quick Start

```bash
# Pull a pre-quantised model from the catalog
squish pull qwen2.5-7b-int4

# Start the daemon with both caches enabled (recommended config)
squish run qwen2.5-7b-int4 \
  --block-kv-cache ~/.cache/squish/blocks \
  --prompt-kv-cache ~/.cache/squish/pkv \
  --port 8080
```

Use it as an OpenAI-compatible client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-int4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Or point any OpenAI / Ollama client at it:

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=squish
# Ollama-compatible /api/* endpoints also work
export OLLAMA_HOST=http://localhost:8080
```

Install the macOS LaunchAgent so the daemon starts at login:

```bash
squish daemon install
```

The **SquishBar** menu-bar app (`apps/macos/SquishBar/`) ships alongside the
daemon — model picker, load progress, and a global hotkey for the chat panel.
Build it from Xcode or grab the signed `.app` from the GitHub release page.

---

## Configuration

| Flag | Purpose |
|---|---|
| `--block-kv-cache <DIR>` | Block-paged KV cache for shifting-prefix workloads (agents, multi-turn). Persists across daemon restarts via `.safetensors` blocks. |
| `--prompt-kv-cache <DIR>` | Exact-prompt KV cache. Single-digit-millisecond TTFT on verbatim repeats. |
| `--block-kv-size N` | Block size in tokens (default 64). |
| `--draft-model <MODEL>` | Speculative-decode draft model (opt-in; see [v5.2 diagnosis](results/benchmarks_v5_2/SPEC_DECODE_DIAGNOSIS.md) for current status — net-negative on M3 INT4 with the draft models tested, kept off by default). |
| `--draft-depth N` | Speculative decode depth K. |
| `--no-spec`, `--no-cache` | Disable flags, intended for benchmark controls. |
| `squish daemon install` / `uninstall` | macOS LaunchAgent integration. |

Picking the right cache for your workload:

- **Exact-prompt repeats** (cached scripts, fixed templates, automated jobs):
  `--prompt-kv-cache` alone. ~9 ms TTFT on a cache hit.
- **Shifting-prefix workloads** (agents, multi-turn conversations):
  `--block-kv-cache` alone, or combined config.
- **General use without knowing the workload**: combined config (both caches
  enabled). Best end-to-end completion time across prompt sizes.

The combined config currently doesn't inherit PKV's fast-hit TTFT due to a
lookup ordering issue documented in
[`results/benchmarks_v5_1_1/DIAGNOSIS.md`](results/benchmarks_v5_1_1/DIAGNOSIS.md);
reordering is tracked as a v5.2 follow-up.

---

## Benchmarks

Full table, methodology, ablation, jitter analysis, and raw per-run JSON:

- [`docs/RESULTS.md`](docs/RESULTS.md) — v5.1.1 section is the source of truth
- [`benchmarks/ollama_vs_squish/RESULTS.md`](benchmarks/ollama_vs_squish/RESULTS.md) — bench harness output
- [`results/benchmarks_v5_1_1/DIAGNOSIS.md`](results/benchmarks_v5_1_1/DIAGNOSIS.md) — combined-cache ordering write-up
- [`results/benchmarks_v5_1_1/JITTER_ANALYSIS.md`](results/benchmarks_v5_1_1/JITTER_ANALYSIS.md) — inter-token p95 explanation
- [`results/benchmarks_v5_2/SPEC_DECODE_DIAGNOSIS.md`](results/benchmarks_v5_2/SPEC_DECODE_DIAGNOSIS.md) — why speculative decoding is currently opt-in

Reproduce locally:

```bash
python benchmarks/ollama_vs_squish/bench_v5_1.py
```

---

## What Squish Doesn't Do

In the spirit of honesty:

- **No GPU support outside Apple Silicon.** It's MLX-based. CUDA users should use vLLM or llama.cpp.
- **No multi-user serving.** Designed for one developer, one machine — not a production API.
- **No multimodal models.** Text only.
- **Higher inter-token p95 on long prompts** than Ollama. Conscious tradeoff (deferred KV-cache restore off the TTFT critical path); details in [`JITTER_ANALYSIS.md`](results/benchmarks_v5_1_1/JITTER_ANALYSIS.md).
- **Slower first-token on short prompts** than Ollama. Fundamental MLX prefill kernel cost.
- **Model conversion is slow and not user-friendly.** Squish needs models in its own format. Conversion takes time and isn't fully automated.

If any of those matter for your workflow, Ollama or LM Studio is the right choice.

---

## Architecture

**Persistent daemon.** The model loads once when the daemon starts and stays
resident. Per-invocation model-load cost becomes a once-per-login cost.

**Two-cache architecture.** A block-paged KV cache stores KV state for
fixed-size token blocks on disk (`.safetensors`) and reconstructs partial-match
prefixes for shifting-prefix workloads. A prompt KV cache catches exact-prefix
repeats with single-digit-millisecond TTFT.

**INT3 quantization with a hard-block list.** INT3 behaviour is not uniform
across model families. Qwen3 holds within ~1pp of FP16; Gemma-3 collapses
(~15pp on common benchmarks). Squish enables INT3 only for families where it's
safe and hard-blocks the rest. Try to load Gemma-3 at INT3 and the accuracy
gate refuses — you can't accidentally ship a config that quietly degrades.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues, benchmarks, and PRs welcome.
The bench harness lives in `benchmarks/ollama_vs_squish/`; if you re-run on
different hardware, please share the raw JSON output.

---

## License

BUSL-1.1 — see [LICENSE](LICENSE).

---

## Links

- Article: _Local LLM Server That Wins End-to-End on Long Contexts_ — in progress
- Org: [konjoai](https://github.com/konjoai) · [konjoai.org](https://konjoai.org)
- Related: [Kohaku](https://github.com/konjoai/kohaku), [Vectro](https://github.com/konjoai/vectro), [Squash](https://github.com/konjoai/squash) (EU AI Act compliance, extracted from squish in v9.15.0)
- HuggingFace models: [huggingface.co/squish-community](https://huggingface.co/squish-community)
- Module reference: [MODULES.md](MODULES.md)
