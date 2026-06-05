# Squish

<img src="assets/squish-logo-1.png" height="320" alt="Squish Logo"/>

## Squeeze the Most Out of Your Models

**Local LLM inference for Apple Silicon. Faster end-to-end response on long contexts, less RAM, INT3 support.**

[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/squish-ai.svg)](https://pypi.org/project/squish-ai/)
[![CI](https://github.com/konjoai/squish/actions/workflows/ci.yml/badge.svg)](https://github.com/konjoai/squish/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](https://github.com/konjoai/squish)
[![HuggingFace](https://img.shields.io/badge/🤗%20Models-squishai-yellow)](https://huggingface.co/squishai)

---

## The Numbers

Measured 2026-06-04 on Apple M3 MacBook Pro, 16 GB unified memory.
Model: Qwen3-8B. Quant: INT4.

| Metric | Ollama 0.18.2 | **Squish (recommended)** |
|---|---:|---:|
| **E2E response @ 4000-token prompt** | 51.7 s | **10.1 s** &nbsp;_(5.1× faster)_ |
| **E2E response @ 75-token prompt** | 8.09 s | **5.50 s** &nbsp;_(1.5× faster)_ |
| **Peak RAM during inference** | 5.32 GB | **2.75 GB** |
| **Disk size — INT4** | 4.36 GB | **4.00 GB** |
| **Disk size — INT3 (Qwen3)** | not supported | **3.56 GB** |
| **TTFT @ 75-token prompt** | **131 ms** | 279 ms &nbsp;_(honest loss)_ |

**Squish wins end-to-end response time at every prompt size measured**, with
the largest win on long contexts (5.4× at 4000 tokens), uses ~33% less RAM,
and supports INT3 for compatible model families.

**Ollama wins time-to-first-token at every prompt size**, and inter-token
jitter on long contexts. If first-byte latency matters more than full-response
latency, Ollama is the right tool.

Full methodology and ablation: [docs/benchmark_guide.md](docs/benchmark_guide.md)

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
brew install konjoai/squish/squish

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

The squish_quant Rust extension is bundled and installs automatically.
Verify it is active with squish doctor — you should see:

```
✓  squish_quant Rust extension (6 GB/s quantizer)
```

---

## Models

```bash
squish catalog                 # browse all 40+ available models
squish search qwen3            # filter by name or tag
squish pull qwen3:8b           # download pre-squished from huggingface.co/squishai
squish pull qwen3:0.6b --int3  # INT3 variant (Qwen3, Qwen2.5, Llama families)
```

---

## Quick Start

```bash
# Pull a pre-quantised model from the catalog
squish pull qwen3:8b

# Start the daemon
squish run qwen3:8b
```

Use it as an OpenAI-compatible client:

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:8b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Or point any OpenAI / Ollama client at it:

```bash
export OPENAI_BASE_URL=http://localhost:11435/v1
export OPENAI_API_KEY=squish
# Ollama-compatible /api/* endpoints also work
export OLLAMA_HOST=http://localhost:11435
```

Install the macOS LaunchAgent so the daemon starts at login:

```bash
squish daemon install
```

The **SquishBar** menu-bar app (`apps/macos/SquishBar/`) ships alongside the
CLI and gives you a native menu bar icon with server status, tok/s display,
and one-click model switching. Build locally with make (requires Xcode 15+
and macOS 13+):

```bash
cd apps/macos/SquishBar
make
open SquishBar.app
```

---

## Configuration

See [Server Flags](docs/api.md#server-flags) for the full flag reference.

---

## Benchmarks

Full table, methodology, ablation, and raw per-run JSON:

- [`docs/benchmark_guide.md`](docs/benchmark_guide.md) — bench methodology and how to reproduce
- [`benchmarks/ollama_vs_squish/RESULTS.md`](benchmarks/ollama_vs_squish/RESULTS.md) — raw results

Reproduce locally:

```bash
bash scripts/test_cli.sh
```

---

## What Squish Doesn't Do

In the spirit of honesty:

- **No GPU support outside Apple Silicon.** It's MLX-based. CUDA users should use vLLM or llama.cpp.
- **No multi-user serving.** Designed for one developer, one machine — not a production API.
- **No multimodal models.** Text only.
- **Higher inter-token p95 on long prompts** than Ollama. Conscious tradeoff (deferred KV-cache restore off the TTFT critical path); details in JITTER_ANALYSIS.md.
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

---

## License

BUSL-1.1 — see [LICENSE](LICENSE).

---

## Links

- Article: _I Tried Every Local LLM Tool. None of Them Were Fast Enough. So I Built My Own._
- Org: [konjoai](https://github.com/konjoai)
- Related: [Kohaku](https://github.com/konjoai/kohaku), [Vectro](https://github.com/konjoai/vectro), [Squash](https://github.com/konjoai/squash) (EU AI Act compliance, extracted from squish in v9.15.0)
- HuggingFace models: [huggingface.co/squishai](https://huggingface.co/squishai)
- Module reference: [MODULES.md](MODULES.md)
