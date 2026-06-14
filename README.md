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

Measured 2026-06-14 on Apple M3 MacBook Pro, 16 GB unified memory.
Model: Qwen2.5-7B-Instruct — Squish INT4 (g=32) vs Ollama `qwen2.5:7b` (Q4_K_M).
**Thermally controlled**: each engine is measured from the same ~50 °C baseline
(120 s cooldown per config), validated by a first-vs-last drift check (≤ 1.7 %)
and live die-temperature logging — so the numbers reflect the engines, not the
order they ran. Tested against **both Ollama 0.18.2 and 0.30.7**; the table
shows 0.30.7 (current), with 0.18.2 within noise.

| Metric | Ollama 0.30.7 | **Squish (recommended)** |
|---|---:|---:|
| **E2E response @ 4000-token prompt** | 37.5 s | **3.8 s** &nbsp;_(9.8× faster)_ |
| **E2E response @ 75-token prompt** | 2.68 s | **2.58 s** |
| **Warm decode @ 75-token** | 20.3 tok/s | **20.5 tok/s** &nbsp;_(INT3: 24.0)_ |
| **Inter-token p95 @ 75-token** | 52.4 ms | **48.4 ms** &nbsp;_(INT3: 42.7)_ |
| **Peak RAM during inference** | 5.14 GB | **3.50 GB** |
| **Disk size — INT4** | 4.36 GB | **4.00 GB** |
| **Disk size — INT3** | not supported | **3.56 GB** |
| **TTFT @ 75-token prompt** | **167 ms** | 192 ms &nbsp;_(honest loss)_ |

**Squish wins end-to-end response time, decode throughput, inter-token tail
latency, and RAM** — by far the largest win on long contexts (**9.8× at 4000
tokens**, where Squish's block/prompt KV cache reuses the prefill instead of
re-running it). INT3 adds another ~18 % decode (24 tok/s) at validated accuracy
parity with INT4 (arc_easy acc_norm within noise).

**Ollama wins time-to-first-token** on a cold short prompt (167 ms vs 192 ms).
If first-byte latency on novel prompts matters more than full-response latency,
Ollama is the right tool.

Full methodology, thermal control, and ablation:
[docs/benchmark_guide.md](docs/benchmark_guide.md)

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

Requires Python 3.11–3.14 and macOS 13 (Ventura) or later on Apple Silicon.

### Homebrew (recommended)

```bash
brew tap konjoai/squish
brew install squish
squish doctor
```

The Homebrew formula pins to Python 3.13 and pre-bundles all binary
dependencies — no compilation, no network access during install.

### pip / pipx

```bash
# pipx (manages isolation automatically)
pipx install squish-ai --python python3.13
squish doctor

# or in a virtual environment
python3.13 -m venv ~/.squish_env
source ~/.squish_env/bin/activate
pip install squish-ai
squish doctor
```

> **macOS Tahoe (26) users:** macOS 26 ships Python 3.14 as the default
> `python3`. squish-ai 9.33.6+ supports Python 3.14 natively. For older
> releases use `--python python3.13` with pipx, or create a venv with
> `python3.13` explicitly (`brew install python@3.13` if needed).

### From source

```bash
git clone https://github.com/konjoai/squish
cd squish
pip install -e .
```

> **Package name vs CLI name:** The PyPI package is `squish-ai` but the CLI
> and Python module are both `squish`:
>
> ```bash
> pip install squish-ai
> squish --version
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
