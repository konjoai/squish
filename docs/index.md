---
template: home.html
title: "Squish: fast local LLMs on Apple Silicon"
hide:
  - navigation
  - toc
---

**Local LLM inference for Apple Silicon. Faster end-to-end response on long contexts, less RAM, INT3 support.**

Squish serves quantized models (INT4 / INT3) through a fully OpenAI-compatible REST API, with an Ollama-compatible endpoint for drop-in migration, all on Apple Silicon, no GPU required.

---

## The Numbers (v9.34.8 / bench v5.1.1)

Measured on Apple M3 MacBook Pro, 16 GB unified memory, thermally controlled.
Model: Qwen2.5-7B-Instruct. Quant: INT3/INT4 (Squish) / Q4_K_M (Ollama 0.30.7).
Five-run medians.

| Metric | Ollama 0.30.7 | **Squish (recommended)** |
|---|---:|---:|
| **Cold start: load + first token (1.5B)** | 20–30 s | **≈0.5 s** &nbsp;_(54×)_ |
| **Full response @ 4000-token prompt** | 37.5 s | **3.8 s** &nbsp;_(9.8× faster)_ |
| **Decode throughput @ 75 tokens** | 20.3 tok/s | **24.0 tok/s** &nbsp;_(INT3)_ |
| **Inter-token tail p95 @ 75 tokens** | 52.4 ms | **42.7 ms** &nbsp;_(INT3)_ |
| **Repeat-prompt TTFT (KV cache hit)** | ~160 ms | **4–11 ms** |
| **Peak RAM during inference** | 5.14 GB | **3.50 GB** |
| **Disk: 7B INT4 / INT3** | 4.36 GB / n/a | **4.00 / 3.56 GB** |
| **Cold, unique-prompt TTFT @ 75 tokens** | 812 ms | **800 ms** &nbsp;_(1.15× faster)_ |

**Squish wins end-to-end response time by up to 9.8× on repeated 4000-token prompts** (the reuse ceiling — completely unique prompts see a smaller 1.15–1.32× margin), uses less RAM (3.50 vs 5.14 GB), and supports INT3 quantization for compatible model families with accuracy within measurement noise.

**Squish wins TTFT too, including on cold, unique prompts** — an earlier draft of these numbers reported Ollama winning short-prompt TTFT (167 ms vs 192 ms), but that came from a flawed same-fixed-prompt-repeated-5× methodology that was actually measuring Ollama's own cache hit, not a genuinely cold comparison. Under real cold, unique-prompt conditions Squish leads at every length tested, down to the shortest prompt measured.

Full table, methodology, and ablation: see [Benchmark Results](RESULTS.md).

---

## Key Features

- **Long-context wins**: up to 9.8× faster end-to-end response at 4000-token prompts vs Ollama 0.30.7 (on repeated prompts; 1.15–1.32× on unique ones)
- **OpenAI- and Ollama-compatible API**: `/v1/chat/completions`, `/v1/models`, `/v1/completions`, plus an Ollama-compatible endpoint for drop-in migration
- **INT4 + INT3 quantization**: INT4 AWQ g=32 with a hard ≥70.6% arc_easy accuracy gate; INT3 for compatible model families (Qwen3)
- **Lower memory**: 32% less RAM than Ollama during inference (3.50 GB vs 5.14 GB on Qwen2.5-7B)
- **Speculative decoding + KV cache management**: context-aware tiering (INT8/INT4/INT2) for long-context workloads
- **CLI first**: `squish pull`, `squish run`, `squish serve`, `squish rm`, `squish search`

---

## Quick Demo

```bash
# Install
brew install konjoai/squish/squish

# Pull a compressed model from the community hub
squish pull llama3.1:8b

# Chat interactively
squish run llama3.1:8b

# Or start the API server and query it like OpenAI
squish serve &
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Platform

!!! warning "macOS + Apple Silicon only"
    Squish uses [Apple MLX](https://github.com/ml-explore/mlx) for inference and requires an M1–M5 chip.
    Linux/CUDA support is on the roadmap. [Watch the repo](https://github.com/konjoai/squish) for updates.

---

## Community

- [GitHub Discussions](https://github.com/konjoai/squish/discussions): get help, share benchmarks, Q&A, ideas, show & tell  
- [HuggingFace](https://huggingface.co/squishai): pre-squished model weights  
- [Contributing](contributing.md): good first issues, dev setup, PR guidelines  
