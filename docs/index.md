---
template: home.html
title: Squish — The fastest local LLMs on Apple Silicon
hide:
  - navigation
  - toc
---

**Local LLM inference for Apple Silicon. Faster end-to-end response on long contexts, less RAM, INT3 support.**

Squish serves quantized models (INT4 / INT3) through a fully OpenAI-compatible REST API, with an Ollama-compatible endpoint for drop-in migration — all on Apple Silicon, no GPU required.

---

## The Numbers (v9.32.0 / bench v5.1.1)

Measured 2026-06-02 on Apple M3 MacBook Pro, 16 GB unified memory.
Model: Qwen2.5-7B-Instruct. Quant: INT4 (squish) / Q4_K_M (Ollama).
Five-run medians.

| Metric | Ollama 0.18.2 | **Squish (recommended)** |
|---|---:|---:|
| **E2E response @ 4000-token prompt** | 69.63 s | **12.78 s** &nbsp;_(5.4× faster)_ |
| **E2E response @ 75-token prompt** | 8.09 s | **5.50 s** &nbsp;_(1.5× faster)_ |
| **Peak RAM during inference** | ~5 GB | **3.36 GB** |
| **Disk size — INT4** | 4.36 GB | **4.00 GB** |
| **Disk size — INT3 (Qwen3)** | not supported | **3.56 GB** |
| **TTFT @ 75-token prompt** | **131 ms** | 279 ms &nbsp;_(honest loss)_ |

**Squish wins end-to-end response time at every prompt size measured**, with the largest win on long contexts (5.4× at 4000 tokens), uses ~33% less RAM, and supports INT3 for compatible model families.

**Ollama wins time-to-first-token at every prompt size**, and inter-token jitter on long contexts. If first-byte latency matters more than full-response latency, Ollama is the right tool.

Full table, methodology, and ablation: see [Benchmark Results](RESULTS.md).

<small>M3 16 GB, thermally controlled. Cold start: Qwen2.5-1.5B. Serving: Qwen2.5-7B INT3 vs Ollama 0.30.7. **The one place Ollama wins** is single-token latency on a cold, novel prompt (167 ms vs 192 ms) — Squish leads everywhere else.</small>

---

## Key Features

- **Long-context wins** — 5.4× faster end-to-end response at 4000-token prompts vs Ollama
- **OpenAI- and Ollama-compatible API** — `/v1/chat/completions`, `/v1/models`, `/v1/completions`, plus an Ollama-compatible endpoint for drop-in migration
- **INT4 + INT3 quantization** — INT4 AWQ g=32 with a hard ≥70.6% arc_easy accuracy gate; INT3 for compatible model families (Qwen3)
- **Lower memory** — ~33% less RAM than Ollama during inference (3.36 GB vs ~5 GB on Qwen2.5-7B INT4)
- **Speculative decoding + KV cache management** — context-aware tiering (INT8/INT4/INT2) for long-context workloads
- **CLI first** — `squish pull`, `squish run`, `squish serve`, `squish rm`, `squish search`

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
    Linux/CUDA support is on the roadmap — [watch the repo](https://github.com/konjoai/squish) for updates.

---

## Community

- [GitHub Discussions](https://github.com/konjoai/squish/discussions) — get help, share benchmarks, Q&A, ideas, show & tell  
- [HuggingFace](https://huggingface.co/squishai) — pre-squished model weights  
- [Contributing](contributing.md) — good first issues, dev setup, PR guidelines  
