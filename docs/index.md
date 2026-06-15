---
template: home.html
title: Squish — Run 70B models on a MacBook
hide:
  - navigation
  - toc
---

Squish compresses model weights into memory-mapped tensors that load in **milliseconds**, then serves them through a fully OpenAI-compatible REST API — faster than Ollama, all on Apple Silicon, no GPU required.

---

## Why Squish?

| | Ollama | LM Studio | **Squish** |
|---|---|---|---|
| Cold start (load + first token) | ~30 s | ~20 s | **< 1 s** |
| Decode throughput (7B) | 20 tok/s | — | **24 tok/s** |
| Full response @ 4000-token prompt | 37 s | — | **3.8 s** |
| Peak RAM (serving 7B) | 5.1 GB | — | **3.5 GB** |
| OpenAI API | ✅ | ✅ | ✅ |
| Batch requests | ❌ | ❌ | **✅** |
| Pre-compressed weights | ❌ | ❌ | **✅ HuggingFace** |
| Quantisation | GGUF | GGUF | **INT4 / INT3 / INT8** |
| Platform | macOS/Linux | macOS/Windows | macOS (M1–M5) |

<small>M3 16 GB, thermally controlled. Cold start: Qwen2.5-1.5B. Serving: Qwen2.5-7B INT3 vs Ollama 0.30.7.</small>

---

## Key Features

- **Instant load** — memory-mapped weights skip all decoding overhead
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`, `/v1/completions`
- **Batch inference** — parallel requests in a single call
- **INT4 / INT3 / INT8** — quantisation tiers for accuracy vs. size trade-offs; INT3 is the recommended default (≈18% faster decode at no measured accuracy cost vs INT4)
- **Zero-copy mmap** — model data never fully loaded into RAM
- **CLI first** — `squish pull`, `squish run`, `squish serve`, `squish rm`, `squish search`

---

## Quick Demo

```bash
# Install
brew install squish-ai/squish

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
