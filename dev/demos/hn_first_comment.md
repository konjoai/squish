# HN First Comment Template

Template for the first Hacker News comment after submitting squish.

---

**Paragraph 1 — The Problem**

Most local LLM runners have a cold-start problem that nobody talks about. Ollama
takes 8-25 seconds to load a 1.5B model before the first token appears — even on
Apple Silicon hardware that is physically capable of loading the file in under a
second. The bottleneck is not memory bandwidth: it is dtype conversion happening
on every load. Stock safetensors files are BF16; the runner converts to FP32
or FP16 at load time, touching every byte twice. On a 7B model that is ~14 GB
of memory writes before inference even starts.

**Paragraph 2 — How Squish Solves It**

Squish stores weights in a format that maps directly into Metal unified memory
without any dtype conversion. The three key decisions: (1) mmap instead of
read() so the OS handles caching across runs, (2) INT4 quantization at ~1.5 GB
for a 1.5B model so the working set fits in L2/L3 cache, and (3) a Metal GPU
warmup pass on the first 8 tokens so the ANE is at peak throughput when the
user's actual prompt arrives. The result is a 0.33-0.53 s cold start on an M3
MacBook Pro — roughly 20-50x faster than Ollama on the same hardware. RAM
during load is ~160 MB because we only map the pages we touch, not the full
file.

**Paragraph 3 — Honest Caveats**

A few things to be upfront about: squish currently requires M-series Apple
Silicon — the mmap + Metal path is macOS-specific and there is no CUDA backend
yet. Several features are labeled [EXPERIMENTAL] in the README (EAGLE-3
speculative decoding, the paged KV cache, the DFloat11 entropy layer) because
they work well in benchmarks but have not been stress-tested across all model
families. The INT4 quantization has a <2% accuracy delta on standard benchmarks
for Qwen and Llama models, but results vary for less common architectures.
Raw benchmark JSON and lm-eval data are publicly accessible at
`dev/benchmarks/` in the repo so you can reproduce or challenge the numbers.
Happy to answer technical questions about the quantization scheme, the Metal
dispatch path, or anything else.
