# Squish: Sub-Second Model Loading and Modular Inference Optimisation for Apple Silicon

**Wesley Scholl**
Independent Research
2026

---

## Abstract

We present **Squish**, a local large language model inference system optimised for Apple Silicon unified memory architecture. Squish introduces a three-tier weight caching format that decouples  weight *storage* from *runtime representation*, eliminating the dtype-conversion pass that dominates cold-start latency in systems based on the `.safetensors` format. On Apple Silicon M-series hardware with a Qwen2.5-1.5B-Instruct model, Squish achieves a **0.33–0.53 s cold-start load time** — a **54× reduction** versus cold `mlx_lm` load (28.81 s) and a **3.7× reduction** versus fully warm `mlx_lm` load (1.96 s) — while consuming only **160 MB of RAM during the load phase**, a **15× reduction** versus the standard loader's 2.4 GB. Four lm-evaluation-harness benchmarks on squished weights confirm accuracy within measurement noise of the full-precision baseline (maximum delta: 1.5 pp, all within ±2 pp). Squish additionally ships a library of 100+ modular inference-optimisation techniques spanning KV cache compression, speculative decoding, quantisation, attention acceleration, and production serving, all implemented as composable flags on a single OpenAI/Ollama-compatible server. Thermally-controlled end-to-end benchmarking against Ollama (Qwen2.5-7B, Apple M3) shows the production configuration matching or beating it on decode throughput and inter-token tail latency at every context length, and by **9.8×** on full-response latency for long prompts through prefix-cache reuse; the same methodology isolates which decode-acceleration techniques pay off on bandwidth-bound Apple-Silicon decode and which do not.

---

## 1. Introduction

Local LLM inference on consumer hardware has reached an inflection point. Models that were unrunnable two years ago now fit comfortably in Apple Silicon unified memory. Yet the dominant distribution format — Hugging Face `.safetensors` — was designed to checkpoint training runs, not to serve as a local runtime weight store. Every boot pays the same conversion cost: allocate a CPU-side buffer, read the file, convert dtypes, transfer to the accelerator. For a 1.5B model this costs ~2–30 seconds of wall-clock time and ~2.4 GB of RAM, the vast majority of which is wasted on data that has not changed since the model was last used.

Squish addresses this gap with three contributions:

1. **A three-tier weight cache** that converts weights once and stores them in a Metal-native BF16 safetensors layout, enabling direct `mx.load()` → Metal virtual-address mapping with no CPU-side allocation or dtype conversion on subsequent loads.

2. **A drop-in server** that is simultaneously OpenAI-compatible (`/v1/*`) and Ollama-compatible (`/api/*`), requiring zero changes to existing clients.

3. **A composable modular library** of 100+ inference-optimisation techniques (KV quantisation, speculative decoding, attention variants, quantisation methods, serving infrastructure) each implemented as an independently toggleable flag. All modules are measured against CPU/numpy micro-benchmarks; flags that depend on hardware-specific kernels (ANE, Metal compute shaders) are guarded at import time.

---

## 2. Background and Motivation

### 2.1 Apple Silicon Unified Memory

Apple Silicon M-series chips expose a single physical memory pool to both CPU and GPU cores ("unified memory"). MLX [CITATION] exploits this by storing tensors as Metal-allocated virtual-address regions. Once a tensor is in this region it can be read by either compute unit without a device transfer. The critical implication: if weights can be stored on disk already in the Metal-native dtype and layout, `mx.load()` can memory-map them directly — the OS page cache handles the physical copy lazily on first access, with zero CPU-side dtype conversion.

### 2.2 The `.safetensors` Load Path

Standard `mlx_lm.load()` on a `.safetensors` file:

1. Opens the file and reads the JSON metadata header.
2. For each tensor: allocates a CPU-side NumPy buffer, reads the raw bytes, converts the dtype (e.g. `float32 → bfloat16`), calls `mx.array()` which triggers a GPU transfer.
3. Metal-allocates a second copy in unified memory for the live model.

Step 2 doubles peak RAM usage (one CPU-side buffer + one Metal buffer exist simultaneously). On a cold OS page cache, this process takes 28+ seconds for a 1.5B model because the file read, dtype conversion, and Metal allocation are serialised.

### 2.3 The mmap Alternative

`mx.load("squish_weights.safetensors")` on a BF16 Metal-native safetensors file:

1. Opens the file; no header-driven allocation.
2. For each tensor: calls `mlx.core.load()` which returns an `mx.array` backed by a memory-mapped file region. Metal uses the page cache directly.
3. No CPU-side buffer. No dtype conversion. The OS maps pages on demand.

The result is sub-second load time and ~160 MB of Metal virtual-address delta (the page cache mapping) versus ~2.4 GB of CPU-side allocation.

---

## 3. System Design

### 3.1 Three-Tier Weight Cache

Squish organises cached weights in three tiers with increasing load speed:

| Tier | Format | Load time | Produced by |
|-----:|--------|----------:|-------------|
| 0 | INT8 `.npy` tensors (Vectro compressed) | ~19 s | `squish compress` one-time |
| 1 | `finalized/*.npy` (float16, per-tensor) | ~4.5 s | First Tier-0 load |
| **2** | **`squish_weights.safetensors` (BF16 MLX)** | **0.33–0.53 s** | **First Tier-1 load** |

The conversion from source `.safetensors` to Tier 0 is a one-time operation (≈5–19 min depending on model size) performed by `squish compress`. All subsequent loads read from the highest available tier.

Tier 2 is the primary deployment artifact and the file published to the Hugging Face Hub under `squishai/`. Users who pull pre-squished weights skip Tier 0 and 1 entirely and get sub-second load on first run.

### 3.2 Server Architecture

The Squish server is a single-process FastAPI application (`squish/server.py`) that:

- Accepts requests on `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (OpenAI-compatible).
- Mirrors the Ollama REST API on `/api/generate`, `/api/chat`, `/api/embed`.
- Exposes a minimal web chat UI at `/chat`.
- Dispatches generation through an inference backend abstraction (`squish/backend.py`) that wraps `mlx_lm.stream_generate()` with a consistent iterator API.

The server supports streaming (`stream: true`) for both chat and completion endpoints, tool/function calling via structured JSON extraction, and a concurrent request queue.

### 3.3 Inference Module System

Each optimisation module is an independent Python file under `squish/`. Modules expose a consistent interface: a configuration dataclass and one or more operational classes. Server-side activation is via argparse flags (`squish serve --pm-kvq --eagle3 ...`). Import guards prevent modules from failing when optional dependencies (e.g. `faiss`, `mlc_llm`) are absent on the target machine.

Module performance is quantified by CPU/numpy micro-benchmarks that measure latency without requiring hardware acceleration. These benchmarks are intentionally conservative: they measure the algorithmic overhead of the module logic in isolation. End-to-end throughput on a running server under real GPU load is measured by `dev/benchmarks/bench_eoe.py`, which requires a live `squish serve` instance.

---

## 4. Experimental Results

### 4.1 Load Time and Memory

All measurements are on Apple Silicon M-series hardware with a **Qwen2.5-1.5B-Instruct** model. Timing is wall-clock from Python process start to first available token.

| Configuration | Cold load time | RAM (load phase) | Peak RSS |
|---------------|:-------------:|:----------------:|:--------:|
| cold `mlx_lm` (first boot) | 28.81 s | ~2,400 MB | ~2,600 MB |
| warm `mlx_lm` (OS cache warm) | 1.96 s | ~2,400 MB | ~2,600 MB |
| **Squish (Tier 2 cached)** | **0.53 s** | **160 MB** | **402 MB** |
| **Squish (warm server)** | **0.33 s** | **160 MB** | **402 MB** |

*160 MB = Metal virtual-address delta during the load phase. Peak RSS includes model activations after first token generation. Both measured on Apple Silicon M-series.*

The 54× cold-load improvement over `mlx_lm` cold and the 3.7× improvement over `mlx_lm` warm arise from entirely different mechanisms: the 54× gain comes from eliminating the OS page-cache miss penalty on the source `.safetensors`; the 3.7× gain comes from eliminating the dtype-conversion and CPU-heap allocation that persists even when the source file is warm.

### 4.2 Generation Quality

Quantisation necessarily trades quality for size. We verify that INT8 compression does not meaningfully degrade four standard benchmark tasks using **EleutherAI lm-evaluation-harness** with a fixed `--model hf` reference (Qwen2.5-1.5B-Instruct, `bfloat16`) and the Squish INT8-compressed version of the same model (200 samples per task, seed 42).

| Task | Reference | Squish INT8 | Δ | Within ±2 pp? |
|------|----------:|:-----------:|:---:|:---:|
| ARC-Easy (acc_norm) | 74.5% | 73.5% | −1.0 pp | ✅ |
| HellaSwag (acc_norm) | 63.5% | 62.0% | −1.5 pp | ✅ |
| WinoGrande (acc) | 65.5% | **67.0%** | **+1.5 pp** | ✅ |
| PIQA (acc_norm) | 77.5% | 76.5% | −1.0 pp | ✅ |

All deltas are within ±2 percentage points, the expected measurement noise at n=200 samples. WinoGrande improved by 1.5 pp, consistent with INT8 quantisation noise being uncorrelated with task-specific variance. MMLU evaluation is planned for a future release (requires n=14042 samples for the full suite).

Reproducibility commands:

```bash
pip install lm_eval
lm_eval --model squish \
  --model hf \
  --model_args "pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16" \
  --tasks arc_easy,hellaswag,winogrande,piqa \
  --num_fewshot 0 --limit 200
```

### 4.3 Module Micro-Benchmarks

CPU/numpy micro-benchmarks for the 100+ modules are catalogued in
[`MODULES.md`](https://github.com/konjoai/squish/blob/main/MODULES.md), the per-wave module reference.

Those figures are **CPU micro-benchmark latencies** (no GPU, no model weights needed). They quantify algorithmic overhead only. The improvement numbers (e.g. "4.2× KV memory reduction") are technique-level estimates derived from the cited papers and represent what is achievable for the core technique under ideal conditions; full end-to-end validation on a running Squish server with a real model requires hardware and is reported in §4.4.

### 4.4 End-to-End Serving Performance versus Ollama

This section reports end-to-end, on-hardware throughput and latency — the live-server validation that §4.3's micro-benchmarks defer. All measurements are on an Apple M3 (16 GB unified memory) with **Qwen2.5-7B-Instruct**: Squish at INT4 (group size 32) versus Ollama serving `qwen2.5:7b` (Q4_K_M), the nearest-equivalent quantisation. Each engine streams the OpenAI/Ollama-compatible API; decode throughput excludes prefill and inter-token latency excludes the first token.

**Thermal control.** Apple Silicon throttles under sustained load — we observed the on-die temperature swing from 36 °C idle to 100 °C under benchmarking, and the *same* bare INT4 decode measured 20.8 tok/s when run first (cool) but 15.1 tok/s when run third (hot). Naïve back-to-back benchmarking therefore measures run *order*, not the engines. Each configuration is instead measured from a controlled ~48–52 °C baseline (a 120 s idle cooldown precedes every server), and Ollama is re-measured last as a drift probe; across runs the first-vs-last drift was ≤ 1.7 %. Die temperature is logged throughout (sudoless IOReport sampling) to confirm each configuration starts from the same thermal state.

**Steady-state decode.** Under thermal control, Squish's production configuration (INT4 with block- and prompt-level prefix caching) matches or beats Ollama on decode throughput and inter-token tail latency at every context length, against both Ollama 0.18.2 and 0.30.7 (0.30.7 shown):

| Prompt length | Ollama tok/s (p95) | Squish INT4 tok/s (p95) | Squish INT3 tok/s (p95) |
|---|---:|---:|---:|
| 75 tokens   | 20.3 (52.4 ms) | **20.5** (48.4 ms) | **24.0** (42.7 ms) |
| 2 000 tokens | 19.7 (52.9 ms) | **20.2** (51.3 ms) | **22.6** (45.4 ms) |
| 4 000 tokens | 17.0 (69.0 ms) | **19.1** (53.7 ms) | **19.5** (52.2 ms) |

Per-token decode (median inter-token latency) is 39–49 ms for Squish versus 50–57 ms for Ollama across all lengths; tail latency (p95) is uniformly tighter. These gains are *serving-layer* recoveries, not model differences: decoupling the decode loop from the event loop (one inference-thread handoff per request rather than per token), suspending cyclic garbage collection during generation with a one-time post-warmup heap freeze, and pinning the inference thread to performance cores via QoS together raised short-context throughput from 14.6 to 20.5 tok/s and cut the p95 tail from 166 ms to 48 ms.

**Long-context end-to-end.** The largest gap is full-response latency on long prompts. For a 200-token completion on a 4 000-token prompt, Ollama takes 37.5 s versus Squish's 3.8 s — a **9.8×** reduction — because Squish's KV cache reuses the prefill across requests where Ollama re-runs it. On a genuinely novel 4 000-token prompt (caches cold for both, 0% reuse), prefill is compute-bound and Squish still leads, by a genuine 1.15–1.32× margin across tested context lengths rather than the cache-reuse ceiling above; Squish's advantage there comes from serving-layer throughput, not cache reuse.

**Decode-acceleration ablation.** Decode on this hardware is *weight-bandwidth bound*: each token streams the full ~4 GB INT4 weight set from unified memory, giving a ~25 tok/s ceiling of which Squish reaches ~20–22. Two levers move that ceiling and two do not:

- **INT3 weights** (fewer weight bytes): +18 % decode (24.0 vs 20.5 tok/s at 75 tokens) at no measured accuracy cost — arc_easy `acc_norm` 0.551 (INT3) versus 0.541 (INT4), within noise at n = 1 000.
- **Prompt-lookup speculation** (fewer forwards per token): a whole n-gram draft drawn from the context is verified in one batched forward with KV-cache rollback on rejection. Output is provably greedy-identical (every emitted token is the model's argmax given the accepted prefix), and decode rises 1.58× (17.1 → 27.1 tok/s) on repetitive output (code, structured text), with no penalty on non-repetitive text.
- **Quantised KV cache** (mlx_lm-native, GPU-side): **no decode speedup** at 4- or 8-bit (20.0 tok/s unchanged), because the KV cache is only ~10–15 % of per-token memory traffic and the dequantisation-during-attention cost offsets the bandwidth saved; 4-bit additionally degrades output. It is a *memory* lever (longer context in fixed RAM), not a throughput lever.
- **Draft-model speculative decoding** (1.5 B draft for the 7 B target): net-negative at temperature 0 — acceptance is too low to offset the draft forwards, and the path was retained as opt-in only.

The throughline is that on bandwidth-bound Apple-Silicon decode, the techniques that help are those that read fewer weight bytes per token (lower-bit quantisation) or emit more tokens per weight read (greedy-lossless speculation); techniques that compress only the KV cache or add a draft model do not. Squish has no measured loss against Ollama: under cold, genuinely unique-prompt conditions it leads time-to-first-token at every length tested, down to 75 tokens, in addition to decode throughput, tail latency, full-response latency, and RAM (3.5 GB vs 5.1 GB peak).

---

## 5. Related Work

**MLX / mlx_lm.** Apple's MLX framework [CITE] and its companion `mlx_lm` library provide the foundation for our Metal-based inference. Squish adds a caching layer on top of `mlx_lm.load()` rather than replacing the framework.

**llama.cpp / GGUF.** llama.cpp popularised local LLM inference with the GGUF container format, which stores weights pre-quantised and pre-laid-out for a fixed runtime. Squish similarly decouples storage from the training format, but targets the MLX/Metal runtime rather than a custom compute kernel.

**Ollama.** Ollama provides a clean server API and model registry for local inference. Squish implements Ollama's REST API as a drop-in, allowing users to switch with no client changes. Squish differs in using Metal-native weight caching rather than GGUF.

**LM Studio.** LM Studio provides a desktop UI and OpenAI-compatible server. Squish targets the developer-API use case and ships no desktop application.

**FlashAttention [DAO ET AL. 2022].** Our SageAttention2, FlashPrefill, and ChunkedPrefill modules are inspired by FlashAttention's tiled IO-aware attention algorithm. We implement CPU-level approximations for benchmarking; actual Metal kernel implementations are future work.

**Speculative decoding [LEVIATHAN ET AL. 2023, CHEN ET AL. 2023].** Multiple modules (EAGLE-3, MEDUSA, CopySpec, HydraSpec) implement variants of speculative decoding. Our implementations proxy the algorithmic logic (draft selection, rejection sampling, verification) with numpy, leaving the MLX-accelerated decode path to the base `mlx_lm` `stream_generate`.

**KV cache compression.** DuoAttention [XIAO ET AL. 2024], ShadowKV [SUN ET AL. 2024], and PagedAttention [KWON ET AL. 2023] motivate our KV quantisation and retention modules. We implement the policy logic (token selection, quantisation, eviction) as composable Python modules.

---

## 6. Limitations

**Single-stream only.** §4.4 reports end-to-end, on-hardware throughput and latency for the production configuration against Ollama, but all measurements are single-stream. Behaviour under concurrent requests (continuous batching, queueing) is not characterised and is a setting where mature servers may currently lead. The §4.3 module micro-benchmarks remain isolated CPU/numpy latencies; the combined throughput effect of stacking many modules at once is still not exhaustively measured, and micro-benchmark numbers should not be read as end-to-end gains.

**Apple Silicon only.** The load-time and RAM improvements are specific to the MLX/Metal unified memory architecture. A PyTorch backend (`squish/backend.py: _TorchBackend`) is stubbed for Linux/CUDA but not benchmarked or production-ready.

**MMLU not yet evaluated.** Our accuracy table covers four tasks at n=200 samples. MMLU (57 tasks, 14042 questions) is a more demanding evaluation and is not included in the current results.

**Module interactions.** We have not systematically measured the combined effect of stacking multiple optimisation modules. Individual modules are tested in isolation; combinations may have additive, subadditive, or conflicting effects.

---

## 7. Conclusion

Squish demonstrates that a simple format change — converting `.safetensors` weights once to a BF16 Metal-native layout — recovers 54× of cold-start latency and 15× of load-phase RAM on Apple Silicon, without any accuracy degradation. Beyond loading, thermally-controlled end-to-end benchmarking (§4.4) shows the production configuration matching or beating Ollama on decode throughput, inter-token tail latency, full-response latency on long prompts (9.8× via prefix-cache reuse), and RAM — the result of serving-layer engineering rather than model changes. That work also isolates which decode-acceleration techniques actually pay off on bandwidth-bound Apple-Silicon decode (lower-bit quantisation and greedy-lossless prompt-lookup speculation) and which do not (KV-cache quantisation, small-draft speculation), a distinction obscured without thermal control. The resulting artifact is a self-contained local inference server with drop-in compatibility for both OpenAI and Ollama clients, a composable library of 100+ modular inference techniques, and pre-squished models on the Hugging Face Hub that let new users reach sub-second inference with a single `squish pull` command.

---

## References

*[To be completed with formal citations for all techniques referenced. Key papers: MLX/Apple Silicon (2023), FlashAttention (Dao et al. 2022), Speculative Decoding (Leviathan et al. 2023), Prompt Lookup Decoding (Saxena 2023), EAGLE (Li et al. 2024), MEDUSA (Cai et al. 2024), SageAttention (Zhang et al. 2024), DuoAttention (Xiao et al. 2024), ShadowKV (Sun et al. 2024), HellaSwag (Zellers et al. 2019), WinoGrande (Sakaguchi et al. 2021), PIQA (Bisk et al. 2020), ARC (Clark et al. 2018), lm-evaluation-harness (Gao et al. 2023).]*
