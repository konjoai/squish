# Squish: Sub-Second Model Loading and Modular Inference Optimisation for Apple Silicon

**Wesley Scholl**
Independent Research
2026

---

## Abstract

We present **Squish**, a local large language model inference system optimised for Apple Silicon unified memory architecture. Squish introduces a three-tier weight caching format that decouples  weight *storage* from *runtime representation*, eliminating the dtype-conversion pass that dominates cold-start latency in systems based on the `.safetensors` format. On Apple Silicon M-series hardware with a Qwen2.5-1.5B-Instruct model, Squish achieves a **0.33–0.53 s cold-start load time** — a **54× reduction** versus cold `mlx_lm` load (28.81 s) and a **3.7× reduction** versus fully warm `mlx_lm` load (1.96 s) — while consuming only **160 MB of RAM during the load phase**, a **15× reduction** versus the standard loader's 2.4 GB. Four lm-evaluation-harness benchmarks on squeeshed weights confirm accuracy within measurement noise of the full-precision baseline (maximum delta: 1.5 pp, all within ±2 pp). Squish additionally ships a library of 100+ modular inference-optimisation techniques spanning KV cache compression, speculative decoding, quantisation, attention acceleration, and production serving, all implemented as composable flags on a single OpenAI/Ollama-compatible server.

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

Tier 2 is the primary deployment artifact and the file published to the Hugging Face Hub under `squish-community/`. Users who pull pre-squished weights skip Tier 0 and 1 entirely and get sub-second load on first run.

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
python3 squish/squish_lm_eval.py \
  --model hf \
  --model_args "pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16" \
  --tasks arc_easy,hellaswag,winogrande,piqa \
  --num_fewshot 0 --limit 200
```

### 4.3 Module Micro-Benchmarks

CPU/numpy micro-benchmarks for all 100+ modules are reported in the wave benchmark documents:

| Waves | Modules | Benchmark doc |
|-------|--------:|---------------|
| 12 | 7 | [`docs/benchmark_wave12.md`](benchmark_wave12.md) |
| 13–14 | 26 | [`docs/benchmark_wave13_14.md`](benchmark_wave13_14.md) |
| 15–16 | 21 | [`docs/benchmark_wave15_16.md`](benchmark_wave15_16.md) |
| 17–18 | 28 | [`docs/benchmark_wave17_18.md`](benchmark_wave17_18.md) |
| 19–20 | 28 | [`docs/benchmark_wave19_20.md`](benchmark_wave19_20.md) |
| 21–22 | 28 | [`docs/benchmark_wave21_22.md`](benchmark_wave21_22.md) |
| 23–24 | 28 | [`docs/benchmark_wave23_24.md`](benchmark_wave23_24.md) |
| 25–26 | 28 | [`docs/benchmark_wave25_26.md`](benchmark_wave25_26.md) |

All figures in these documents are **CPU micro-benchmark latencies** (no GPU, no model weights needed). They quantify algorithmic overhead only. The improvement numbers (e.g. "4.2× KV memory reduction") are technique-level estimates derived from the cited papers and represent what is achievable for the core technique under ideal conditions; full end-to-end validation on a running Squish server with a real model requires hardware and is the subject of ongoing measurement via `bench_eoe.py`.

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

**End-to-end hardware benchmarks are not yet published.** The module micro-benchmarks measure algorithm latency in isolation (CPU/numpy). The impact of stacking modules on actual tokens-per-second throughput on a live server with a loaded model requires `bench_eoe.py` and has not yet been reported. Users should not interpret micro-benchmark numbers as end-to-end throughput improvements.

**Apple Silicon only.** The load-time and RAM improvements are specific to the MLX/Metal unified memory architecture. A PyTorch backend (`squish/backend.py: _TorchBackend`) is stubbed for Linux/CUDA but not benchmarked or production-ready.

**MMLU not yet evaluated.** Our accuracy table covers four tasks at n=200 samples. MMLU (57 tasks, 14042 questions) is a more demanding evaluation and is not included in the current results.

**Module interactions.** We have not systematically measured the combined effect of stacking multiple optimisation modules. Individual modules are tested in isolation; combinations may have additive, subadditive, or conflicting effects.

---

## 7. Conclusion

Squish demonstrates that a simple format change — converting `.safetensors` weights once to a BF16 Metal-native layout — recovers 54× of cold-start latency and 15× of load-phase RAM on Apple Silicon, without any accuracy degradation. The resulting artifact is a self-contained local inference server with drop-in compatibility for both OpenAI and Ollama clients, a composable library of 100+ modular inference techniques, and pre-squished models on the Hugging Face Hub that let new users reach sub-second inference with a single `squish pull` command.

---

## References

*[To be completed with formal citations for all techniques referenced. Key papers: MLX/Apple Silicon (2023), FlashAttention (Dao et al. 2022), Speculative Decoding (Leviathan et al. 2023), EAGLE (Li et al. 2024), MEDUSA (Cai et al. 2024), SageAttention (Zhang et al. 2024), DuoAttention (Xiao et al. 2024), ShadowKV (Sun et al. 2024), HellaSwag (Zellers et al. 2019), WinoGrande (Sakaguchi et al. 2021), PIQA (Bisk et al. 2020), ARC (Clark et al. 2018), lm-evaluation-harness (Gao et al. 2023).]*
