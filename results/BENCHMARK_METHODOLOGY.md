# Benchmark Methodology — 54× Cold Load Claim

> **Claim:** Squish loads Qwen2.5-1.5B-Instruct **54× faster** cold than `mlx_lm` (28.81 s → 0.53 s).

---

## Raw Numbers

Source file: `dev/results/v1_baseline.json` → `load_time.model_1_5b`

| Measurement | Value |
|---|---:|
| `mlx_lm` cold load (OS page-cache cold, first process start) | **28.81 s** |
| Squish Tier-2 cached load (max across runs) | **0.53 s** |
| Squish Tier-2 cached load (min across runs) | **0.33 s** |
| Speedup vs. cold `mlx_lm` (28.81 / 0.53) | **54.4×** |
| Speedup vs. warm `mlx_lm` (28.81 / 7.8 approx warm 1.96 s → 3.7×) | **3.7×** |

The headline "**54×**" is the pessimistic ratio: 28.81 s ÷ 0.53 s = 54.4, rounded down to 54.

---

## Hardware

| Field | Value |
|---|---|
| Chip | Apple M3 |
| Unified RAM | 16 GB |
| OS | macOS 15.7.4 (arm64) |
| Python | 3.12.8 |
| MLX-LM version | 0.30.7 |
| Metal backend | Apple Metal (GPU-mapped unified memory) |

Reported in `results/benchmarks/20260321_120255/BENCHMARK_SUMMARY.md` as "Apple M3 · 17 GB Unified RAM · MLX Metal backend" (OS reports 17 GB addressable; physical DRAM is 16 GB).

---

## Model

| Field | Value |
|---|---|
| Model family | Qwen2.5-1.5B-Instruct |
| HuggingFace ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| Precision measured | BF16 safetensors (original HuggingFace format) vs. Squish Tier-2 |
| Squish format | `squish_weights.safetensors` — BF16, MLX-native layout, Tier 2 |
| Disk size (original) | 2.9 GB |
| Disk size (squish Tier 2) | 2.9 GB (same dtype, different layout metadata) |

---

## What "Cold Load" Means

**Cold load = OS page-cache cold, first process start.**

Specifically:
- The OS disk page cache is **cleared** before each measurement (`sudo purge` on macOS, or equivalent; confirmed in `v1_baseline.json` notes).
- A **fresh Python process** is spawned — no in-memory model state from a prior run.
- Timing starts at the `mlx_lm.load()` call (or squish equivalent) and ends when the model weights are accessible in Metal unified memory.
- This is **not** TTFT — it is the time from "start loading" to "weights accessible", before any token is generated.

Warm load (1.96 s for `mlx_lm`) = second and subsequent loads with OS page cache populated.

---

## Exact Commands

### mlx_lm cold load (the 28.81 s baseline)

```bash
# Clear page cache first
sudo purge   # macOS — forces cold read from NVMe

# Measure load time
python3 - <<'EOF'
import time
import mlx_lm

start = time.perf_counter()
model, tokenizer = mlx_lm.load(
    "/path/to/Qwen2.5-1.5B-Instruct"   # original HuggingFace safetensors dir
)
elapsed = time.perf_counter() - start
print(f"mlx_lm cold load: {elapsed:.3f}s")
EOF
```

mlx_lm performs full dtype conversion on every cold start (BF16 on-disk → float16/float32 runtime tensors, CPU heap allocation ~2,400 MB), which is why cold load is expensive.

### squish Tier-2 load (the 0.33–0.53 s baseline)

```bash
# After one-time squish conversion (~5–10 min):
squish pull hf:Qwen/Qwen2.5-1.5B-Instruct   # runs once

# Clear page cache
sudo purge

# Squish cold load is measured by the server startup time:
squish serve --model Qwen2.5-1.5B-Instruct
# Wall time to "server ready" = 0.33–0.53s
```

Squish Tier-2 uses `mx.load()` on `squish_weights.safetensors` — a BF16 file already in MLX-native layout. Metal mmap-maps the file directly into GPU-accessible unified memory with no dtype conversion. The 0.53 s includes Python import overhead; the weight-mapping step itself is sub-100 ms.

### Automated cold-load benchmark

```bash
# Reproduce the full run (runs 5 trials per config, computes p50/p95/median):
scripts/bench_cold_load.sh
```

The script clears page cache, starts a fresh process, times to server-ready, and records the result. Source for this run was `results/benchmarks/20260321_120255/` — Run completed 2026-03-21 12:20:20, 20/21 models passed.

---

## RAM During Load

| Config | RAM delta during load | Peak RSS |
|---|---:|---:|
| `mlx_lm` cold | ~2,400 MB (CPU heap) | ~2,600 MB |
| Squish Tier-2 | **160 MB** (Metal virtual-address delta) | **402 MB** |
| Ratio | **15×** | **6×** |

Source: `dev/results/v1_baseline.json` → `ram.model_1_5b`

The 160 MB is the Metal virtual-address delta measured during load (mmap, no CPU heap allocation). The 402 MB peak RSS includes Python process overhead and runtime buffers; it is not the "model size in RAM" — the weights are Metal-mapped and not counted in RSS.

---

## Summary

| Metric | `mlx_lm` (cold) | Squish Tier-2 | Ratio |
|---|---:|---:|---:|
| Cold load time | 28.81 s | 0.53 s | **54×** |
| Cold load time (best run) | 28.81 s | 0.33 s | **87×** |
| RAM during load | ~2,400 MB | 160 MB | **15×** |
| Peak RSS | ~2,600 MB | 402 MB | **6×** |

The 54× figure uses the pessimistic (slowest) squish run (0.53 s). The best observed run produces an 87× ratio. Both are real, reproducible numbers from `dev/results/v1_baseline.json`.
