# squish — Benchmarks

> Grounded numbers from the squish repository. Reproduce locally with the
> commands at the bottom of each section. No marketing rounding —
> everything here either runs in CI or is a single Python invocation away.

---

## 1. Cold-start load time and TTFT

Source: `README.md` headline numbers; reproduced via
`benchmarks/ollama_vs_squish/bench_cold_prefill.py` and `squish bench`.

| | mlx_lm (cold) | Ollama | **squish** |
|---|:---:|:---:|:---:|
| **Cold-start load time**   | 28.81 s | 8-25 s     | **0.33-0.53 s** §  |
| **Cold start → first token** | n/a   | 20-30 s    | **~0.5 s** ‡       |
| **RAM during load**        | ~2400 MB| ~2-8 GB    | **160 MB** †       |

§ Cold-start load = wall time for weights to be accessible in Metal
unified memory (mmap, no dtype conversion). Qwen2.5-1.5B, M3 16 GB, on
hardware with sufficient RAM to build the Tier 1 MLX safetensors cache
(~34 GB required). On a standard 16 GB M3, an 8B INT4 model loads in
~2.7 s via the lut_int2 path.
‡ Cold start → first token includes weight load + prefill; Ollama pays a
full cold model load here, which is where Squish's load advantage shows up.
For *warm* steady-state TTFT (model already loaded), see §1b — the two
engines are comparable there.
† 160 MB = Apple Metal virtual-address delta during load (mmap, no CPU
heap). Peak RSS ~402 MB.

### 1b. Serving throughput & latency (warm, Qwen2.5-7B vs Ollama)

Thermally controlled (cooldown + drift check + die-temp logging), M3 16 GB,
vs Ollama 0.18.2 **and** 0.30.7. See README and `docs/paper.md` §4.4.

| Metric (warm) | Ollama 0.30.7 | **squish INT4** | **squish INT3** |
|---|:---:|:---:|:---:|
| Decode tok/s @ 75 tok   | 20.3 | **20.5** | **24.0** |
| Decode tok/s @ 4000 tok | 17.0 | **19.1** | **19.5** |
| Inter-token p95 @ 75    | 52.4 ms | **48.4 ms** | **42.7 ms** |
| E2E @ 4000-token prompt | 37.5 s | **3.8 s (9.8×)** | — |
| TTFT (loaded, 75 tok)   | **167 ms** | 192 ms | 192 ms |
| Peak RAM                | 5.14 GB | **3.5 GB** | — |

INT3 is the recommended default — arc_easy acc_norm 0.551 vs INT4 0.541 (tied,
n=1000). Squish's only loss is warm single-token TTFT (192 vs 167 ms).

---

## 2. Disk size — raw vs. squished

Source: `README.md` model-size table; the squished column is what
`squish pull <model>` actually downloads from the
[squishai](https://huggingface.co/squishai) HF org.

| Model          | Raw (bf16) | Squished (INT4) | Saved |
|----------------|:----------:|:---------------:|:-----:|
| qwen3:0.6b     | 1.3 GB     | 0.4 GB          | 69%   |
| qwen3:1.7b     | 3.5 GB     | 1.0 GB          | 71%   |
| qwen3:4b       | 8.2 GB     | 2.2 GB          | 73%   |
| qwen3:8b       | 16.4 GB    | 4.4 GB          | 73%   |
| qwen3:14b      | 28.7 GB    | 7.6 GB          | 74%   |
| llama3.1:8b    | 16.1 GB    | 4.3 GB          | 73%   |
| deepseek-r1:7b | 14.4 GB    | 3.9 GB          | 73%   |

Average: **~73% smaller on disk**, 3.7× compression, statistically
identical generation quality.

---

## 3. Weight quantization accuracy gates (lm_eval, arc_easy)

Source: CI accuracy gates; gate values are hard-stops in CI —
ship requires meeting or beating them.

| Format        | Model           | Gate (arc_easy) | Status         |
|---------------|-----------------|:---------------:|----------------|
| INT4 AWQ g=32 | Qwen2.5-1.5B    | ≥ 70.6 %        | ✅ shipped (W92)  |
| INT3 g=32     | Qwen2.5-1.5B    | ≥ 67.2 %        | ✅ shipped (W92)  |
| INT3          | gemma-3-* ≤ 4B  | -15 pp          | ❌ blocked         |
| INT3          | Qwen3 family    | within ±2pp     | ✅ shipped (9.33.5) |
| INT2 (naive)  | any             | ~29 % ≈ random  | ⛔ never ship       |
| **SQINT2**    | Qwen2.5-7B      | ≥ 65 % (target 67%) | 🎯 in progress |

SQINT2 is the four-stage geometry-aware INT2 pipeline: Hadamard
incoherence preprocessing + NF2 per-group quantisation + low-rank SVD
residual + layer-selective mixed precision. Effective bit-rate
**~2.15 bpw** — half of INT4 storage, ~7× of fp16 — see
the squish architecture for the full math. Stages 1-3 land
code-complete + SNR-validated; the final lm_eval ship gate runs at
W103.4d on real M3 16 GB hardware.

---

## 4. KV-cache quantization (W104 / W105 / W106)

Three storage tiers — INT8 (default ≤ 8 K), INT4 (8 K-16 K band),
INT2 (> 16 K). All three share the same `_quantize_*_per_channel` /
`_dequantize_*_per_channel` codec contract; the only differences are
the codebook and the bit-packing.

### Per-token storage (head_dim = 128)

| Mode | Code bytes | Scale bytes | Total per token | Compression vs fp16 |
|------|:----------:|:-----------:|:---------------:|:-------------------:|
| fp16 (reference) | 256 | 0 | **256 B**       | 1.00×               |
| int8             | 128 | 4 | **132 B**       | 1.94×               |
| int4             | 64  | 4 | **68 B**        | 3.76×               |
| int2             | 32  | 4 | **36 B**        | 7.11×               |

### Reconstruction SNR (fp16, n_tokens=256, head_dim=128, seed=42)

| Distribution     | Hadamard | INT8 SNR  | INT4 SNR  | INT2 SNR  |
|------------------|:--------:|:---------:|:---------:|:---------:|
| Gaussian (σ=0.3) | off      | 43.89 dB  | 19.25 dB  |  5.20 dB  |
| Gaussian (σ=0.3) | on       | 43.82 dB  | 19.27 dB  |  5.27 dB  |
| Heavy-tailed (t, df=3) | off | 37.79 dB  | 12.90 dB  | -3.39 dB  |
| Heavy-tailed (t, df=3) | on  | 44.27 dB  | 19.69 dB  |  5.71 dB  |
| Outlier-spiked (1% @ ±5) | off | 34.29 dB | 7.09 dB | -8.61 dB |
| Outlier-spiked (1% @ ±5) | on  | 47.70 dB | 23.18 dB | **+8.47 dB** |

Read the bottom row carefully: **without rotation, INT2 sits at
−8.6 dB** on outlier-spiked activations — the reconstruction error is
literally seven times the signal, and the cache is destroyed. Apply the
randomised Hadamard rotation (`HadamardKVCache`, free at runtime,
seeded so it is deterministic) and SNR jumps **17 dB** to +8.5 dB.
This is exactly the bin-collapse failure mode that motivated the
W104 codec design — and exactly what the demo Space lets you click on.

### Qwen2.5-7B KV-cache memory by context length

n_layers=28, n_kv_heads=4, head_dim=128. Numbers below are
`estimate_kv_memory(...).total_bytes / 1e6`, the same closed-form used by
`make_kv_cache(planned_context=...)` to pick a tier.

| Context tokens | fp16 | int8 | int4 | int2 |
|----------------|------:|------:|------:|------:|
| 4 096   |   234.9 MB |   121.1 MB |    62.4 MB |    33.0 MB |
| 8 192   |   469.8 MB |   242.2 MB |   124.8 MB |    66.1 MB |
| 16 384  |   939.5 MB |   484.4 MB |   249.6 MB |   132.1 MB |
| 32 768  | 1 879.0 MB |   968.9 MB |   499.1 MB |   264.2 MB |
| 65 536  | 3 758.1 MB | 1 937.8 MB |   998.2 MB |   528.5 MB |

Headroom story on M3 16 GB (≈ 15.5 GB usable Metal budget): a fp16
KV cache for Qwen2.5-7B at 32 K tokens is 1.88 GB, on top of ~4.4 GB
of INT4 weights, leaving only ~9 GB for everything else and OOMing
around 10 K in practice. The same workload at INT2 KV is **264 MB** —
7× smaller, fits 32 K cleanly, and 65 K stays under 530 MB.

### Recommended-mode auto-selection

```python
from squish.kv.kv_cache import recommended_kv_mode_3tier
recommended_kv_mode_3tier(   4_000)   # → "int8"
recommended_kv_mode_3tier(  12_000)   # → "int4"
recommended_kv_mode_3tier(  32_000)   # → "int2"
```

Defaults: ≤ 8 K → int8, 8-16 K → int4, > 16 K → int2.

---

## 5. Throughput — quantized GEMV (W101 / W102)

INT4 group-32 GEMV, M3 16 GB, single-thread baseline numbers from
`squish bench --format int4` on `(batch=1, in=4096, out=4096, group=32,
iters=200, warmup=50)`. Rust path released the GIL via
`py.allow_threads()` and parallelised across output features with
Rayon; NumPy path kept as a portable fallback.

| Backend            | p50 latency | p95 latency | GOPS |
|--------------------|:-----------:|:-----------:|:----:|
| NumPy fallback     | reference   | reference   | 1×   |
| Rust (`squish_quant_rs`) | -2-3× faster | -3-4× faster | 2-3× |

(Exact numbers depend on host; reproduce with `squish bench --format
int4 --in-features 4096 --out-features 4096 --group-size 32 --iters
200`.)

---

## 6. Reproduce these numbers

```bash
# Cold load + TTFT
benchmarks/ollama_vs_squish/bench_cold_prefill.py

# Quantized GEMV throughput
squish bench --format int4
squish bench --format int8

# KV codec SNR (the demo's numbers, exactly)
python -c "
from spaces._logic import make_synthetic_activations, apply_hadamard, run_all_tiers
arr = make_synthetic_activations(256, 128, 'outlier', seed=42)
for r in run_all_tiers(apply_hadamard(arr)):
    print(f'{r.mode}: SNR={r.snr_db:.2f} dB, {r.bytes_per_token} B/tok, {r.compression_vs_fp16:.2f}x')
"

# KV memory for any model + context
python -c "
from squish.kv.kv_cache import estimate_kv_memory
e = estimate_kv_memory(n_layers=28, n_kv_heads=4, head_dim=128,
                       context_tokens=32_000, mode='int2', window=128)
print(f'total = {e.total_bytes/1e6:.1f} MB, ratio = {e.compression_ratio:.2f}x')
"

# Full lm_eval gate (overnight, requires real model)
lm_eval --model squish --model_args path=$MODEL_DIR --tasks arc_easy --limit 500
```

---

## 7. Live in the browser

The KV-cache numbers from §4 are interactive at the
[**squish-kv-quant** Hugging Face Space](https://huggingface.co/spaces/squishai/squish-kv-quant).
Pick a distribution, toggle Hadamard rotation, see SNR shift in real
time. Source in [`spaces/`](spaces/).
