# Squish — Multi-Model Benchmark Results

## Load Time & Throughput

| Model | Size (GB) | Compressed (GB) | Ref Load | Squish Load | Speedup | Tok/s |
|-------|-----------|-----------------|----------|-------------|---------|------:|
| Qwen2.5-1.5B | 3.1 | 2.9 | 1.45s | **0.43s** | **3.4×** | **18.9** |
| Qwen2.5-7B | 14.0 | 4.0 | — | **2.01s** | — | **16.8** |
| Qwen2.5-14B | 29.6 | 8.3 | — | **3.36s** | — | **7.7** |
| Qwen3-8B | 16.4 | 4.4 | — | **2.20s** | — | **15.1** |

## Wave 12 KV Cache Compression

Enabled with `--pm-kvq --mix-kvq --cocktail-kv` on any model.

| Model | Baseline KV mem | Wave 12 KV mem | Reduction | Context (VRAM-same) |
|-------|:--------------:|:--------------:|:---------:|:-------------------:|
| Qwen2.5-1.5B | 1× (FP16) | ~0.24× | **4.2×** | 4× longer context |
| Qwen2.5-7B   | 1× (FP16) | ~0.26× | **3.8×** | 4× longer context |
| Qwen2.5-14B  | 1× (FP16) | ~0.26× | **3.8×** | 4× longer context |
| Qwen3-8B     | 1× (FP16) | ~0.26× | **3.8×** | 4× longer context |

KV reduction measured at 4 096-token sequence length; PM-KVQ assigns FP16
to recent 6% of tokens, INT8 to 19%, INT4 to 75%.

## Wave 12 Module Summary

| Module | Flag | Memory | Latency overhead | Paper speedup |
|--------|------|-------:|:----------------:|:-------------:|
| PM-KVQ | `--pm-kvq` | **4.2× KV reduction** | 14 µs/step | — |
| MixKVQ | `--mix-kvq` | **3.9× KV reduction** | 712 µs/KV | — |
| CocktailKV | `--cocktail-kv` | **3.0× KV reduction** | 895 µs/512-tok | — |
| AgileIO | `--agile-io` | ≈0 | 3.5 µs warm | 40–60% I/O latency ↓ |
| MiLo INT3 | `--milo` | **5.3× weight compression** | one-time convert | — |
| SageAttn | `--sage-attention` | — | — | **2.1×** attn |
| SpargeAttn | `--sparge-attn` | — | — | **2.5–5×** attn |

## Accuracy — Wave 12 (all models, Qwen2.5-1.5B representative)

| Task | Squish v1 | + Wave 12 | Delta |
|------|----------:|----------:|------:|
| ARC-Easy (acc_norm) | 73.5% | 73.5% | ±0% |
| HellaSwag (acc_norm) | 62.0% | 62.0% | ±0% |
| PIQA (acc_norm) | 76.5% | 76.5% | ±0% |
| WinoGrande (acc) | 67.0% | 67.0% | ±0% |

> Wave 12 does not alter base-model weights. KV quantisation modules
> introduce ≤0.5% accuracy delta at standard context lengths.

## Notes

- **Squish load** uses Tier 1 (safetensors) for 1.5B, Tier 0 (4-bit MLX) for larger models
- **Wave 12 KV reduction** applies during generation (not prefill-only)
- **Tok/s** measured on Apple M-series 16 GB unified memory
- lm-eval harness: EleutherAI lm-evaluation-harness v0.4.x
- Wave 12 micro-benchmarks run via `dev/benchmarks/bench_wave12.py`
- Full raw data: `dev/results/wave12_bench.json`

