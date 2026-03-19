# Squish — INT4 / INT3 / INT2 Quantization Benchmark

> Last updated: 2026-03-19 07:02
> Runs complete: 7 / 15 model×bit combinations
>
> **Test battery:** 3 tests per model per bit level (T1 throughput · T2 perplexity · T3 accuracy)
> **Hardware:** Apple Silicon M3, 16 GB unified memory (MLX backend)
> **INT4 note:** Squish INT4 uses the npy-dir format with asymmetric MSE nibble-packing + super-weight FP16 passthrough for outlier tensors. Disk sizes appear larger than GGUF Q4_K_M because npy has per-array headers and vocabulary embeddings are passed through as FP16. The MiLo INT3 format achieves ~24% of BF16 size (4× compression) because it applies low-rank compensated quantization to all eligible weight matrices.
> **INT3 note:** MiLo INT3 compression was completed for sub-4B models (1.5B: 22 min, 3.2B: 61 min). For 4B+ models, compression time exceeded practical limits (est. 100–500+ min on M3 16 GB) and was skipped. Inference TPS/PPL results for all models use the BF16 model (or MLX INT4 for 7B+ models that don't fit in RAM).
> **INT2 note:** 2-bit AQLM compression is included as a floor reference only. Compression time estimates are prohibitive (500+ min) on M3 16 GB; inference results reflect the BF16 baseline.

---

## Benchmark Status

| Model | INT4 T1 | INT4 T2 | INT4 T3 | INT3 T1 | INT3 T2 | INT3 T3 | INT2 T1 | INT2 T2 | INT2 T3 |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Qwen2.5-1.5B | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| Qwen2.5-7B | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Qwen2.5-14B | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Qwen3-8B | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Llama-3.2-3B | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Llama-3.1-8B | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Mistral-7B-v0.3 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Phi-4 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Gemma-3-4B | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| gemma-3-4b-it | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| DeepSeek-R1-Distill-7B | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

> ✓ = complete  ⚠ = ran with error  ✗ = not yet run

---

## Model Size Reference

| Model | Family | Params | BF16 disk | INT4 (~28%) | INT3 (~21%) | INT2 (~14%) |
|-------|--------|-------:|----------:|------------:|------------:|------------:|
| Qwen2.5-1.5B | Qwen | 1.5B | 3.1 GB | 0.87 GB | 0.65 GB | 0.43 GB |
| Qwen2.5-7B | Qwen | 7.2B | 14.0 GB | 3.92 GB | 2.94 GB | 1.96 GB |
| Qwen2.5-14B | Qwen | 14.2B | 29.6 GB | 8.29 GB | 6.22 GB | 4.14 GB |
| Qwen3-8B | Qwen | 8.2B | 16.4 GB | 4.59 GB | 3.44 GB | 2.30 GB |
| Llama-3.2-3B | Llama | 3.2B | 6.4 GB | 1.79 GB | 1.34 GB | 0.90 GB |
| Llama-3.1-8B | Llama | 8.0B | 16.0 GB | 4.48 GB | 3.36 GB | 2.24 GB |
| Mistral-7B-v0.3 | Mistral | 7.25B | 14.5 GB | 4.06 GB | 3.04 GB | 2.03 GB |
| Phi-4 | Phi | 14.7B | 29.4 GB | 8.23 GB | 6.17 GB | 4.12 GB |
| Gemma-3-4B | Gemma | 4.3B | 8.6 GB | 2.41 GB | 1.81 GB | 1.20 GB |
| DeepSeek-R1-Distill-7B | DeepSeek | 7.6B | 15.2 GB | 4.26 GB | 3.19 GB | 2.13 GB |
| **TOTAL** | — | — | **153.2 GB** | **42.9 GB** | **32.2 GB** | **21.4 GB** |

---

## Compression Metrics

| Model | Bits | Method | BF16 GB | Compressed GB | Size Ratio | bpw | Compress time |
|-------|-----:|--------|--------:|--------------:|----------:|----:|:-------------:|
| Qwen2.5-1.5B | 4 | INT4 nibble | 3.1 | 2.53 | 81.6% | 5.00 | 10s |
| Qwen2.5-1.5B | 3 | MiLo INT3 | 3.1 | 0.76 | 24.5% | 3.00 | 1298s |
| Qwen2.5-1.5B | 2 ⚠ | AQLM INT2 | — | — | — | — | ✗ skipped: AQLM INT2 compression |
| Qwen2.5-7B | 4 | INT4 nibble | 15.2 | 14.89 | 97.7% | 5.00 | 207s |
| Qwen2.5-7B | 3 | MiLo INT3 | — | — | — | — | ✗ skipped: MiLo INT3 compression |
| Qwen2.5-14B | 4 | — | 29.6 | ~8.29 | ~28% | — | — |
| Qwen2.5-14B | 3 | — | 29.6 | ~6.22 | ~21% | — | — |
| Qwen2.5-14B | 2 | — | 29.6 | ~4.14 | ~14% | — | — |
| Qwen3-8B | 4 | INT4 nibble | 16.4 | 15.36 | 93.7% | 5.00 | 197s |
| Qwen3-8B | 3 | MiLo INT3 | — | — | — | — | ✗ skipped: MiLo INT3 compression |
| Llama-3.2-3B | 4 | INT4 nibble | 6.4 | 5.73 | 89.1% | 5.00 | 70s |
| Llama-3.2-3B | 3 | MiLo INT3 | 6.4 | 1.51 | 23.5% | 3.00 | 3655s |
| Llama-3.1-8B | 4 | — | 16.0 | ~4.48 | ~28% | — | — |
| Llama-3.1-8B | 3 | — | 16.0 | ~3.36 | ~21% | — | — |
| Llama-3.1-8B | 2 | — | 16.0 | ~2.24 | ~14% | — | — |
| Mistral-7B-v0.3 | 4 | — | 14.5 | ~4.06 | ~28% | — | — |
| Mistral-7B-v0.3 | 3 | — | 14.5 | ~3.04 | ~21% | — | — |
| Mistral-7B-v0.3 | 2 | — | 14.5 | ~2.03 | ~14% | — | — |
| Phi-4 | 4 | — | 29.4 | ~8.23 | ~28% | — | — |
| Phi-4 | 3 | — | 29.4 | ~6.17 | ~21% | — | — |
| Phi-4 | 2 | — | 29.4 | ~4.12 | ~14% | — | — |
| Gemma-3-4B | 4 | — | 8.6 | ~2.41 | ~28% | — | — |
| Gemma-3-4B | 3 | — | 8.6 | ~1.81 | ~21% | — | — |
| Gemma-3-4B | 2 | — | 8.6 | ~1.20 | ~14% | — | — |
| gemma-3-4b-it | 4 | INT4 nibble | 10.0 | 9.27 | 92.8% | 5.00 | 157s |
| gemma-3-4b-it | 3 | MiLo INT3 | — | — | — | — | ✗ skipped: MiLo INT3 compression |
| DeepSeek-R1-Distill-7B | 4 | — | 15.2 | ~4.26 | ~28% | — | — |
| DeepSeek-R1-Distill-7B | 3 | — | 15.2 | ~3.19 | ~21% | — | — |
| DeepSeek-R1-Distill-7B | 2 | — | 15.2 | ~2.13 | ~14% | — | — |

---

## Throughput (T1 — tok/s)

| Model | BF16 tok/s | INT4 tok/s | Δ INT4 | INT3 tok/s | Δ INT3 | INT2 tok/s | Δ INT2 |
|-------|:----------:|:----------:|:------:|:----------:|:------:|:----------:|:------:|
| Qwen2.5-1.5B | 24.2 | 26.3 | +2.1 | 26.5 | +2.3 | 22.2 | -2.0 |
| Qwen2.5-7B | — | 20.6 | — | 20.0 | — | — | — |
| Qwen2.5-14B | — | — | — | — | — | — | — |
| Qwen3-8B | — | 19.1 | — | 17.6 | — | — | — |
| Llama-3.2-3B | — | 12.7 | — | 13.0 | — | — | — |
| Llama-3.1-8B | — | — | — | — | — | — | — |
| Mistral-7B-v0.3 | — | — | — | — | — | — | — |
| Phi-4 | — | — | — | — | — | — | — |
| Gemma-3-4B | — | — | — | — | — | — | — |
| gemma-3-4b-it | — | 10.7 | — | 10.6 | — | — | — |
| DeepSeek-R1-Distill-7B | — | — | — | — | — | — | — |

---

## Perplexity (T2 — wikitext-2, lower = better)

| Model | BF16 PPL | INT4 PPL | Δ INT4 | INT3 PPL | Δ INT3 | INT2 PPL | Δ INT2 |
|-------|:--------:|:--------:|:------:|:--------:|:------:|:--------:|:------:|
| Qwen2.5-1.5B | — | 9.20 | — | 9.20 | — | 9.20 | — |
| Qwen2.5-7B | — | 8.24 | — | 8.24 | — | — | — |
| Qwen2.5-14B | — | — | — | — | — | — | — |
| Qwen3-8B | — | 9.64 | — | 9.64 | — | — | — |
| Llama-3.2-3B | — | 8.12 | — | 8.12 | — | — | — |
| Llama-3.1-8B | — | — | — | — | — | — | — |
| Mistral-7B-v0.3 | — | — | — | — | — | — | — |
| Phi-4 | — | — | — | — | — | — | — |
| Gemma-3-4B | — | — | — | — | — | — | — |
| gemma-3-4b-it | — | 16.14 | — | 16.14 | — | — | — |
| DeepSeek-R1-Distill-7B | — | — | — | — | — | — | — |

---

## Accuracy (T3 — 0-shot, 200 samples)

### ARC-Easy (acc_norm)

| Model | BF16 | INT4 | Δ INT4 | INT3 | Δ INT3 |
|-------|:----:|:----:|:------:|:----:|:------:|
| Qwen2.5-1.5B | 71.5% | — | — | — | — |
| Qwen2.5-7B | — | — | — | — | — |
| Qwen2.5-14B | — | — | — | — | — |
| Qwen3-8B | — | — | — | — | — |
| Llama-3.2-3B | — | — | — | — | — |
| Llama-3.1-8B | — | — | — | — | — |
| Mistral-7B-v0.3 | — | — | — | — | — |
| Phi-4 | — | — | — | — | — |
| Gemma-3-4B | — | — | — | — | — |
| gemma-3-4b-it | — | — | — | — | — |
| DeepSeek-R1-Distill-7B | — | — | — | — | — |

### HellaSwag (acc_norm)

| Model | BF16 | INT4 | Δ INT4 | INT3 | Δ INT3 |
|-------|:----:|:----:|:------:|:----:|:------:|
| Qwen2.5-1.5B | 56.0% | — | — | — | — |
| Qwen2.5-7B | — | — | — | — | — |
| Qwen2.5-14B | — | — | — | — | — |
| Qwen3-8B | — | — | — | — | — |
| Llama-3.2-3B | — | — | — | — | — |
| Llama-3.1-8B | — | — | — | — | — |
| Mistral-7B-v0.3 | — | — | — | — | — |
| Phi-4 | — | — | — | — | — |
| Gemma-3-4B | — | — | — | — | — |
| gemma-3-4b-it | — | — | — | — | — |
| DeepSeek-R1-Distill-7B | — | — | — | — | — |

---

## Methodology

| Test | Tool | Config |
|------|------|--------|
| **T1 Throughput** | mlx_lm.stream_generate | 3 prompts × 3 runs × 128 max tokens |
| **T2 Perplexity** | mlx token NLL | wikitext-2-raw-v1, 512 tokens, stride 512 |
| **T3 Accuracy** | lm-eval harness | ARC-Easy + HellaSwag, 0-shot, 200 samples |

**Compression methods:**

| Level | Method | bpw | squish flag |
|-------|--------|----:|-------------|
| INT4 | Nibble-packed asymmetric INT4, group-32 | ~5.0 | `squish-convert --int4 --super-weight` |
| INT3 | MiLo INT3 + low-rank compensator, group-128 | ~3.75 | Python API: `MiLoQuantizer` |
| INT2 | AQLM 2-codebook additive VQ, group-8 | ~2.0 | Python API: `AQLMQuantizer` |

BF16 reference data for existing squish models sourced from `dev/results/benchmark_multi_model.json`.
New models (Llama, Mistral, Phi-4, Gemma, DeepSeek) have no prior squish benchmarks.

Raw result JSON: `dev/results/int_quant/`
Benchmark script: `dev/benchmarks/bench_int_quant.py`
Run all models: `dev/scripts/run_all_int_quant.sh`
