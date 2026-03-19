# INT Quantization Benchmark Plan

Benchmark plan for evaluating Squish's INT4, INT3, and INT2 quantization
across 10 production LLMs, measuring throughput, perplexity, accuracy,
and storage requirements at each precision level.

---

## Overview

| Item | Value |
|------|-------|
| Bit levels | BF16 (reference), INT4, INT3, INT2 |
| Models | 10 (4 existing benchmarked + 6 new) |
| Tests per model per bit level | 3 (T1: throughput, T2: perplexity, T3: accuracy) |
| Total test runs | 10 × 3 bit levels × 3 tests = **90 runs** |
| Platform | Apple Silicon (M-series), primary |
| Scripts | `dev/benchmarks/bench_int_quant.py` (per-model) |
| Aggregation | `dev/benchmarks/aggregate_int_quant.py` (combined report) |
| Shell orchestration | `dev/scripts/run_all_int_quant.sh` |
| Results | `dev/results/int_quant/*.json` |
| Output doc | `docs/benchmark_int_quant.md` |

---

## Model Selection

### 10 Target Models

| # | Model | HF Repo | Params | BF16 Size |
|---|-------|---------|--------|-----------|
| 1 | Qwen2.5-1.5B-Instruct | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | ~3.1 GB |
| 2 | Llama-3.2-3B-Instruct | meta-llama/Llama-3.2-3B-Instruct | 3.2B | ~6.4 GB |
| 3 | Gemma-3-4B-IT | google/gemma-3-4b-it | 4B | ~8.6 GB |
| 4 | Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | 7.6B | ~14.0 GB |
| 5 | Mistral-7B-Instruct-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | 7.2B | ~14.5 GB |
| 6 | DeepSeek-R1-Distill-Qwen-7B | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7.6B | ~15.2 GB |
| 7 | Llama-3.1-8B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct | 8.0B | ~16.0 GB |
| 8 | Qwen3-8B | Qwen/Qwen3-8B | 8.2B | ~16.4 GB |
| 9 | Phi-4 | microsoft/phi-4 | 14.7B | ~29.4 GB |
| 10 | Qwen2.5-14B-Instruct | Qwen/Qwen2.5-14B-Instruct | 14.8B | ~29.6 GB |

**Previously benchmarked (existing data):** Models 1, 4, 8, 10  
**New models to benchmark:** Models 2, 3, 5, 6, 7, 9

### Selection Rationale

- **Architecture variety:** Qwen2.5 (MLA/GQA), Llama-3.x (standard GQA), Mistral (sliding window), Gemma-3 (multi-query), Phi-4 (mixture of experts), DeepSeek (distilled reasoning)
- **Size spread:** 1.5B → 14.8B — covers mobile-class to workstation-class
- **Popularity:** Top-tier HuggingFace download counts in each size class
- **License coverage mix:** Apache 2.0, MIT, Llama Community License, Gemma TOS

---

## Compression Methods

### T0 — BF16 Reference

No compression. Models loaded directly with `mlx_lm` or `transformers` in `bfloat16`.
Used as the baseline for all quality delta calculations.

### INT4 (Primary)

- **CLI:** `squish-convert --int4 --super-weight --int4-group-size 32`
- **Format:** 4-bit nibble packed, group-quantised with super-weight correction
- **Expected bpw:** ~4.5 bpw (4-bit weights + scale overhead)
- **Expected size reduction:** ~73% vs BF16
- **Quality signal:** Primary target — should retain >95% of BF16 accuracy

### INT3

- **API:** `MiLoQuantizer` Python class (`squish.quant.milo_quant`)
- **Config:** `MiLoConfig(group_size=128, max_rank=16)`
- **Iterates:** safetensors shards; applies per-layer mixed INT3
- **Expected bpw:** ~3.5 bpw
- **Expected size reduction:** ~78% vs BF16
- **Quality signal:** Secondary — expect mild PPL degradation vs INT4

Note: No `--int3` CLI flag exists in `convert.py`. Python API required.

### INT2

- **CLI:** `squish-convert --aqlm --aqlm-n-codebooks 2 --aqlm-codebook-size 16 --aqlm-group-size 8`
- **Format:** Additive Quantisation of Language Model Weights (AQLM)
- **Expected bpw:** ~2.0–2.5 bpw
- **Expected size reduction:** ~86% vs BF16
- **Quality signal:** Research/reference only — significant quality loss expected especially below 3B params

---

## Test Definitions

### T1 — Throughput (tok/s)

**What it measures:** Generation speed in output tokens per second.

**Method:**
1. Load compressed model via `mlx_lm.load()`
2. Run 5 standard prompts × `--runs` iterations (default 3)
3. Measure output tokens / wall-clock seconds
4. Report: mean, stddev, 95th-percentile

**Prompts used:**
```
1. "Explain the theory of relativity in simple terms."
2. "Write a Python function to compute Fibonacci numbers."
3. "What are the pros and cons of electric vehicles?"
4. "Summarise the French Revolution in 3 sentences."
5. "What is the difference between supervised and unsupervised learning?"
```

**Max new tokens per call:** 256  
**Temperature:** 0.0 (deterministic)

### T2 — Perplexity (PPL)

**What it measures:** Language modelling quality; lower = better.

**Method:**
1. Load first 512 tokens of WikiText-2 test split
2. Compute token-level negative log-likelihood with `mlx.core`
3. PPL = exp(mean NLL)

**Threshold for acceptable quality:**
- INT4 delta vs BF16: < 1.0 PPL points
- INT3 delta vs BF16: < 2.5 PPL points
- INT2 delta vs BF16: < 8.0 PPL points (informational)

### T3 — Accuracy (Arc-Easy + HellaSwag)

**What it measures:** Zero-shot multiple-choice accuracy on standard NLP benchmarks.

**Method:**
1. Use `lm_eval` harness with `HFLM` wrapper
2. Tasks: `arc_easy` + `hellaswag`
3. Sample limit: 200 examples per task
4. Report: accuracy on each task + combined average

**Note:** T3 is the most time-intensive test (~30 min per model on M2 Max).
Run separately using `--eval-acc` flag. Omit from initial throughput-only runs.

---

## Disk Space Requirements

### Per-Model Storage Estimate

| Model | BF16 | INT4 | INT3 | INT2 | Total (all) |
|-------|------|------|------|------|------------|
| Qwen2.5-1.5B | 3.1 GB | 0.9 GB | 0.7 GB | 0.4 GB | 5.1 GB |
| Llama-3.2-3B | 6.4 GB | 1.8 GB | 1.4 GB | 0.9 GB | 10.5 GB |
| Gemma-3-4B | 8.6 GB | 2.4 GB | 1.9 GB | 1.2 GB | 14.1 GB |
| Qwen2.5-7B | 14.0 GB | 3.9 GB | 3.1 GB | 2.0 GB | 23.0 GB |
| Mistral-7B-v0.3 | 14.5 GB | 4.0 GB | 3.2 GB | 2.0 GB | 23.7 GB |
| DeepSeek-R1-Distill-7B | 15.2 GB | 4.2 GB | 3.3 GB | 2.1 GB | 24.8 GB |
| Llama-3.1-8B | 16.0 GB | 4.4 GB | 3.5 GB | 2.2 GB | 26.1 GB |
| Qwen3-8B | 16.4 GB | 4.6 GB | 3.6 GB | 2.3 GB | 26.9 GB |
| Phi-4 | 29.4 GB | 8.2 GB | 6.5 GB | 4.1 GB | 48.2 GB |
| Qwen2.5-14B | 29.6 GB | 8.2 GB | 6.5 GB | 4.1 GB | 48.4 GB |
| **Total** | **153 GB** | **42.6 GB** | **33.7 GB** | **21.3 GB** | **250.6 GB** |

**Plan A — Full benchmark (all bits, keep all):** ~251 GB free required  
**Plan B — Rolling benchmark (delete BF16 after compress):** ~100 GB free  
**Plan C — INT4 only, BF16 kept:** ~196 GB free  
**Plan D — INT4 only, delete BF16 after compress:** ~43 GB free (minimum)

Recommended: Plan B for initial run. Use `--keep-compressed` flag to retain quantized weights.

---

## Execution Plan

### Stage 1 — Validate pipeline (1 model, INT4 only)

```bash
./dev/scripts/run_all_int_quant.sh \
    --models "Qwen2.5-1.5B" \
    --bits 4 \
    --eval-tps --eval-ppl \
    --runs 1
```

Expected runtime: ~15 min (download 3.1 GB + compress + 2 tests)

### Stage 2 — INT4 throughput sweep (all 10 models, tok/s only)

```bash
./dev/scripts/run_all_int_quant.sh \
    --bits 4 \
    --eval-tps \
    --runs 3
```

Expected runtime: ~4 hours

### Stage 3 — INT4 full quality (all 10, all 3 tests)

```bash
./dev/scripts/run_all_int_quant.sh \
    --bits 4 \
    --eval-tps --eval-ppl --eval-acc \
    --runs 3
```

Expected runtime: ~8 hours (PPL + accuracy are slow)

### Stage 4 — INT3 sweep

```bash
./dev/scripts/run_all_int_quant.sh \
    --bits 3 \
    --eval-tps --eval-ppl \
    --runs 3
```

Expected runtime: ~5 hours (MiLo compression is CPU-bound, slower)

### Stage 5 — INT2 sweep

```bash
./dev/scripts/run_all_int_quant.sh \
    --bits 2 \
    --eval-tps --eval-ppl \
    --runs 2
```

Expected runtime: ~5 hours (AQLM compression is very slow for large models)

### Stage 6 — Generate combined report

```bash
python3 dev/benchmarks/aggregate_int_quant.py \
    --results-dir dev/results/int_quant \
    --output docs/benchmark_int_quant.md \
    --json-output dev/results/int_quant/combined.json
```

---

## Output Format

### Per-run JSON (`dev/results/int_quant/<model>_<N>bit.json`)

```json
{
  "model_id": "Qwen2.5-7B-Instruct",
  "bits": 4,
  "timestamp": "2025-03-18T14:22:00",
  "compression": {
    "original_gb": 14.0,
    "compressed_gb": 3.9,
    "ratio": 0.279,
    "time_sec": 312.4
  },
  "throughput": {
    "mean_tps": 47.3,
    "std_tps": 1.2,
    "p95_tps": 45.1,
    "runs": 3
  },
  "perplexity": {
    "ppl": 6.84,
    "bf16_ppl": 6.21,
    "delta": 0.63
  },
  "accuracy": {
    "arc_easy": 0.712,
    "hellaswag": 0.613,
    "combined": 0.663,
    "bf16_combined": 0.701,
    "delta": -0.038
  }
}
```

### Combined markdown table (`docs/benchmark_int_quant.md`)

```
| Model | BF16 | INT4 | INT4↓% | INT3 | INT3↓% | INT2 | INT2↓% |
| Model | BF16 GB | INT4 GB | INT3 GB | INT2 GB |
| Model | BF16 tok/s | INT4 tok/s | INT4 speedup | INT3 tok/s | INT2 tok/s |
| Model | BF16 PPL | INT4 PPL | INT3 PPL | INT2 PPL |
| Model | BF16 ARC | INT4 ARC | INT3 ARC | INT2 ARC |
```

---

## Success Criteria

| Metric | INT4 target | INT3 target | INT2 target |
|--------|------------|------------|------------|
| Size reduction | ≥ 70% | ≥ 75% | ≥ 83% |
| PPL delta vs BF16 | < 1.0 | < 2.5 | < 8.0 |
| ARC accuracy delta | < -3% | < -6% | informational |
| Throughput vs BF16 | ≥ 1.5× faster | ≥ 1.3× faster | informational |
| Models meeting ALL criteria | ≥ 8 / 10 | ≥ 7 / 10 | — |

---

## Infrastructure

All benchmark scripts are in `/dev/benchmarks/`:

| Script | Purpose |
|--------|---------|
| `bench_int_quant.py` | Per-model benchmark runner |
| `aggregate_int_quant.py` | Combines JSONs → markdown tables |

Shell script in `/dev/scripts/`:

| Script | Purpose |
|--------|---------|
| `run_all_int_quant.sh` | Download + run all 10 models, supports --bits, --eval-tps/ppl/acc |

Results directory: `dev/results/int_quant/`  
Published doc: `docs/benchmark_int_quant.md`

---

*Last updated: 2025 — initial INT quantization benchmark planning.*
