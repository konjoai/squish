# INT3 Accuracy Table — arc_easy (acc_norm)

Evaluation harness: EleutherAI `lm-evaluation-harness` v0.4.11  
Metric: `arc_easy` `acc_norm`, 0-shot, limit=500  
Hardware: Apple M3 · 16 GB RAM · macOS 15.7.4 · MLX-LM 0.30.7  
Source files: `results/lmeval_*.json`

> **Gate definition:** SAFE = delta ≤ −5 pp; BORDERLINE = −5 to −10 pp; BLOCKED = delta > −10 pp  
> Per `BENCHMARKS.md` §3 and `PLAN.md` accuracy gates.

---

## Results Table

| Model | BF16 baseline | INT3 score | Δ pp | Verdict | Source |
|---|:---:|:---:|:---:|:---:|---|
| **Qwen2.5-1.5B-Instruct** | 75.0% | **67.2%** | −7.8 pp | BORDERLINE | BF16: `dev/BENCHMARK_REFERENCE.md`; INT3: `lmeval_Qwen2.5-1.5B-int3_20260328T000807.json` |
| **Qwen2.5-7B-Instruct** | ❓ missing | **79.0%** | — | — | INT3: `lmeval_Qwen2.5-7B-int3_20260331T215251.json`; BF16 not run |
| **Qwen3-4B** | ~73.2% † | **58.4%** | −14.8 pp | **BLOCKED** | INT3: `lmeval_Qwen3-4B-int3_20260402T052412.json`; BF16 inferred from CHANGELOG |
| **Qwen3-8B** | ~79.2% † | **71.4%** | −7.8 pp | BORDERLINE | INT3: `lmeval_Qwen3-8B-int3_20260401T054532.json`; BF16 inferred from CHANGELOG |
| **Llama-3.2-1B** | 42.0% | **37.4%** | −4.6 pp | SAFE | BF16: `lmeval_comparison_20260323T110658.json`; INT3: `lmeval_Llama-3.2-1B-int3_20260328T030315.json` |
| **Llama-3.2-3B** | ❓ missing | **42.2%** | — | — | INT3: `lmeval_Llama-3.2-3B-int3_20260329T021205.json`; BF16 not run |
| **Gemma-3-1b** | ❓ missing | **38.0%** | — | — | INT3: `lmeval_gemma-3-1b-int3_20260328T045951.json`; BF16 errored |
| **Gemma-3-4b** | ~80.8% † | **64.4%** | −16.4 pp | **BLOCKED** | INT3: `lmeval_gemma-3-4b-int3_20260329T145216.json`; BF16 inferred from CHANGELOG |

† Inferred from CHANGELOG delta annotation — not a direct measurement. See notes below.

---

## What's Confirmed vs. Inferred

### Directly measured BF16 baselines
- **Qwen2.5-1.5B**: 75.0% — squish-path BF16 eval, 500 samples, documented in `dev/BENCHMARK_REFERENCE.md`
- **Llama-3.2-1B**: 42.0% — from `results/lmeval_comparison_20260323T110658.json`

### Inferred BF16 baselines (from CHANGELOG delta annotations)
CHANGELOG entry (`CHANGELOG.md` ~line 1949–1951) records the following deltas when the INT3 runs were evaluated against their respective BF16 references:
- **Qwen3-4B**: INT3 = −14.8 pp → BF16 = 58.4 + 14.8 = **73.2%** (corroborated by INT4 score of 73.2%)
- **Qwen3-8B**: INT3 = −7.8 pp → BF16 = 71.4 + 7.8 = **79.2%** (matches INT4 score of 79.2%)
- **Gemma-3-4b**: INT3 = −16.4 pp → BF16 = 64.4 + 16.4 = **80.8%** (matches INT4 score of 80.8%)

### Missing BF16 baselines (runs failed or never run)
- **Qwen2.5-7B**: `lmeval_Qwen2.5-7B-bf16_*.json` — no run found; INT3=79.0% is solid but delta unknown
- **Llama-3.2-3B**: no BF16 lmeval run found; INT3=42.2% is solid but delta unknown
- **Gemma-3-1b**: BF16 eval errored with `mlx_lm exit code 1` — all 6 tasks failed; INT3=38.0% is solid; note that INT4=53.2% and INT3=38.0% is a −15.2 pp drop from INT4, likely BLOCKED from BF16 as well

---

## Verdicts Summary

| Verdict | Models |
|---|---|
| ✅ SAFE (≤ −5 pp) | Llama-3.2-1B |
| ⚠️ BORDERLINE (−5 to −10 pp) | Qwen2.5-1.5B, Qwen3-8B |
| 🚫 BLOCKED (> −10 pp) | Qwen3-4B, Gemma-3-4b |
| ❓ Incomplete (no BF16) | Qwen2.5-7B, Llama-3.2-3B, Gemma-3-1b |

Qwen2.5-1.5B INT3 has a **shipping gate** of ≥ 67.2% which it meets exactly — this model is BORDERLINE but gate-passing per `PLAN.md`.

---

## Commands to Fill the Gaps

The following BF16 evaluations have not been run. Each command takes ~30–60 min on M3 16 GB.

```bash
# Qwen2.5-7B BF16 baseline
lm_eval --model squish \
    --model_args "model_dir=~/models/Qwen2.5-7B-Instruct-bf16,compressed_dir=~/models/Qwen2.5-7B-Instruct-bf16" \
    --tasks arc_easy \
    --num_fewshot 0 \
    --limit 500 \
    --output_path results/lmeval_Qwen2.5-7B-bf16_$(date +%Y%m%dT%H%M%S).json

# Llama-3.2-3B BF16 baseline
lm_eval --model squish \
    --model_args "model_dir=~/models/Llama-3.2-3B-Instruct-bf16,compressed_dir=~/models/Llama-3.2-3B-Instruct-bf16" \
    --tasks arc_easy \
    --num_fewshot 0 \
    --limit 500 \
    --output_path results/lmeval_Llama-3.2-3B-bf16_$(date +%Y%m%dT%H%M%S).json

# Gemma-3-1b BF16 baseline (prior run errored — retry with explicit backend)
lm_eval --model squish \
    --model_args "model_dir=~/models/gemma-3-1b-it-bf16,compressed_dir=~/models/gemma-3-1b-it-bf16" \
    --tasks arc_easy \
    --num_fewshot 0 \
    --limit 500 \
    --output_path results/lmeval_gemma-3-1b-bf16_$(date +%Y%m%dT%H%M%S).json
```

---

## All INT3 Run Files Referenced

| Model | File | arc_easy |
|---|---|:---:|
| Qwen2.5-1.5B-int3 | `lmeval_Qwen2.5-1.5B-int3_20260328T000807.json` | 67.2% |
| Qwen2.5-7B-int3 | `lmeval_Qwen2.5-7B-int3_20260331T215251.json` | 79.0% |
| Qwen3-4B-int3 | `lmeval_Qwen3-4B-int3_20260402T052412.json` | 58.4% |
| Qwen3-8B-int3 | `lmeval_Qwen3-8B-int3_20260401T054532.json` | 71.4% |
| Llama-3.2-1B-int3 | `lmeval_Llama-3.2-1B-int3_20260328T030315.json` | 37.4% |
| Llama-3.2-3B-int3 | `lmeval_Llama-3.2-3B-int3_20260329T021205.json` | 42.2% |
| Gemma-3-1b-int3 | `lmeval_gemma-3-1b-int3_20260328T045951.json` | 38.0% |
| Gemma-3-4b-int3 | `lmeval_gemma-3-4b-int3_20260329T145216.json` | 64.4% |
