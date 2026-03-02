# Squish — 7B Benchmark Results

**Model**: Qwen2.5-7B-Instruct  
**Evaluation**: EleutherAI lm-evaluation-harness v0.4.11 (industry standard)  
**Limit**: 1000 examples per task  
**Hardware**: Apple M3, 16 GB unified memory, macOS  

---

## Reference Model: CANNOT RUN on 16 GB hardware

The Qwen2.5-7B-Instruct BF16 model (~14 GB) **exceeds available system memory**
on 16 GB hardware. Loading the model triggers severe swap thrashing:

| Metric | Measured value |
|--------|---------------|
| Elapsed time | 43 min 43 sec |
| Actual CPU work | 4 min 52 sec (11.1% efficiency) |
| Swap used | 9,411 MB / 10,240 MB (91.9% exhausted) |
| Swap I/O | 1.2B reads + 1.6B writes |
| Free RAM | ~70 MB |

**Inference is not practically possible** without Squish compression.

Published reference scores (Qwen2.5-7B-Instruct, Open LLM Leaderboard):

| Task | Published Reference |
|------|----------------:|
| ARC-Challenge acc_norm | ~63.1% |
| HellaSwag acc_norm | ~80.6% |
| Winogrande acc | ~74.0% |
| PIQA acc_norm | ~81.1% |

---

## Squish Compressed Model: Runs fine on 16 GB

**Load time: 2.08s** (no swap, model fits entirely in RAM at ~3.5 GB)

| Task | Published Ref | Squish 4-bit | Δ |
|------|-------------:|-------------:|--:|
| ARC-Challenge acc_norm | ~63.1% | 54.6% | -8.5% |
| HellaSwag acc_norm | ~80.6% | 68.4% | -12.2% |
| Winogrande acc | ~74.0% | 70.7% | -3.3% |
| PIQA acc_norm | ~81.1% | 80.5% | -0.6% |

---

## Key Finding

> On 16 GB hardware, the 7B BF16 reference model is **completely unusable**.
> Squish 4-bit compression reduces the model from ~14 GB to ~3.5 GB (4× reduction),
> enabling inference on hardware that cannot run the original model at all.
> Accuracy degradation ranges from -0.6% (PIQA) to -12.2% (HellaSwag) vs
> published reference scores — a reasonable tradeoff for the capability it unlocks.

---

## Note on Reference Comparison

Direct local benchmarking of the reference 7B model was not possible due to
hardware memory constraints (confirmed above). Published leaderboard scores
used as reference baseline. These scores are measured on full datasets
(not limited to 1000 samples), so the comparison is approximate.
