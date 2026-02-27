# Squish PoC — Benchmark Results

**Model**: Qwen2.5-7B-bs1  
**Evaluation**: EleutherAI lm-evaluation-harness (industry standard)  
**Limit**: 200 examples per task (representative sample)  

## Accuracy — Reference vs Compressed

| Task | Reference | Compressed | Δ | Status |
|------|----------:|-----------:|--:|--------|
| ARC-Easy acc_norm | — | 75.0% | — | — |
| HellaSwag acc_norm | — | 69.5% | — | — |

## Load Time

| Strategy | Load time |
|----------|----------:|
| Compressed (finalized⚡) | 4.41s |

## Methodology

Evaluation uses [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
— the same framework used to evaluate every model on the Open LLM Leaderboard.

The compressed model loads weights from the Squish compressed cache WITHOUT
the original `.safetensors` — demonstrating full independence from the
original weight format. Large models use 4-bit MLX cache (squish_4bit);
small models use INT8 Vectro npy-dir + MLX safetensors cache.

Tasks:
- **ARC-Easy acc_norm** (`arc_easy`)
- **HellaSwag acc_norm** (`hellaswag`)
