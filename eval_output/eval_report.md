# Squish PoC — Benchmark Results

**Model**: Qwen2.5-1.5B-Instruct  
**Evaluation**: EleutherAI lm-evaluation-harness (industry standard)  
**Limit**: 1000 examples per task (representative sample)  

## Accuracy — Reference vs Compressed

| Task | Reference | Compressed | Δ | Status |
|------|----------:|-----------:|--:|--------|
| ARC-Chall acc_norm | 46.3% | 60.4% | +14.1% | ✅ |
| HellaSwag acc_norm | 59.7% | 74.9% | +15.2% | ✅ |
| Winogrande acc | 63.0% | 75.6% | +12.6% | ✅ |
| PIQA acc_norm | 76.7% | 81.2% | +4.5% | ✅ |

## Load Time

| Strategy | Load time |
|----------|----------:|
| Compressed (finalized⚡) | 4.09s |

## Methodology

Evaluation uses [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
— the same framework used to evaluate every model on the Open LLM Leaderboard.

The compressed model loads weights from the Squish compressed cache WITHOUT
the original `.safetensors` — demonstrating full independence from the
original weight format. Large models use 4-bit MLX cache (squish_4bit);
small models use INT8 Vectro npy-dir + MLX safetensors cache.

Tasks:
- **ARC-Chall acc_norm** (`arc_challenge`)
- **HellaSwag acc_norm** (`hellaswag`)
- **Winogrande acc** (`winogrande`)
- **PIQA acc_norm** (`piqa`)
