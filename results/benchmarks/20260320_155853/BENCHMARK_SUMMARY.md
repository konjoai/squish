# Squish — Full Model Benchmark Results

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API

| Model | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|-------------:|----------:|--------|
| `Llama-3.2-1B-Instruct-bf16` | 6257 | 30.9 | OK |
| `Llama-3.2-3B-Instruct-bf16` | 23234 | 8.5 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | 7894 | 14.6 | OK |
| `Qwen3-0.6B-bf16` | 4093 | 61.4 | OK |
| `Qwen3-4B-bf16` | 34495 | 7.4 | OK |
| `Qwen3-8B-bf16` | 28520 | 9.0 | OK |
| `gemma-3-1b-it-bf16` | 9747 | 25.4 | OK |
| `gemma-3-4b-it-bf16` | 29075 | 6.5 | OK |

---
*Regenerated from individual per-model files in `/Users/wscholl/squish/results/benchmarks/20260320_155853`.*
