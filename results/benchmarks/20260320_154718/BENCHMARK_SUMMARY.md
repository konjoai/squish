# Squish — Full Model Benchmark Results

Generated: 2026-03-20 15:47:18 by `scripts/run_all_benchmarks.sh`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API

| Model | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|-------------:|----------:|--------|
| `Qwen3-0.6B-bf16` | 4140 | 61.3 | OK |
| `Llama-3.2-1B-Instruct-bf16` | 5900 | 30.3 | OK |
| `gemma-3-1b-it-bf16` | ? | ? | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | ? | ? | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-awq` | n/a | n/a | FAIL (startup timeout) |
