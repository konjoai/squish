# Squish — Full Model Benchmark Results

Generated: 2026-03-21 11:15:04 by `scripts/run_all_benchmarks.sh`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API  
Server flags: squish default (all optimizations) / `--stock` (no optimizations, Ollama comparable)

| Model | Tier | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|------|-------------:|----------:|--------|
| `Qwen3-0.6B-bf16` | squish | 350 | 17.5 | OK |
| `Llama-3.2-1B-Instruct-bf16` | squish | 969 | 4.5 | OK |
| `gemma-3-1b-it-bf16` | squish | 693 | 4.3 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | squish | n/a | n/a | FAIL (bench error) |
