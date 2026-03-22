# Squish — Full Model Benchmark Results

Generated: 2026-03-21 11:40:06 by `scripts/run_all_benchmarks.sh`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API  
Server flags: squish default (all optimizations) / `--stock` (no optimizations, Ollama comparable)

| Model | Tier | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|------|-------------:|----------:|--------|
| `Qwen3-0.6B-bf16` | squish | 261 | 50.1 | OK |
| `Llama-3.2-1B-Instruct-bf16` | squish | 391 | 21.4 | OK |
| `gemma-3-1b-it-bf16` | squish | 497 | 25.5 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | squish | 354 | 19.8 | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-awq` | squish | 546 | 17.7 | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-mse` | squish | 289 | 24.5 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed` | squish | 314 | 22.9 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v2` | squish | 367 | 20.2 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v3` | squish | 648 | 16.2 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16attn-noawq` | squish | 398 | 18.8 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16embed` | squish | 600 | 17.6 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16mlp` | squish | 383 | 19.3 | OK |
| `Qwen2.5-1.5B-Instruct-squished-g8-mixed` | squish | 525 | 17.2 | OK |
| `Qwen2.5-1.5B-Instruct-squished-lossless` | squish | 590 | 17.2 | OK |
| `Qwen2.5-1.5B-Instruct-bf16-compressed` | squish | 595 | 17.7 | OK |
| `Llama-3.2-3B-Instruct-bf16` | squish | n/a | n/a | FAIL (bench error) |
| `Qwen3-4B-bf16` | squish | n/a | n/a | FAIL (bench error) |
