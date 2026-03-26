# Squish — Full Model Benchmark Results

Generated: 2026-03-26 12:10:07 by `scripts/run_all_benchmarks.sh`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API  
Tiers: squish=default (auto-profile/blazing) · maxopt=--all-optimizations · stock=--stock (Ollama-comparable)

| Model | Tier | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|------|-------------:|----------:|--------|
| `Qwen3-0.6B-bf16` | squish | 561 | 26.0 | OK |
| `Qwen3-0.6B-bf16` | maxopt | 1498 | 26.6 | OK |
| `Llama-3.2-1B-Instruct-bf16` | squish | 754 | 17.1 | OK |
| `Llama-3.2-1B-Instruct-bf16` | maxopt | 2994 | 11.8 | OK |
| `gemma-3-1b-it-bf16` | squish | 555 | 22.9 | OK |
| `gemma-3-1b-it-bf16` | maxopt | 2886 | 17.2 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | squish | 621 | 10.3 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | maxopt | 3122 | 9.4 | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-awq` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-int4-mse` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-mixed` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v2` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v3` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-fp16attn-noawq` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-fp16embed` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-fp16mlp` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-g8-mixed` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-squished-lossless` | — | n/a | n/a | SKIP (not found) |
| `Qwen2.5-1.5B-Instruct-bf16-compressed` | — | n/a | n/a | SKIP (not found) |
| `Llama-3.2-1B-Instruct-int3` | squish | n/a | n/a | FAIL (startup) |
| `Llama-3.2-1B-Instruct-int3` | maxopt | n/a | n/a | FAIL (startup) |
| `Llama-3.2-3B-Instruct-bf16` | squish | 1212 | 6.3 | OK |
| `Llama-3.2-3B-Instruct-bf16` | maxopt | 8421 | 5.1 | OK |
