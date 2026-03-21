# Squish — Full Model Benchmark Results

Generated: 2026-03-21 12:02:55 by `scripts/run_all_benchmarks.sh`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API  
Server flags: squish default (all optimizations) / `--stock` (no optimizations, Ollama comparable)

| Model | Tier | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|------|-------------:|----------:|--------|
| `Qwen3-0.6B-bf16` | squish | 182 | 61.3 | OK |
| `Llama-3.2-1B-Instruct-bf16` | squish | 246 | 31.2 | OK |
| `gemma-3-1b-it-bf16` | squish | 436 | 35.0 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | squish | 268 | 24.2 | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-awq` | squish | 272 | 24.0 | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-mse` | squish | 336 | 23.9 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed` | squish | 269 | 24.5 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v2` | squish | 278 | 24.3 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v3` | squish | 270 | 24.0 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16attn-noawq` | squish | 282 | 24.2 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16embed` | squish | 284 | 24.3 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16mlp` | squish | 280 | 23.8 | OK |
| `Qwen2.5-1.5B-Instruct-squished-g8-mixed` | squish | 422 | 23.3 | OK |
| `Qwen2.5-1.5B-Instruct-squished-lossless` | squish | 282 | 24.6 | OK |
| `Qwen2.5-1.5B-Instruct-bf16-compressed` | squish | 332 | 22.0 | OK |
| `Llama-3.2-3B-Instruct-bf16` | squish | 629 | 11.6 | OK |
| `Qwen3-4B-bf16` | squish | 728 | 9.9 | OK |
| `gemma-3-4b-it-bf16` | squish | 1116 | 9.1 | OK |
| `Qwen2.5-7B-Instruct-bf16` | squish | n/a | n/a | FAIL (OOM/crash) |
| `Qwen3-8B-bf16-compressed` | squish | 443 | 17.9 | OK |
| `Qwen3-8B-bf16` | squish | 535 | 13.8 | OK |

---
**Run completed**: 2026-03-21 12:20:20  
**Passed**: 20 / 21  
**Results dir**: `/Users/wscholl/squish/results/benchmarks/20260321_120255`

Individual markdown tables saved as `<model>_<tier>.md` in the results directory.
