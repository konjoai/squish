# Squish — Full Model Benchmark Results

Generated: 2026-03-26 11:06:15 by `scripts/run_all_benchmarks.sh`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API  
Server flags: squish default (all optimizations) / `--stock` (no optimizations, Ollama comparable)

| Model | Tier | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|------|-------------:|----------:|--------|
| `Qwen3-0.6B-bf16` | squish | 521 | 29.3 | OK |
| `Llama-3.2-1B-Instruct-bf16` | squish | 490 | 17.4 | OK |
| `gemma-3-1b-it-bf16` | squish | 940 | 18.3 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | squish | 584 | 11.4 | OK |
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
| `Llama-3.2-3B-Instruct-bf16` | squish | n/a | n/a | FAIL (startup) |
| `Qwen3-4B-bf16` | squish | n/a | n/a | FAIL (startup) |
| `gemma-3-4b-it-bf16` | squish | n/a | n/a | FAIL (startup) |
| `Qwen2.5-7B-Instruct-bf16` | squish | n/a | n/a | FAIL (startup) |
| `Qwen3-8B-bf16-compressed` | — | n/a | n/a | SKIP (not found) |
| `Qwen3-8B-bf16` | squish | n/a | n/a | FAIL (startup) |

---
**Run completed**: 2026-03-26 11:22:27  
**Passed**: 4 / 9  
**Results dir**: `/Users/wscholl/squish/results/benchmarks/20260326_110615`

Individual markdown tables saved as `<model>_<tier>.md` in the results directory.
