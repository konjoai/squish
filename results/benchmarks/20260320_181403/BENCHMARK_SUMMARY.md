# Squish — Full Model Benchmark Results

Generated: 2026-03-20 18:14:03 by `scripts/run_all_benchmarks.sh`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API  
Server flags: `--no-agent` (baseline) / `--all-optimizations` (optimized)

| Model | Tier | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|------|-------------:|----------:|--------|
| `Qwen3-0.6B-bf16` | baseline | 4164 | 60.9 | OK |
| `Llama-3.2-1B-Instruct-bf16` | baseline | 6581 | 31.2 | OK |
| `gemma-3-1b-it-bf16` | baseline | 6995 | 34.4 | OK |
| `Qwen2.5-1.5B-Instruct-bf16` | baseline | 6796 | 25.0 | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-awq` | baseline | 2752 | 22.4 | OK |
| `Qwen2.5-1.5B-Instruct-squished-int4-mse` | baseline | 6147 | 25.2 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed` | baseline | 3460 | 23.6 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v2` | baseline | 3590 | 23.1 | OK |
| `Qwen2.5-1.5B-Instruct-squished-mixed-v3` | baseline | 5921 | 22.6 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16attn-noawq` | baseline | 4914 | 18.4 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16embed` | baseline | 5335 | 22.2 | OK |
| `Qwen2.5-1.5B-Instruct-squished-fp16mlp` | baseline | 3105 | 22.6 | OK |
| `Qwen2.5-1.5B-Instruct-squished-g8-mixed` | baseline | 3505 | 22.5 | OK |
| `Qwen2.5-1.5B-Instruct-squished-lossless` | baseline | 4769 | 22.3 | OK |
| `Qwen2.5-1.5B-Instruct-bf16-compressed` | baseline | 4260 | 21.7 | OK |
| `Llama-3.2-3B-Instruct-bf16` | baseline | 18991 | 10.6 | OK |
| `Qwen3-4B-bf16` | baseline | 27631 | 9.1 | OK |
| `gemma-3-4b-it-bf16` | baseline | 24201 | 7.5 | OK |
| `Qwen2.5-7B-Instruct-bf16` | baseline | n/a | n/a | CRASH (Abort trap: OOM — 14 GB model on 15.5 GB budget) |
| `Qwen3-8B-bf16-compressed` | baseline | 17391 | 14.7 | OK |
| `Qwen3-8B-bf16` | baseline | 20459 | 12.4 | OK |

---
**Run completed**: 2026-03-20 18:32:49  
**Passed**: 21 / 21  
**Results dir**: `/Users/wscholl/squish/results/benchmarks/20260320_181403`

Individual markdown tables saved as `<model>_<tier>.md` in the results directory.
