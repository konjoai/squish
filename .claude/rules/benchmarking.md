---
paths: ["**/benchmarks/**", "**/bench_*.py", "**/perf/**"]
---
# Benchmarking Rules
- Minimum 5 warmup runs. Report p50/p95/p99/stddev — not just mean.
- Document hardware: chip, RAM, OS, MLX version, PyTorch version.
- Quantization accuracy gates are hard stops: INT4 AWQ g=32 >= 70.6% arc_easy (Qwen2.5-1.5B).
- Results in `benchmarks/results/<timestamp>/`. Never overwrite.
- Regression gate: >5% p95 latency = hard stop.
