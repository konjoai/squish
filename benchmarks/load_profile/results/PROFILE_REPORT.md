# Cold-load profile report  (20260601_124014)

Host: Apple M3 MacBook Pro 16 GB · macOS 25.5.0 · Python 3.14.3 · MLX-LM 0.31.1
Model: `Qwen2.5-7B-Instruct-int4` (INT4 MLX safetensors)

Times are cumulative seconds from process start (t0).
`Δ` is the per-phase delta (this phase only).

| Phase             | Cold cum (s) | Cold Δ (s) | Warm cum (s) | Warm Δ (s) |
|-------------------|-------------:|-----------:|-------------:|-----------:|
| squish_imports    |        0.473 |      0.473 |        0.450 |      0.450 |
| mlx_core_import   |        0.473 |      0.000 |        0.450 |      0.000 |
| mlx_lm_import     |        3.083 |      2.610 |        2.985 |      2.535 |
| mlx_utils_import  |        3.083 |      0.000 |        2.985 |      0.000 |
| weights_loaded    |        4.468 |      1.385 |        4.314 |      1.329 |
| tokenizer_loaded  |        5.093 |      0.624 |        4.982 |      0.668 |
| warmup_done       |        6.205 |      1.113 |        6.147 |      1.165 |
| server_bound      |        6.355 |      0.150 |        6.290 |      0.142 |

Subprocess wall (spawn→exit) — cold median 6.95 s, warm median 6.86 s.

## Per-run cumulative times

