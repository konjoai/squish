# Cold-load profile report  (20260601_122932)

Host: Apple M3 MacBook Pro 16 GB · macOS 25.5.0 · Python 3.14.3 · MLX-LM 0.31.1
Model: `Qwen2.5-7B-Instruct-int4` (INT4 MLX safetensors)

Times are cumulative seconds from process start (t0).
`Δ` is the per-phase delta (this phase only).

| Phase             | Cold cum (s) | Cold Δ (s) | Warm cum (s) | Warm Δ (s) |
|-------------------|-------------:|-----------:|-------------:|-----------:|
| squish_imports    |        0.347 |      0.347 |        0.363 |      0.363 |
| mlx_core_import   |        0.381 |      0.034 |        0.399 |      0.036 |
| mlx_lm_import     |        2.940 |      2.559 |        3.044 |      2.645 |
| mlx_utils_import  |        2.940 |      0.000 |        3.044 |      0.000 |
| weights_loaded    |        4.265 |      1.325 |        4.344 |      1.300 |
| tokenizer_loaded  |        4.936 |      0.671 |        5.087 |      0.742 |
| warmup_done       |        5.956 |      1.020 |        6.112 |      1.025 |
| server_bound      |        6.052 |      0.095 |        6.198 |      0.087 |

Subprocess wall (spawn→exit) — cold median 6.66 s, warm median 6.87 s.

## Per-run cumulative times

