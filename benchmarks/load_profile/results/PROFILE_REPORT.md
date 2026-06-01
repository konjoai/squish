# Cold-load profile report (baseline)

**Date:** 2026-06-01
**Run:** `cold_20260601_122044_baseline`
**Host:** Apple M3 MacBook Pro 16 GB · macOS 25.5.0 · Python 3.14.3 · MLX-LM 0.31.1
**Model:** `Qwen2.5-7B-Instruct-int4` (INT4 MLX safetensors)
**Page cache:** `sudo purge` unavailable (no NOPASSWD sudoers entry) — cold runs use process restart only, matching the Ollama-vs-Squish benchmark's definition of "cold."

## Phase breakdown (medians over 5 cold + 3 warm runs)

Cumulative seconds from process start (t0). `Δ` is the per-phase delta.

| Phase             | Cold cum (s) | Cold Δ (s) | Warm cum (s) | Warm Δ (s) |
|-------------------|-------------:|-----------:|-------------:|-----------:|
| squish_imports    |        0.340 |      0.340 |        0.355 |      0.355 |
| mlx_core_import   |        0.374 |      0.033 |        0.388 |      0.034 |
| **mlx_lm_import** |    **4.426** |  **4.052** |    **4.231** |  **3.842** |
| mlx_utils_import  |        4.426 |      0.000 |        4.231 |      0.000 |
| weights_loaded    |        5.823 |      1.397 |        5.545 |      1.315 |
| tokenizer_loaded  |        6.585 |      0.762 |        6.171 |      0.626 |
| warmup_done       |        7.848 |      1.263 |        7.324 |      1.152 |
| server_bound      |        8.007 |      0.159 |        7.451 |      0.128 |

Subprocess wall (spawn→exit) — cold median **8.68 s**, warm median **8.12 s**.

## Where the 4 s `mlx_lm` import comes from

`python -X importtime -c "import mlx_lm"` shows the chain:

```
import time:       538 |    4652779 | mlx_lm
import time:       537 |    4646940 |   mlx_lm.convert          <- 4.65 s
import time:      2129 |    4577155 |     mlx_lm.utils
import time:    390052 |    4226124 |       mlx_lm.tokenizer_utils
import time:    639074 |    2242017 |         transformers      <- 2.24 s
import time:    639074 |    1601658 |           transformers.dependency_versions_check
import time:    639074 |    1547553 |             transformers.utils.generic
import time:    639074 |    1255478 |           transformers.generation.candidate_generator
import time:    701    |    1254512 |             sklearn.metrics  <- 1.25 s
import time:    48975  |     856077 |               torch        <- 0.86 s
```

**`mlx_lm/__init__.py` eagerly imports `from .convert import convert`** — the
offline conversion CLI used by `python -m mlx_lm.convert`. That tool is not
needed at server runtime, but its import chain pulls in:
- `mlx_lm.tokenizer_utils` → `transformers` (2.2 s)
- `transformers.generation.candidate_generator` → `sklearn.metrics` (1.25 s)
- `sklearn` → `torch` (0.86 s) + scipy + pandas

So **most of the 4 s mlx_lm import is import-side-effect overhead from a
module squish never calls at runtime.**

## Per-run samples (for variance)

Cold runs, `mlx_lm_import` Δ per run: 5.22, 3.41, 4.68, 4.05, 3.81 s
Warm runs, `mlx_lm_import` Δ per run: 3.78, 3.82, 3.97 s

Variance is real (σ ≈ 0.7 s cold) because file-cache fill of the transformers
Python source tree is not uniform. The warm runs are tighter (σ ≈ 0.1 s).

## Reproducing

```bash
source .venv/bin/activate
python benchmarks/load_profile/profile_cold_load.py --label baseline --no-purge
# Per-run JSON + cProfile + this report all land under
# benchmarks/load_profile/results/
```

The `--no-purge` flag is set because this host doesn't have `sudo purge`
configured for NOPASSWD. With purge available, cold numbers are typically
3–5 s higher because the safetensors file is also cold-read.
