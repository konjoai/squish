# Cold-load profile report — baseline → all fixes

**Host:** Apple M3 MacBook Pro 16 GB · macOS 25.5.0 · Python 3.14.3 · MLX-LM 0.31.1
**Model:** `Qwen2.5-7B-Instruct-int4` (INT4 MLX safetensors)
**Page cache:** `sudo purge` unavailable on this host — cold runs use process restart only, matching the Ollama-vs-Squish bench's definition of "cold".

This report tracks the four runs we did across Phase 1 → Phase 3. Each
row is the median across 5 cold + 3 warm runs of
``profile_cold_load.py``.

## Per-phase deltas (warm median, seconds)

| Phase             | Baseline | Fix #1 (sklearn) | Fix #1+#2 (parallel tok) | Fix #1+#2+#3 (bg mlx_lm) |
|-------------------|---------:|-----------------:|-------------------------:|-------------------------:|
| squish_imports    |    0.355 |            0.347 |                    0.338 |                    0.450 |
| mlx_core_import   |    0.034 |            0.036 |                    0.034 |                    0.000 |
| mlx_lm_import     |    3.842 |            2.645 |                    2.602 |                    2.535 |
| weights_loaded    |    1.315 |            1.300 |                    1.368 |                    1.329 |
| tokenizer_loaded  |    0.626 |            0.742 |                    0.613 |                    0.668 |
| warmup_done       |    1.152 |            1.025 |                    1.068 |                    1.165 |
| server_bound      |    0.128 |            0.087 |                    0.090 |                    0.142 |
| **Subprocess wall** | **8.12** |       **6.87**   |                  **6.66** |                  **6.86** |

Note: the per-phase profile measures load_model and load_tokenizer
**serially** (so each phase has a clean delta). The parallelisation
win from Fix #2 (~0.5 s) is therefore invisible here — it shows up in
the production-path microbench below and in the Ollama-vs-Squish
bench.

The per-phase wall is also numerically slightly *worse* with Fix #3
because the bg-thread spawn at module-load adds ~0.1 s and the
profile_child does its own sequential mlx_lm import immediately
after (no other work for the bg thread to overlap with). In real
``squish.server`` usage there's ~0.5 s of pre-load main() work that
gives the bg thread something to hide behind.

## Production-path microbench (load_mlx_model end-to-end, median of 5–7 runs)

| Config                       | Median load_mlx_model (s) | Δ vs baseline |
|------------------------------|--------------------------:|--------------:|
| Baseline (no fixes)          |                  ~7.5     |        —      |
| Fix #1 (sklearn stub)        |                   5.86    |       −1.6 s  |
| Fix #1 + Fix #2 (parallel tok) |                 5.18    |       −2.3 s  |
| Fix #1 + Fix #2 + Fix #3 (bg) |                  5.07    |       −2.4 s  |

The microbench launches a fresh subprocess and times
``squish.server.load_mlx_model(model_dir)`` from a single ``import
squish.server`` already done outside the timer. This isolates the
"how long does the model load itself take" question from process
startup and import overhead, and is the best proxy for the
``cold_wall_s`` metric in the Ollama bench.

## Where the 4 s `mlx_lm` import came from (Phase 1 finding)

``python -X importtime -c "import mlx_lm"`` showed:

```
import time:       538 |    4652779 | mlx_lm
import time:       537 |    4646940 |   mlx_lm.convert          <- 4.65 s
import time:      2129 |    4577155 |     mlx_lm.utils
import time:    390052 |    4226124 |       mlx_lm.tokenizer_utils
import time:    639074 |    2242017 |         transformers      <- 2.24 s
import time:    639074 |    1255478 |           transformers.generation.candidate_generator
import time:    701    |    1254512 |             sklearn.metrics  <- 1.25 s
import time:    48975  |     856077 |               torch        <- 0.86 s
```

`mlx_lm/__init__.py` eagerly imports `from .convert import convert`
(the offline conversion CLI used by `python -m mlx_lm.convert`),
which transitively loads transformers, sklearn (~1.3 s) and torch
(~0.9 s). squish never calls `convert` at runtime, but it pays the
import cost anyway.

Fix #1 attacks the sklearn branch (which transformers'
`candidate_generator` imports for assisted-decoding feature squish
doesn't use). Stubbing `sklearn` + `sklearn.metrics` in `sys.modules`
before mlx_lm loads cuts ~1.2 s in practice.

## Reproducing

```bash
source .venv/bin/activate
# Per-phase profile (5 cold + 3 warm runs)
python benchmarks/load_profile/profile_cold_load.py --label myrun --no-purge

# Production-path microbench (5 runs by default)
python benchmarks/load_profile/microbench_load_mlx_model.py --runs 5
```

The `--no-purge` flag is set because this host doesn't have `sudo
purge` configured for NOPASSWD. With purge available, cold numbers
are typically 3–5 s higher because the safetensors file is also
cold-read.

See `ANALYSIS.md` for the classification of each phase as
Fundamental / Avoidable / Cacheable / Parallelizable, and the
``perf(load):`` git commits for the implementation of each fix.
