# NEXT_SESSION_PROMPT.md — Squash Phase 4: Retroactive sidecar backfill + EvalBinder auto-wiring

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — two targeted changes, one commit.

---

## Prompt

**Code session. Minimum viable. Two changes, one commit, no new modules.**

---

## This session is done when

1. `dev/squash_backfill.py` generates `cyclonedx-mlbom.json` in all 26 model dirs under `~/models/` that have weight files and no existing sidecar.
2. `dev/squash_backfill.py` then calls `EvalBinder.bind()` for every model dir that has a matching result in `results/lmeval_<name>_*.json` (6 currently validated).
3. `dev/benchmarks/bench_lmeval_all_models.py` auto-calls `EvalBinder.bind()` in `_save_result()` whenever the saved JSON lands, so future bench runs self-populate sidecars.
4. `tests/test_squash_backfill.py` — 5 tests covering: (a) no sidecar → sidecar generated, (b) bind called when result JSON exists, (c) existing sidecar not overwritten if `--no-overwrite`, (d) missing result JSON → bind skipped gracefully, (e) smoke-run of the `_save_result` auto-bind path.
5. All 35 squash tests pass: `pytest tests/test_eval_binder.py tests/test_oms_signer.py tests/test_governor_middleware.py tests/test_squash_backfill.py -v -W error::DeprecationWarning`
6. `find squish -name "*.py" | grep -v __pycache__ | wc -l` **≤ 100** (no new module in `squish/`).
7. Committed and pushed with lm_eval-waiver (no weights touched).

---

## Context

**Repo:** `/Users/wscholl/squish`, HEAD `5c54044`
**Branch:** `main`
**Python:** 3.12.8 · pytest 8.4.2 · module count: 97/100

### What was just completed (commit `5c54044`)

- **Phase 1** `squish/squash/sbom_builder.py` — `CycloneDXBuilder.build()` writes `cyclonedx-mlbom.json` as a post-hook from `squish compress`.
- **Phase 2** `squish/squash/eval_binder.py` — `EvalBinder.bind(bom_path, lmeval_json_path, baseline_path)` mutates `performanceMetrics` in an existing sidecar.
- **Phase 3** `squish/squash/governor.py` — `SquashGovernor` Starlette middleware; hash integrity + arc_easy accuracy boot-gate; `GET /v1/sbom` + `GET /v1/health/model` routes wired into `server.py`; `--strict-compliance` + `--min-accuracy-ratio` flags in `cli.py`.

### The gap Phase 4 closes

`CycloneDXBuilder.build()` only runs as a post-hook when `squish compress` is called.
Every model in `~/models/` was compressed before Phase 1 existed — **zero sidecars exist**.
Six validated bench results exist that can already populate `performanceMetrics`,
but nothing has called `EvalBinder.bind()` for them yet.

```
Qwen3-0.6B-int4    result=True(6/6)  sidecar=False
Qwen3-0.6B-int3    result=True(6/6)  sidecar=False
Llama-3.2-1B-int4  result=True(6/6)  sidecar=False
Llama-3.2-1B-int3  result=True(6/6)  sidecar=False
Qwen2.5-1.5B-int4  result=True(6/6)  sidecar=False
Qwen2.5-1.5B-int3  result=True(6/6)  sidecar=False
```

### Bench result ↔ model dir name mapping

The bench script uses shortened names; the model dirs use full Hugging Face names:

| Bench name | Model dir name |
|---|---|
| `Qwen3-0.6B-int4` | `~/models/Qwen3-0.6B-int4` |
| `Qwen3-0.6B-int3` | `~/models/Qwen3-0.6B-int3` |
| `Llama-3.2-1B-int4` | `~/models/Llama-3.2-1B-Instruct-int4` |
| `Llama-3.2-1B-int3` | `~/models/Llama-3.2-1B-Instruct-int3` |
| `Qwen2.5-1.5B-int4` | `~/models/Qwen2.5-1.5B-Instruct-int4` |
| `Qwen2.5-1.5B-int3` | `~/models/Qwen2.5-1.5B-Instruct-int3` |
| `gemma-3-1b-int4` | `~/models/gemma-3-1b-it-int4` |
| `gemma-3-1b-int3` | `~/models/gemma-3-1b-it-int3` |

The mapping is a simple transform: strip `-Instruct` suffix, replace `gemma-3-1b-` with `gemma-3-1b-it-`.
Build this as a dict in `squash_backfill.py` — do not hard-code paths.

### Baseline pairing for `EvalBinder.bind(baseline_path=...)`

For each INT3 model, the INT4 result of the same family is the baseline.
For INT4 models, there is no baseline (pass `baseline_path=None`).

| Quantized | Baseline |
|---|---|
| `Qwen3-0.6B-int3` | most-recent `results/lmeval_Qwen3-0.6B-int4_*.json` |
| `Llama-3.2-1B-int3` | most-recent `results/lmeval_Llama-3.2-1B-int4_*.json` |
| `Qwen2.5-1.5B-int3` | most-recent `results/lmeval_Qwen2.5-1.5B-int4_*.json` |
| `gemma-3-1b-int3` | most-recent `results/lmeval_gemma-3-1b-int4_*.json` |

`_most_recent(glob_pattern)` — return the most recently modified file matching pattern, or None.

---

## Step 1 — Read these files first

Before writing a single line:

```
squish/squash/sbom_builder.py   lines 1–80     (CompressRunMeta, CycloneDXBuilder.build signature)
squish/squash/eval_binder.py    lines 1–60     (EvalBinder.bind signature + schema)
dev/benchmarks/bench_lmeval_all_models.py  lines 555–600  (_save_result function)
```

State your understanding of `CycloneDXBuilder.build(meta)` — what is `CompressRunMeta` and what fields does it require?
Then ask: what is the minimum `CompressRunMeta` you can construct for a model that was compressed externally (we don't have the original `squish compress` args)?

The answer: `CompressRunMeta` needs `model_name: str`, `model_dir: Path`, `format: str`, `group_size: int`, `awq_alpha: float`.
Infer `format` from the dir-name suffix (`-int4` → `"int4"`, `-int3` → `"int3"`, `-int2` → `"int2"`).
Infer `group_size` from format: `int4` → 64, `int3` → 32, `int2` → 16. Use `awq_alpha=0.10` (0.07 for Qwen3 — call `detect_model_family`).
Do not infer anything else — only populate fields that `CycloneDXBuilder.build()` actually uses.

---

## Step 2 — Write `dev/squash_backfill.py`

Location: `dev/squash_backfill.py` — a **standalone dev script**, not a module.
No `squish/` import of this file. No new entry in `__init__.py`. Module count unchanged.

```
Usage:
  python3 dev/squash_backfill.py [--models-root ~/models] [--results-dir results]
                                  [--no-overwrite] [--dry-run]

Flags:
  --models-root   Root directory containing model dirs (default: ~/models)
  --results-dir   Directory containing lmeval_*.json files (default: results)
  --no-overwrite  Skip model dirs that already have cyclonedx-mlbom.json
  --dry-run       Print what would happen; write nothing

Exit codes:  0 success  1 user/arg error  2 runtime error (see stderr)
```

**Logic:**

```
for each dir in models_root:
    if not has_weight_files(dir):            # skip dirs with no .safetensors/.npy
        continue
    if sidecar_exists and --no-overwrite:
        print(f"SKIP {name} (sidecar exists)")
        continue
    meta = _infer_meta(dir)                  # construct CompressRunMeta
    CycloneDXBuilder.build(meta)             # writes cyclonedx-mlbom.json
    print(f"WROTE sidecar: {dir}/cyclonedx-mlbom.json")

    bom_path    = dir / "cyclonedx-mlbom.json"
    bench_name  = _dir_to_bench_name(dir)    # reverse the name mapping
    result_json = _most_recent(f"results/lmeval_{bench_name}_*.json")
    if result_json is None:
        print(f"  no result for {bench_name} — skipping EvalBinder")
        continue
    # Baseline: INT4 result for INT3 models; None otherwise
    baseline_json = _baseline_for(bench_name)
    EvalBinder.bind(bom_path, result_json, baseline_json)
    print(f"  BOUND {result_json.name} → performanceMetrics")
```

Print a summary table at the end:
```
Sidecars written:   N
Sidecars skipped:   N  (already existed + --no-overwrite)
Scores bound:       N
Models with no result yet:  N
```

---

## Step 3 — Wire `EvalBinder.bind()` into `bench_lmeval_all_models.py`

Find `_save_result()` (around line 563). After `out_file.write_text(...)`, add:

```python
# ── Auto-populate CycloneDX sidecar if squish[squash] is installed ──────────
try:
    from squish.squash.eval_binder import EvalBinder as _EB
    from squish.squash.eval_binder import _most_recent as _mr  # if you extract it
    _bom = Path(model_dir) / "cyclonedx-mlbom.json"
    if _bom.exists():
        # For INT3 models pair with the INT4 baseline; others pass None.
        _baseline = None
        if model_name.endswith(("-int3", "-INT3")):
            _b4 = model_name[:-5] + "-int4"
            _baseline = _most_recent_result(_b4, output_dir)
        _EB.bind(_bom, out_file, _baseline)
except ImportError:
    pass  # squish[squash] optional
except Exception as _e:
    print(f"  squash bind warning: {_e}", file=sys.stderr)
```

Where `_most_recent_result(bench_name, output_dir)` returns the most recently
modified `output_dir/lmeval_{bench_name}_*.json`, or None.

This is a **non-fatal try/except block** — if `squish[squash]` is not installed
or bind fails for any reason, the bench continues normally. Never let squash
failures abort a bench run.

---

## Step 4 — Write `tests/test_squash_backfill.py`

5 tests. Use `tempfile.TemporaryDirectory`. Patch `CycloneDXBuilder.build` with a
stub that writes a minimal `cyclonedx-mlbom.json`; patch `EvalBinder.bind` to
record call args. Tests must be **pure unit** (no filesystem side-effects outside
temp dirs, no imports of bench script globals).

| # | Test | Assertion |
|---|---|---|
| 1 | `test_sidecar_written_for_model_with_weights` | Given dir with a `.safetensors` file + no sidecar → `CycloneDXBuilder.build` called once |
| 2 | `test_bind_called_when_result_exists` | Given sidecar + matching `results/lmeval_X_*.json` → `EvalBinder.bind` called with correct paths |
| 3 | `test_no_overwrite_skips_existing_sidecar` | `--no-overwrite` + existing sidecar → `build` NOT called |
| 4 | `test_bind_skipped_when_no_result_json` | Sidecar written but no matching result JSON → `bind` NOT called |
| 5 | `test_dry_run_writes_nothing` | `--dry-run` flag → no files created, nothing written |

Import the logic as importable functions (not `__main__` only). Either:
- Extract `_process_one_model(dir, ...)` as a testable function in `squash_backfill.py`, OR
- Test via `subprocess.run(['python3', 'dev/squash_backfill.py', '--dry-run', '--models-root', td], ...)` — subprocess class is acceptable here.

---

## Step 5 — Run the backfill against real models

**After** all tests pass, run for real:

```bash
cd /Users/wscholl/squish
python3 dev/squash_backfill.py \
  --models-root ~/models \
  --results-dir results \
  --no-overwrite
```

Verify 6 sidecars gained `performanceMetrics`:

```bash
python3 -c "
import json, os
models = [
    ('~/models/Qwen3-0.6B-int4',          'Qwen3-0.6B-int4'),
    ('~/models/Qwen3-0.6B-int3',          'Qwen3-0.6B-int3'),
    ('~/models/Llama-3.2-1B-Instruct-int4','Llama-3.2-1B-int4'),
    ('~/models/Llama-3.2-1B-Instruct-int3','Llama-3.2-1B-int3'),
    ('~/models/Qwen2.5-1.5B-Instruct-int4','Qwen2.5-1.5B-int4'),
    ('~/models/Qwen2.5-1.5B-Instruct-int3','Qwen2.5-1.5B-int3'),
]
for d, name in models:
    sidecar = os.path.expanduser(f'{d}/cyclonedx-mlbom.json')
    if not os.path.exists(sidecar):
        print(f'MISSING {name}')
        continue
    bom = json.load(open(sidecar))
    metrics = (bom.get('components', [{}])[0]
               .get('modelCard', {})
               .get('quantitativeAnalysis', {})
               .get('performanceMetrics', []))
    n = len(metrics)
    arc = next((m for m in metrics if m.get('slice') == 'arc_easy'), None)
    ae = arc.get('value', '—') if arc else '—'
    delta = arc.get('deltaFromBaseline', '—') if arc else '—'
    print(f'{name}: metrics={n}  arc_easy={ae}  delta={delta}')
"
```

Expected output (values from existing bench results):
```
Qwen3-0.6B-int4:    metrics=6 arc_easy=35.0  delta=—
Qwen3-0.6B-int3:    metrics=6 arc_easy=37.2  delta=+2.2
Llama-3.2-1B-int4:  metrics=6 arc_easy=40.8  delta=—
Llama-3.2-1B-int3:  metrics=6 arc_easy=37.4  delta=-3.4
Qwen2.5-1.5B-int4:  metrics=6 arc_easy=70.6  delta=—
Qwen2.5-1.5B-int3:  metrics=6 arc_easy=67.2  delta=-3.4
```

---

## Step 6 — Commit and push

```bash
cd /Users/wscholl/squish
git add dev/squash_backfill.py dev/benchmarks/bench_lmeval_all_models.py \
        tests/test_squash_backfill.py \
        ~/models/*/cyclonedx-mlbom.json   # <-- actually outside git root, skip
# Sidecars live outside the repo — do NOT add them to git.
git commit -F /tmp/squash_phase4_msg.txt
git push origin main
```

Commit message template:

```
feat(squash): Phase 4 retroactive sidecar backfill + EvalBinder auto-wiring

- dev/squash_backfill.py: generates CycloneDX sidecars for all existing model dirs,
  then calls EvalBinder.bind() for any dir with a matching lmeval result JSON.
  Flags: --no-overwrite, --dry-run.  Zero new squish/ modules.

- dev/benchmarks/bench_lmeval_all_models.py: _save_result() now auto-calls
  EvalBinder.bind() after writing each result JSON.  Non-fatal try/except —
  squash failures never abort a bench run.

Results: 6 model sidecars populated with performanceMetrics (Qwen3-0.6B int4/int3,
Llama-3.2-1B int4/int3, Qwen2.5-1.5B int4/int3).

35/35 squash tests passing. Module count: 97/100.

lm_eval-waiver: no inference path or quantization logic changed
expected-delta: 0pp (weights untouched; sidecar files live outside repo)
```

---

## Hard stops

- **Do not create any new file inside `squish/`** — that would consume module quota.
  `dev/squash_backfill.py` and `tests/test_squash_backfill.py` are the only new files.
- **Do not add sidecar files to git.** They live in `~/models/` which is outside the repo.
- If `CycloneDXBuilder.build()` requires fields from `CompressRunMeta` that cannot be
  inferred from the model dir alone, read `sbom_builder.py` more carefully and use
  the minimum set of fields it actually accesses. Do not guess at API shape.
- The bench-script change must be wrapped in `try/except ImportError` and
  `try/except Exception` — squash must never abort a production bench run.
- `tests/test_squash_backfill.py` must be pure unit (temp dirs + patches) — no
  real model loading, no Metal, no subprocess that touches `~/models/`.

