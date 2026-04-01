# NEXT_SESSION_PROMPT.md — Squash Phase 5: `squish sbom` CLI subcommand + `squish doctor` squash check

> Paste the content below verbatim as your opening prompt.
> This is a **code session** — two targeted changes, one commit.

---

## Prompt

**Code session. Minimum viable. Two targeted changes, one commit, zero new squish/ modules.**

---

## This session is done when

1. `squish sbom show <model-dir>` prints a human-readable summary of the CycloneDX sidecar: component name, version, serialNumber (first 12 chars), composite weight SHA-256 (first 12 chars), and a performanceMetrics table. Exits 1 if no sidecar exists.
2. `squish sbom verify <model-dir>` re-hashes all weight files and compares the composite SHA-256 to the one stored in the sidecar. Exits 0 on match, 1 on mismatch. Prints "✓ hash match" or "✗ hash mismatch — expected \<stored\> got \<actual\>".
3. `squish sbom bind <model-dir> --result lmeval.json [--baseline baseline.json]` calls `EvalBinder.bind(bom_path, lmeval_path, baseline_path)` and prints "✓ bound N metrics". Exits 1 if no sidecar or `--result` missing.
4. `squish sbom sign <model-dir>` calls `OmsSigner.sign(bom_path)` and prints "✓ signed → \<sig-path\>" or "⚠ sigstore not installed — install: pip install sigstore". Exit 0 in both cases.
5. `squish sbom --help` prints a usage block listing all 4 sub-actions with one-line examples each.
6. `squish doctor` gains a squash check row between the network check and the final summary: `squish[squash] installed`, fix: `pip install "squish[squash]"`.
7. `tests/test_cli_sbom.py` — 7 tests all passing.
8. `find squish -name "*.py" | grep -v __pycache__ | wc -l` **≤ 100** (no new module in `squish/`; `cmd_sbom` lives in `cli.py`).
9. Committed and pushed with lm_eval-waiver (no weights touched, no quant logic changed).

---

## Context

**Repo:** `/Users/wscholl/squish`, HEAD `226b4d9`
**Branch:** `main`
**Python:** 3.12.8 · pytest 8.4.2 · module count: 97/100

### What was just completed (commit `226b4d9`)

- **Phase 4** `dev/squash_backfill.py` — generated CycloneDX sidecars for all 26 model dirs under `~/models/` with weight files, then called `EvalBinder.bind()` for all 22 models with complete lmeval results.
- **Phase 4** `dev/benchmarks/bench_lmeval_all_models.py` — `_save_model_result()` now auto-calls `EvalBinder.bind()` after every bench run.
- **Phase 4** `tests/test_squash_backfill.py` — 6 tests; 36/36 squash tests passing.

### The gap Phase 5 closes

The squash subsystem now generates sidecars, binds scores, and enforces compliance at serve time.
But there is **no CLI surface** for developers to inspect or manipulate sidecars without starting
a server or calling Python directly. The only user-visible squash entry points are:

- `squish compress` auto-generates a sidecar (side-effect, not inspectable from CLI).
- `squish serve/run --strict-compliance` passes traffic through the governor (server-side only).
- `GET /v1/sbom` returns the sidecar JSON (requires server to be running).

Phase 5 adds `squish sbom` — the missing developer-facing CLI — and adds squash to `squish doctor`.

---

## Read these files first

Before writing a single line:

```
squish/squash/sbom_builder.py   lines 1–80      CompressRunMeta fields; CycloneDXBuilder JSON schema
                                                  — especially the exact nested key path for the
                                                  composite weight hash (needed for sbom verify)
squish/squash/eval_binder.py    lines 1–60      EvalBinder.bind(bom_path, lmeval_json_path, baseline_path)
squish/squash/oms_signer.py     lines 1–80      OmsSigner.sign(bom_path) → Path | None
squish/squash/governor.py       lines 1–50      Understand _run_boot_checks weight hashing — reuse
                                                  the same logic in sbom verify rather than duplicating
squish/cli.py                   lines 1581–1790 cmd_doctor internals: _check() helper signature,
                                                  _results list, ok flag; find exact insertion point
squish/cli.py                   lines 5540–5570 How add_parser + set_defaults looks for catalog/rm;
                                                  follow this pattern for p_sbom
```

**Before writing any code**, state the exact JSON field path for the composite weight hash inside
`components[0]` of the CycloneDX document (confirm from `sbom_builder.py`, do not guess).
Then state where in `cmd_doctor` the squash check row belongs (line reference and surrounding code).

---

## Implementation spec

### `cmd_sbom(args)` in `squish/cli.py`

```python
def cmd_sbom(args) -> None:
    """squish sbom {show|verify|bind|sign} <model-dir> [options]"""
```

Wrap the entire function body in:

```python
try:
    from squish.squash.sbom_builder import _WEIGHT_EXTENSIONS
    from squish.squash.eval_binder import EvalBinder
    from squish.squash.oms_signer import OmsSigner
except ImportError:
    print("squish[squash] not installed — run: pip install 'squish[squash]'",
          file=sys.stderr)
    sys.exit(1)
```

**Sub-action dispatch** on `args.sbom_action`:

| action | Behaviour |
|---|---|
| `show` | Load sidecar JSON. Print component name, version, serialNumber[:12], hash[:12], and a performanceMetrics table (task \| value \| deltaFromBaseline). Exit 1 + stderr if no sidecar. |
| `verify` | Hash all weight files in model_dir matching `_WEIGHT_EXTENSIONS` (sorted, sha256, hex-concatenated and sha256'd again to a composite). Compare to hash in sidecar. Print ✓ or ✗ + values. Exit 0 / 1. Exit 1 if no sidecar. |
| `bind` | `EvalBinder.bind(bom_path, Path(args.result), Path(args.baseline) if args.baseline else None)`. Read the written BOM to count metrics; print "✓ bound N metrics to \<bom_path\>". Exit 1 if no sidecar or `--result` file missing. |
| `sign` | `sig = OmsSigner.sign(bom_path)`. If sig: print "✓ signed → \<sig\>". Else: print "⚠ sigstore not installed — install: pip install sigstore". Exit 0 in both cases. |

**Weight hashing for `verify`**: hash each weight file individually (sha256, read in 1 MB chunks),
collect hex digests sorted by filename, then sha256 the concatenated string. This is the same
composite the governor uses — confirm by reading `governor.py::_run_boot_checks`.

**Argparse** (placed after `p_catalog` in the parser block):

```python
p_sbom = sub.add_parser(
    "sbom",
    help="Inspect, verify, bind scores to, or sign the CycloneDX ML-BOM sidecar",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=(
        "Manage the CycloneDX ML-BOM sidecar (cyclonedx-mlbom.json).\n\n"
        "Sub-actions:\n"
        "  show    Pretty-print sidecar metadata and performance metrics.\n"
        "  verify  Re-hash weight files and compare against sidecar.\n"
        "  bind    Bind an lmeval result JSON into performanceMetrics.\n"
        "  sign    Sign the sidecar with Sigstore (requires: pip install sigstore).\n\n"
        "Examples:\n"
        "  squish sbom show ~/models/Qwen2.5-1.5B-Instruct-int4\n"
        "  squish sbom verify ~/models/Qwen2.5-1.5B-Instruct-int4\n"
        "  squish sbom bind ~/models/Qwen2.5-1.5B-Instruct-int4 \\\n"
        "    --result results/lmeval_Qwen2.5-1.5B-int4_20260328.json\n"
        "  squish sbom sign ~/models/Qwen2.5-1.5B-Instruct-int4\n"
    ),
)
p_sbom.add_argument("sbom_action", choices=["show", "verify", "bind", "sign"],
                    help="Sub-action: show | verify | bind | sign")
p_sbom.add_argument("model_dir", metavar="MODEL_DIR",
                    help="Path to the model directory containing cyclonedx-mlbom.json")
p_sbom.add_argument("--result", metavar="PATH",
                    help="(bind only) Path to lmeval result JSON")
p_sbom.add_argument("--baseline", metavar="PATH",
                    help="(bind only) Path to INT4 baseline result JSON for delta computation")
p_sbom.set_defaults(func=cmd_sbom)
```

### `squish doctor` squash check

Inside `cmd_doctor`, immediately before the final ok/fail summary block (just before the
`print()` that precedes `if ok:`), insert:

```python
# squash (optional)
try:
    import squish.squash.sbom_builder  # noqa: F401
    _check("squish[squash] installed", True)
except ImportError:
    _check("squish[squash] installed", False,
           'pip install "squish[squash]"')
```

This uses the `_check()` helper already in scope. Non-fatal — squash is optional.

---

## Tests — `tests/test_cli_sbom.py`

7 pure-unit tests. All use `tempfile.TemporaryDirectory` and `unittest.mock.patch`.
Import `cmd_sbom` from `squish.cli`. Build `argparse.Namespace` objects manually.

| # | Name | What it asserts |
|---|---|---|
| 1 | `test_show_prints_metrics` | Temp dir + minimal valid sidecar JSON → `cmd_sbom(show, dir)` prints component name and a metrics line. No SystemExit. |
| 2 | `test_show_exits_1_no_sidecar` | Dir exists, no sidecar → `SystemExit(1)`. |
| 3 | `test_verify_ok` | Sidecar hash exactly matches re-computed hash of temp weight file → exit 0, stdout contains "✓". |
| 4 | `test_verify_fail_mismatch` | Sidecar contains wrong hash → `SystemExit(1)`, stderr contains "mismatch". |
| 5 | `test_bind_calls_eval_binder` | Patch `squish.squash.eval_binder.EvalBinder.bind`. Confirm called with correct positional Path args. |
| 6 | `test_bind_exits_1_no_sidecar` | No sidecar → `SystemExit(1)`. |
| 7 | `test_sign_no_sigstore` | Patch `squish.squash.oms_signer.OmsSigner.sign` to return `None`. Confirm exit 0 and "⚠ sigstore" in captured stdout. |

**Patch targets must be the source-module paths** (`squish.squash.eval_binder.EvalBinder.bind` etc.)
because `cmd_sbom` uses local imports inside the function body — patching `cli.EvalBinder.bind`
will miss the real call site.

**Minimal sidecar fixture** for tests 1, 3, 5, 7 (use in a helper `_write_minimal_sidecar(path)`):

```python
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.7",
  "serialNumber": "urn:uuid:test-serial-0001",
  "components": [{
    "type": "machine-learning-model",
    "name": "test-model",
    "version": "test-int4",
    "hashes": [{"alg": "SHA-256", "content": "<correct-hash-goes-here>"}],
    "modelCard": {
      "quantitativeAnalysis": {
        "performanceMetrics": [
          {"type": "arc_easy", "value": 70.6, "slice": "arc_easy"}
        ]
      }
    }
  }]
}
```

For test 3, compute the real SHA-256 of your temp weight file and put it in `content`.
For test 4, put a wrong hash in `content`.

Confirm the exact `hashes` field path from `sbom_builder.py` before writing fixtures.

---

## Smoke-test after implementation

```bash
# 1. show — should print metrics table
squish sbom show ~/models/Qwen2.5-1.5B-Instruct-int4

# 2. verify — should print ✓ (unmodified weights)
squish sbom verify ~/models/Qwen2.5-1.5B-Instruct-int4

# 3. doctor — should show squash check row
squish doctor

# 4. tests
pytest tests/test_cli_sbom.py -v -W error::DeprecationWarning
pytest tests/test_eval_binder.py tests/test_oms_signer.py \
       tests/test_governor_middleware.py tests/test_squash_backfill.py \
       tests/test_cli_sbom.py -v
```

---

## Ship gate checklist

- [ ] `squish sbom --help` shows 4 sub-actions with examples
- [ ] `squish sbom show ~/models/Qwen2.5-1.5B-Instruct-int4` prints metrics table
- [ ] `squish sbom verify ~/models/Qwen2.5-1.5B-Instruct-int4` exits 0 (unmodified weights)
- [ ] `squish doctor` shows squash check row
- [ ] 7/7 `test_cli_sbom.py` tests pass
- [ ] Full squash suite (43 tests) pass: `pytest tests/test_eval_binder.py tests/test_oms_signer.py tests/test_governor_middleware.py tests/test_squash_backfill.py tests/test_cli_sbom.py -v`
- [ ] `find squish -name "*.py" | grep -v __pycache__ | wc -l` ≤ 100
- [ ] CHANGELOG.md entry written
- [ ] `git commit -m "feat(squash): Phase 5 squish sbom CLI subcommand + doctor squash check"` and push
- [ ] lm_eval-waiver in commit body (no weights or quant logic touched)
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

