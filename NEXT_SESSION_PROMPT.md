# NEXT_SESSION_PROMPT.md — Phase 0: Fill Baseline Gaps

> Paste the content below verbatim as your opening prompt.
> This is a **bench-and-commit session** — no source files are written.

---

## Prompt

**Code session. Run bench, commit results, no new source files.**

---

## This session is done when

1. `results/lmeval_Qwen2.5-7B-int3_<ts>.json` — `tasks=6/6 errors=0`
2. `results/lmeval_Qwen3-8B-int3_<ts>.json` — `tasks=6/6 errors=0`
3. `results/lmeval_Qwen3-4B-int4_<ts>.json` — `tasks=6/6 errors=0`
4. `results/lmeval_Llama-3.2-3B-int4_<ts>.json` — `tasks=6/6 errors=0`
5. `results/lmeval_gemma-3-4b-int4_<ts>.json` — `tasks=6/6 errors=0`
6. All five committed and pushed to `origin/main`
7. Safety gate table printed (see Step 4)
8. No new Python source files created

---

## Context

**Repo:** `/Users/wscholl/squish`, HEAD `9080c8d`
**Branch:** `main`
**Bench script:** `dev/benchmarks/bench_lmeval_all_models.py`
**Results dir:** `results/`
**Model root:** `~/models/`
**Platform:** macOS M3 16 GB, mlx_lm 0.31.1, lm_eval 0.4.7, Python 3.14.3

**No bench is currently running.** All previous attempts at these five models
errored or are incomplete. Exact gap state as of March 30, 2026:

| Model | Best existing result | Gap |
|---|---|---|
| `Qwen2.5-7B-int3` | All 3 files `tasks=0/6 errors=6` | Needs full 6-task run |
| `Qwen3-8B-int3` | Both files `tasks=0/6 errors=6` | Needs full 6-task run |
| `Qwen3-4B-int4` | Best file `tasks=1/6` (only arc_easy=37.0%) | Fresh full run with `--force` |
| `Llama-3.2-3B-int4` | `tasks=5/6`, missing `arc_challenge` | Full run with `--force` |
| `gemma-3-4b-int4` | All files `tasks=0/6 errors=6` | Needs full 6-task run |

**Why these matter (Phase 1 dependency):**
- `Qwen3-4B-int4`: INT3=42.2% > INT4=37.0% is physically impossible; must resolve
  the anomaly before Phase 1 can classify Qwen3-4B.
- `Llama-3.2-3B-int4`: missing `arc_challenge` → no clean Δ for safety gate.
- `gemma-3-4b-int4`: zero data → gate cannot classify `gemma-3-4b-int3=64.4%`.
- `Qwen2.5-7B-int3` and `Qwen3-8B-int3`: complete the Tier 2/3 INT3 evidence table.

**Known safe:** all five models are ≤4.0 GB in the registry (default
`--max-model-gb=9.0` guard will not skip any of them).

---

## Step 1 — Verify current state

```bash
cd /Users/wscholl/squish
python3 -c "
import json, glob
models = ['Qwen2.5-7B-int3', 'Qwen3-8B-int3', 'Qwen3-4B-int4', 'Llama-3.2-3B-int4', 'gemma-3-4b-int4']
for m in models:
    files = sorted(glob.glob(f'results/lmeval_{m}_*.json'))
    if not files: print(f'{m}: NO FILES'); continue
    for f in files:
        d = json.load(open(f))
        s, e = d.get('scores', {}), d.get('errors', {})
        print(f'{m}: tasks={len(s)}/6 errors={len(e)} file={f.split(\"/\")[-1]}')
"
```

If any model already shows `tasks=6/6 errors=0`, skip it (do not `--force` it —
remove it from the `--models` list instead).

---

## Step 2 — Launch the bench in a background terminal

Run as **background terminal** (`isBackground=true`) so the process survives
terminal closure. All five models run serially, each releasing the Metal heap
before the next starts.

```bash
cd /Users/wscholl/squish && python3 dev/benchmarks/bench_lmeval_all_models.py \
  --models Qwen2.5-7B-int3 Qwen3-8B-int3 Qwen3-4B-int4 Llama-3.2-3B-int4 gemma-3-4b-int4 \
  --tasks arc_easy arc_challenge hellaswag winogrande piqa openbookqa \
  --limit 500 \
  --output-dir results \
  --force \
  --gen-sanity
```

Flag rationale:
- `--force` — re-run even if errored partial JSONs already exist
- `--gen-sanity` — 3-prompt coherence check; records results but does not skip
- `--limit 500` — matches all prior committed results in this bench series
- `--tasks ...` — the canonical 6-task set used throughout

---

## Step 3 — Poll and commit each result as it lands

Each model writes its JSON immediately on completion. Commit one at a time —
do not batch.

Check completion:

```bash
cd /Users/wscholl/squish && python3 -c "
import json, glob
models = ['Qwen2.5-7B-int3', 'Qwen3-8B-int3', 'Qwen3-4B-int4', 'Llama-3.2-3B-int4', 'gemma-3-4b-int4']
for m in models:
    files = sorted(glob.glob(f'results/lmeval_{m}_*.json'))
    if not files: print(f'{m}: PENDING'); continue
    d = json.load(open(files[-1]))
    s, e = d.get('scores', {}), d.get('errors', {})
    status = 'COMPLETE' if len(s)==6 and len(e)==0 else f'PARTIAL {len(s)}/6 err={len(e)}'
    print(f'{m}: {status}  arc_easy={s.get(\"arc_easy\",\"—\")}')
"
```

Commit pattern (one per model):

```bash
# Replace MODEL, TS, and scores with real values
git add results/lmeval_MODEL_TS.json results/_mlx_lmeval_raw/MODEL/
git commit -m "bench(results): MODEL lm_eval (limit=500, 2026-03-30)

arc_easy=XX.X arc_challenge=XX.X hellaswag=XX.X winogrande=XX.X piqa=XX.X openbookqa=XX.X"
git push
```

---

## Step 4 — Print the safety gate table after all five complete

```bash
cd /Users/wscholl/squish && python3 -c "
import json, glob

pairs = [
    ('Qwen3-0.6B',   'Qwen3-0.6B-int4',   'Qwen3-0.6B-int3'),
    ('Llama-3.2-1B', 'Llama-3.2-1B-int4',  'Llama-3.2-1B-int3'),
    ('gemma-3-1b',   'gemma-3-1b-int4',    'gemma-3-1b-int3'),
    ('Qwen2.5-1.5B', 'Qwen2.5-1.5B-int4',  'Qwen2.5-1.5B-int3'),
    ('Llama-3.2-3B', 'Llama-3.2-3B-int4',  'Llama-3.2-3B-int3'),
    ('Qwen3-4B',     'Qwen3-4B-int4',      'Qwen3-4B-int3'),
    ('gemma-3-4b',   'gemma-3-4b-int4',    'gemma-3-4b-int3'),
    ('Qwen2.5-7B',   'Qwen2.5-7B-int4',    'Qwen2.5-7B-int3'),
    ('Qwen3-8B',     'Qwen3-8B-int4',      'Qwen3-8B-int3'),
]

def best(model):
    files = sorted(glob.glob(f'results/lmeval_{model}_*.json'))
    for f in reversed(files):
        d = json.load(open(f))
        s = d.get('scores', {})
        if 'arc_easy' in s:
            return s
    return {}

header = f\"{'Model':<16} {'INT4 ae':>8} {'INT3 ae':>8} {'Δpp':>6} {'Ratio':>7} {'Gate':>6}\"
print(header)
print('-' * len(header))
for name, m4, m3 in pairs:
    s4, s3 = best(m4), best(m3)
    ae4, ae3 = s4.get('arc_easy'), s3.get('arc_easy')
    if ae4 and ae3:
        delta = ae3 - ae4
        ratio = ae3 / ae4
        gate = 'PASS' if ratio >= 0.92 else 'FAIL'
        print(f'{name:<16} {ae4:>8.1f} {ae3:>8.1f} {delta:>+6.1f} {ratio:>7.2%} {gate:>6}')
    else:
        ae4s = f'{ae4:.1f}' if ae4 else '—'
        ae3s = f'{ae3:.1f}' if ae3 else '—'
        print(f'{name:<16} {ae4s:>8} {ae3s:>8} {\"—\":>6} {\"—\":>7} {\"PENDING\":>6}')
"
```

This table is the direct input to Phase 1 (`int3_baselines.py`). Paste it into
the Phase 1 prompt.

---

## Hard stops

- **Do not run INT2 variants.** Permanently cancelled — naive INT2 is confirmed
  incoherent at all tested sizes (arc_easy ~27%, near random).
- **Do not run BF16 reference models** (`--include-bf16` would OOM at 14–15 GB).
- **Do not modify any source file.** Bench-and-commit only.
- If a model's gen-sanity check flags repetition/garbage, record it in the commit
  message but still commit the lmeval scores.
- If any model crashes again (`tasks=0 errors=6`), inspect stdout/stderr from the
  background terminal for the Metal error, and report it rather than retrying
  blindly. Common cause: Metal heap fragmentation from prior failed run — try
  rebooting and running that model alone.
