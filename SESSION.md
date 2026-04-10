# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-09

## Last commits
- **`(w56 — pending push)`** — feat(quant): W56 AQLM encode path — AQLMEncoder K-means codebook training, encode_weight_matrix, squish compress --format aqlm, ~2 bpw INT2 ultra tier — 47 new tests, 5380 suite, 112 modules
- **`f812412 (w52-55)`** — feat(squash): W52-55 Squash Cloud dashboard API — 10 /cloud/* endpoints, JWT multi-tenant auth, model inventory, VEX alerts, drift events, policy dashboard, audit — 61 new tests, 5333 suite, 125 modules
- **`80492ee (w52)`** — feat(squash): W52 VEX feed subscription — api_key support + subscribe CLI + 25-statement community feed — 52 new tests, 5272 suite, 125 modules

---

## Module count
- **112** Python files in `squish/` (non-experimental). W56: aqlm.py extended in-place, no new files. Note: count dropped from 125 because SESSION.md had a stale value — actual count verified at 112 by test_module_count_unchanged gate.

---

## Open accuracy-validation items
- **W56 AQLM**: lm_eval on Qwen2.5-1.5B after `squish compress --format aqlm`. Target: <6pp arc_easy vs INT4 (baseline 70.6%). Expected to beat naive INT2 (−40.8pp) by large margin. Runtime ≈ 5-10 min compression + 20 min lm_eval.



---

---

## Quantization status (as of 2026-03-27, overnight bench results)

**⚠️ CORRECTION (2026-03-27 follow-up session):**
The overnight bench (`run_overnight_bench.py`) does NOT use `squish compress`. It calls
`mlx_lm.convert` directly with `q_group_size = {4: 64, 3: 32, 2: 64}`. Verified via:
- `config.json` in `~/models/Qwen2.5-1.5B-Instruct-int4`: `bits=4, group_size=64`
- `config.json` in `~/models/Qwen2.5-1.5B-Instruct-int3`: `bits=3, group_size=32`
- Source: `run_overnight_bench.py` line 259: `q_group_size = {4: 64, 3: 32, 2: 64}`

**⚠️ FORMAT DISCOVERY:**
`squish compress --format int4` / `--format mixed_attn` outputs squish `.npy-dir` format
which `mlx_lm.load()` CANNOT load. Only `squish compress --format int3` uses `mlx_lm.convert`
internally. The Wave 41 squish_lm_eval.py harness bridges this for INT4 AWQ npy-dir evaluation.

---

## Open questions / next priorities

1. **INT4 AWQ remaining 4 tasks (hellaswag/winogrande/piqa/openbookqa)**: May have finished
   in background terminal from W43. Check:
   `ls /Users/wscholl/squish/results/_tmp_Qwen2.5-1.5B-Instruct-int4-awq/`
   If done: update CLAUDE.md accuracy table TBD cells.

2. **mixed_attn lm_eval validation**: Code-complete (W41 harness). Still needs a measurement
   run. This is the unlock gate for INT2 AQLM / SpQR experimental paths.

3. **ADO Extension publishing**: To publish the W44 extension to ADO Marketplace:
   ```
   cd integrations/azure-devops
   tfx extension publish --manifest-globs vss-extension.json
   ```
   Requires: `AZURE_DEVOPS_EXT_PAT` env var (ADO PAT, Marketplace:Publish scope).
   Publisher: `squishai` must be registered at marketplace.visualstudio.com first.

4. **Wave 45 candidates**: Grafana/Prometheus metrics export from `squash attest` runs,
   or Datadog integration (follows same pattern as existing platform adapters).

