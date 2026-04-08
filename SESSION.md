# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-09

## Last commits
- **`W44 (pending)`** — feat(wave44): Azure DevOps SquashAttest@1 marketplace extension
- **`e93edf7 (w43)`** — feat(wave43): CircleCI Orb + Artifact Hub Helm chart publish workflows; 23 tests
- **`784df22 (w42)`** — fix(wave42): bias formula + norm exclusion + g=32 default; INT4 AWQ arc_easy=70.8% PASS
- `98885d0 (w41)` — feat(wave41): squish-native lm_eval harness — dev/benchmarks/squish_lm_eval.py (52 tests, 4642 suite)
- `d20b0ea` — feat(wave40): GCP Vertex AI integration — VertexAISquash platform adapter (24 tests)

---

## Module count
- **122** Python files in `squish/` (non-experimental). W44 added `azure_devops.py` (#122).
  Justification on file: completes CI/CD platform matrix (GitHub Actions, GitLab, Jenkins,
  Argo, CircleCI, Azure DevOps). No further additions planned for CI/CD tier.

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

