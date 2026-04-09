# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-09

## Last commits
- **`(w52 — pending push)`** — feat(squash): W52 VEX feed subscription — api_key support + subscribe CLI + 25-statement community feed — 52 new tests, 5272 suite, 125 modules
- **`(w51 — pending push)`** — feat(squash): W51 SBOM drift detection — check_drift + squash drift-check CLI — 54 new tests, 5220 suite, 125 modules
- **`(w50 — pending push)`** — feat(squash): W50 shadow AI detection — K8s pod scanner scan_pod_for_model_files + ShadowAiScanner + squash shadow-ai scan CLI — 65 new tests, 5166 suite, 124 modules
- **`(w49 — pending push)`** — feat(squash): W49 air-gapped/sovereign AI mode — _is_offline + keygen/sign_local/verify_local/pack_offline + CLI + REST — 68 new tests, 5101 suite, 124 modules
- **`(w48 — pending push)`** — feat(wave48): model transformation lineage chain — LineageChain(Merkle) + squash lineage CLI + /lineage REST — 69 new tests, 5033 suite (0 failures), 124 modules
- **`(w47 — pending push)`** — feat(wave47): RAG KB integrity scanner — RagScanner(index/verify) + scan-rag CLI + /rag/index /rag/verify REST — 57 new tests, 4964 suite (0 failures), 123 modules
- **`ed27727 (w46)`** — feat(wave46): agent audit trail — AgentAuditLogger JSONL hash chain + SquashAuditCallback + CLI + REST — 66 new tests, 4907 suite (0 failures), 122 modules

---

## Module count
- **125** Python files in `squish/` (non-experimental). W51: `squash/drift.py` +1 (justified: new security domain — SBOM drift detection, CMMC/EU AI Act/DoD IL4/IL5). W50: no new modules. W48: `squash/lineage.py` +1 (justified: EU AI Act Annex IV). W47: `squash/rag.py` +1. W45: `mcp.py` +1, `eval_binder.py` −1 = net zero.

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

