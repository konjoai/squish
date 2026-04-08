# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-10

## Last commits
- **`(w47 — pending push)`** — feat(wave47): RAG KB integrity scanner — RagScanner(index/verify) + scan-rag CLI + /rag/index /rag/verify REST — 57 new tests, 4964 suite (0 failures), 123 modules
- **`ed27727 (w46)`** — feat(wave46): agent audit trail — AgentAuditLogger JSONL hash chain + SquashAuditCallback + CLI + REST — 66 new tests, 4907 suite (0 failures), 122 modules
- **`(w45 — pending push)`** — feat(wave45): MCP server manifest attestation — McpScanner + McpSigner + attest-mcp CLI + POST /attest/mcp — 110 tests, 0 failures, 122 modules
- **`bf4a5eb (w44)`** — feat(wave44): Azure DevOps SquashAttest@1 marketplace extension
- **`e93edf7 (w43)`** — feat(wave43): CircleCI Orb + Artifact Hub Helm chart publish workflows; 23 tests
- `98885d0 (w41)` — feat(wave41): squish-native lm_eval harness — dev/benchmarks/squish_lm_eval.py (52 tests, 4642 suite)

---

## Module count
- **123** Python files in `squish/` (non-experimental). W47: `squash/rag.py` +1 (pre-authorised slot from W45). W45: `mcp.py` +1, `eval_binder.py` −1 = net zero.

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

