# Next Session Prompt — W57

## Context
W56 is complete. AQLM encode path is live:
- `AQLMEncoder` K-means codebook training in `squish/quant/aqlm.py`
- `squish compress --format aqlm` CLI wired (K=2, C=256, G=8 → ≈2 bpw default)
- sklearn MiniBatchKMeans fast path; pure-NumPy fallback
- npy-dir format: `__aqlm_idx.npy` + `__aqlm_cb.npy` + passthrough + `squish.json`
- 47 new tests, 5380 suite, 112 modules
- **lm_eval PENDING**: validate Qwen2.5-1.5B AQLM vs INT4 (target <6pp arc_easy delta)

## IMMEDIATE: lm_eval validation (accuracy gate)
```bash
# 1. Compress (≈5-10 min on M3 with sklearn):
squish compress --format aqlm ~/.cache/huggingface/hub/Qwen2.5-1.5B-Instruct --output /tmp/qwen2.5-1.5b-aqlm

# 2. Run lm_eval (≈20 min):
python3 scripts/squish_lm_eval.py --model-dir /tmp/qwen2.5-1.5b-aqlm --tasks arc_easy --limit 200

# 3. Compare vs baseline (70.6% arc_easy INT4)
```
Target: ≥64.6% arc_easy (< 6pp delta). Beats naive INT2 (≈27-30%). If it passes, promote AQLM to "ultra" catalog tier.

## W57 Candidate A: AQLM dequantize inference path
If AQLM validates, wire `squish/loader.py` to detect `__aqlm_idx.npy` and route through `aqlm_dequantize` at load time. This unblocks `squish serve` with AQLM models.

## W57 Candidate B: Cloud API SQLite persistence
Replace in-memory deques with SQLite-backed stores. `SQUASH_CLOUD_DB` env var selects path; default `:memory:` preserves existing test behavior.

## State
- 5380 tests pass, 112 modules, commit pending push from W56
- lm_eval-waiver filed for W56 (compression runtime > session budget)

1. All 61 W52-55 tests still pass with `:memory:` backend.
2. `SQUASH_CLOUD_DB=/tmp/squash.db squish serve` survives a restart and retains data.
3. No new Python module (inline into `squish/squash/api.py` or a new `squish/squash/cloud_db.py` with written justification).

## W56 Candidate C: drift-check REST endpoint

`POST /drift-check` — REST wrapper around the existing `check_drift(bom_a, bom_b)` logic.

## Open questions
- Which candidate is highest value? B (persistence) unblocks production deployment.
  A (Sigstore) delivers supply chain signing compliance. C is the simplest.
- Dashboard Next.js repo: do we scaffold it this session or stay API-only?

## lm_eval status
- INT4 AWQ g=32 (squish): 70.8% arc_easy (Qwen2.5-1.5B, limit=500, partial — 2 tasks pending)
- INT3: 67.2% arc_easy — below 72% gate; "efficient" tier only
- gemma INT3 ≤4B: UNSAFE (−15–16pp)
- Qwen3-4B INT3: UNSAFE (−14.8pp)
