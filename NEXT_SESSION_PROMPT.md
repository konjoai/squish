# Next Session Prompt — W53

## Context
W52 is complete and committed. VEX subscription infrastructure is live:
- `squash vex subscribe / unsubscribe / list-subscriptions` works end-to-end
- `SQUASH_VEX_API_KEY` auth propagates through VexCache → VexFeed.from_url
- Community feed expanded to 25 statements (v1→v2)
- 5272 tests pass, 125 modules

## W53 Candidate: squash lineage sign (Sigstore cosign integration)

**Goal:** `squash lineage sign <model_dir>` — sign the CycloneDX BOM (or the
weight-hash manifest) with Sigstore's keyless signing workflow (cosign). Produce
a `.sig` and `.cert` alongside the BOM, compatible with `cosign verify`.

**Acceptance criteria (define before writing any code):**
1. `squash lineage sign <model_dir> [--bom BOM_PATH] [--quiet]` exits 0.
2. Writes `<bom_path>.sig` and `<bom_path>.cert` next to the BOM.
3. `squash lineage verify <model_dir> [--bom BOM_PATH]` exits 0 on valid sig, 1 on invalid.
4. No new Python module (inline into `squash/lineage.py` or `squash/cli.py`).
5. Graceful degradation: if `cosign` binary not found → exit 2 with clear error.
6. All tests pass.

**Alternative W53 candidate:** drift-check REST endpoint (`POST /drift-check`).
This was the original W52 proposal before the VEX subscription backlog item arrived.

## Open questions
- Sigstore vs standalone Ed25519 keygen (offline/air-gapped) — pick one approach before coding.
- Module count: confirm 125 before adding anything.

## lm_eval status
- INT4 AWQ g=32 (squish): 70.8% arc_easy (Qwen2.5-1.5B, limit=500, partial — 2 tasks pending)
- INT3: 67.2% arc_easy — below 72% gate; "efficient" tier only
- gemma INT3 ≤4B: UNSAFE (−15–16pp)
- Qwen3-4B INT3: UNSAFE (−14.8pp)
