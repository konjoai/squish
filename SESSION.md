# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-29 (W102 shipped; W103 SQINT2 starting)

## Last commits
- **`f109942`** — docs(squash): update compliance section to point to standalone konjoai/squash repo
- **`75935cb`** — feat(squash): W82 HQQ float bits + W83 NIST AI RMF 1.0 controls
- **`ec2bdf3`** — feat(squash): W81 remediation plan generator

## This session (2026-04-28)
- Removed `squish/squash/` module from squish — now standalone at `konjoai/squash` (`pip install squash-ai`)
- Deleted 80 `tests/test_squash_*.py` test files
- Updated `squish/server.py` and `squish/cli.py` to import from standalone `squash` package (optional; try/except guarded)
- Updated `pyproject.toml`: removed `squash`/`squash-api` extras and `squash` CLI entrypoint
- Updated `tests/test_cli_eval.py`, `test_cli_sbom.py`, `test_sbom_builder.py`, `test_eval_binder.py`, `test_oms_signer.py`, `test_governor_middleware.py` to use `squash.*` imports with `pytest.importorskip` guards

---

## Module count
- Python files in `squish/` (excluding experimental/__pycache__): **84** (post-W103.1).
  - Squash separation (2026-04-28): 112 → 68 raw / 83 with experimental excluded.
  - W103.1 (2026-04-29): 83 → 84 (+`squish/quant/sqint2.py`).
- Ceiling: 125 (CLAUDE.md). Headroom: 41.

---

## Open accuracy-validation items
- **W56 AQLM**: lm_eval on Qwen2.5-1.5B never run. Not blocking current work.
- **mixed_attn lm_eval**: Code-complete (W41), still unvalidated. Not blocking.
- **W103 SQINT2**: design-complete, implementation starting. arc_easy gate ≥ 65% on Qwen2.5-7B.

---

## ✅ Recently shipped
- **W100** (2026-04-28) — Pre-download HF model scanner (48/48 tests; pre-load ACE surface closed).
- **W101** (2026-04-28) — Rust GIL-free INT4 fused dequantize + GEMV (`squish_quant_rs`); 18/18 tests.
- **W102** (2026-04-28) — `squish bench` throughput subcommand + Python 3.9 CI repair (44 → 3 failures).
- **W103.1** (2026-04-29) — `squish/quant/sqint2.py` (Hadamard + NF2 + per-group asymmetric +
  Lloyd-Max refinement); 64/64 new tests in `tests/test_sqint2.py`. SNR **9.69 dB** at g=32
  on σ=0.02 IID Gaussian, +2.86 dB over naive INT2 baseline (6.83 dB). Gate revised
  9 dB (was 12 dB — past the 2-bit Lloyd-Max ceiling for Gaussian). Module count 83 → 84.
  Zero new failures in full suite (2185 passed; 3 pre-existing `importlib.metadata`
  failures unchanged).

---

## Next wave: W103 — SQINT2 (Coherent INT2 Weight Compression)

**Goal:** Ship coherent INT2 — ~2.15 bpw effective, ≥ 65% arc_easy on Qwen2.5-7B,
~50% INT4 disk. Naive INT2 is a confirmed dead end (~29% ≈ random); SQINT2 is the
geometry-aware path: Hadamard incoherence + NF2 codebook + low-rank residual + layer-
selective mixed precision.

**Sequenced stages:**
- ✅ **W103.1 (2026-04-29) — SHIPPED.** `squish/quant/sqint2.py`: Hadamard preprocess
  + NF2 codebook + per-group asymmetric + Lloyd-Max refinement + 2-bit packing.
  **Gate revised to ≥ 9 dB** on σ=0.02 IID Gaussian at g=32 (12 dB was past the
  2-bit Lloyd-Max ceiling on Gaussian — Stage 1+2 alone physically cannot exceed
  it; 12 dB is reserved for the W103.4 full-pipeline ship gate after the Stage 3
  low-rank residual lands). **Measured: 9.69 dB** across 5 seeds, +2.86 dB over
  naive INT2's 6.83 dB. 64/64 new tests in `tests/test_sqint2.py`; 2185 total
  pass (3 pre-existing `importlib.metadata` failures unchanged). Module count 84
  (CLAUDE.md ceiling 125). Reuses Hadamard construction from
  `squish/kv/kv_cache.py` (Wave 19/20 QuaRot infra) — lifted as a standalone
  function in sqint2.py rather than cross-module imported.
- **W103.2** — Low-rank residual SVD fit (rank=16) + INT4 storage of L,R. Gate: joint SNR ≥ 16 dB.
- **W103.3** — Layer-selective routing in `squish/quant/quantizer.py` + `compress --format sqint2`
  CLI. Gate: E2E compress on Qwen2.5-1.5B; disk ≤ 50% of INT4.
- **W103.4** — Fused inference path (Metal NF2 + Rust low-rank GEMV via W101 path);
  `compressed_loader.py` SQINT2 unpack. Gate: arc_easy ≥ 65% on Qwen2.5-7B; tok/s ≥ INT4 baseline.

**Module count budget:** 83 → 84 (one new file `squish/quant/sqint2.py`; ceiling 125 ✅).

**Hard stops (DO NOT SHIP):**
- arc_easy < 60% on any 7B → revert.
- Python `dequant → numpy matmul` anywhere on the SQINT2 path.
- Runtime Hadamard application — must be a build-time bake.
- Naive INT2 fallback. SQINT2 only.

**Parallel side-quest — W104 INT2 KV cache:** Add `mode="int2"` to existing
`HadamardKVCache` in `squish/kv/kv_cache.py`. Zero new modules. Auto-enable at
context > 8K. Target: Qwen2.5-7B at 32K context on M3 16GB (currently OOMs ~10K with
INT4 KV).

**Reference:** Full W103 spec in `docs/PLAN.md` Phase 4.1; ship-gate criteria in `PLAN.md` W103.
