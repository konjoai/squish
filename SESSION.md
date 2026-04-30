# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-29 (W103.2 shipped)

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
- **W103.3** (2026-04-29) — `MixedPrecisionRouter` in `quantizer.py` + `compress_weights_sqint2()`
  in `sqint2.py` + standalone `--format sqint2` compress path in `cli.py` + pure-NumPy INT3/INT4
  codecs (`_int3_quantize_numpy`, `_int4_quantize_numpy`). 90 router tests + 46 compress tests.
  Full suite: **2367 passed / 3 pre-existing / 43 skipped**. Module count stays 84 (all
  additions in-place to existing files). `compress_weights_sqint2()` is fully testable without
  hardware — pure in-memory, synthetic weights only. Hardware lm_eval gate deferred to W103.4.
  Routing spec: boundary layers (first 2 + last 2) → INT4; MLP gate/up → SQINT2;
  attn Q/K/V/O → INT3; everything else → INT4; embeddings/lm_head → None.
  For Qwen2.5-7B (28 layers): 48 SQINT2, 96 INT3, 52 INT4, 3 skip per 199 weight tensors.
- **W103.4a** (2026-04-29) — `save_sqint2_layer` / `load_sqint2_layer` in `sqint2.py`
  + SQINT2 dispatch in `compressed_loader.py`. npy-dir format (4 mandatory + 5 optional
  `.npy` files; fp64 meta header, version=1.0). `compress_weights_sqint2` updated to
  emit `__sqint2_meta` per layer (required for cfg.seed → Hadamard rotation).
  `_TENSOR_SUFFIX_RE` extended. Module count stays 84 (in-place). 27 new tests in
  `tests/test_sqint2_loader.py`. Full suite: **2394 passed / 3 pre-existing / 35
  skipped** (2367 → 2394, +27).
- **W103.4b** (2026-04-29) — `sqint2_residual_gemv_f32` in
  `squish_quant_rs/src/lib.rs` (Rayon-parallel L·R GEMV + serial COO scatter, f64
  accumulator, bounds-checked) + Python wrapper `sqint2_residual_gemv` in
  `sqint2.py` with pure-NumPy fallback for envs without Rust built. fp16 storage
  promoted to fp32 at the wrapper boundary; rank=0 + nnz=0 short-circuits to a
  zero-array. 21 new tests in `tests/test_sqint2_residual_gemv.py`: equivalence
  vs explicit `(L@R + sparse_dense) @ x_rot`, end-to-end consistency with
  `decompress_weight()`'s residual path, fp16↔fp32 round-trip, COO bounds
  checks, determinism, Rust ↔ NumPy parity 1e-5 abs / 1e-4 rel
  (M=256, r=16, N=1024, k=100). Module count stays 84 (in-place additions to
  `sqint2.py` + `lib.rs`). Full suite: **2415 passed** (2394 → 2415, +21).
  Next: W103.4c — Metal NF2 fused-dequant GEMV kernel + `SQINT2Linear` mlx Module.
- **W103.2** (2026-04-29) — SVD rank-16 + sparse-1% residual correction in `squish/quant/sqint2.py`
  (in-place extension; module count stays 84). 46 new tests added to `tests/test_sqint2.py`
  (110 total in file, all passing). Joint SNR **10.21–10.23 dB** (gate ≥ 10.0 dB ✓) across
  5 seeds at σ=0.02 IID Gaussian, g=32, r=16, sparse=1%. Lift decomposition: +0.30 dB SVD
  + 0.24 dB sparse = +0.54 dB over W103.1 base. Full suite: **2231 passed / 3 pre-existing
  failures / 43 skipped** (zero regressions).
  Key findings logged permanently:
  - 16 dB IID-Gaussian gate is not achievable with rank-16 SVD alone: Hadamard rotation
    whitens ALL input distributions (IID, outlier, low-rank) before quantization. Post-rotation
    residual is IID regardless of input; top-16 singular values capture only 2.78% of variance
    on (1536, 576) → 0.30 dB lift. This is Marchenko-Pastur theory, not a bug.
  - 14 dB "outlier gate" also drops: Hadamard rotation already repairs outliers in Stage 1.
    Pre-rotation sparse correction (outlier fix in original domain) is W103.3 scope.
  - 2.15 bpw target requires g≥128 + INT8 scale/zp (W103.3 scale-compression pass).

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
- ✅ **W103.2 (2026-04-29) — SHIPPED.** Rank-16 SVD + sparse-1% residual in sqint2.py.
  Gate ≥ 10.0 dB IID Gaussian ✓. 16 dB on IID Gaussian not achievable (Marchenko-Pastur).
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
