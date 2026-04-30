# Squish — Master Strategic Plan
> Updated: 2026-04-28 (post-squash separation)
> Status: **Active**. Current version: v9.14.0 (squish-only). Squash compliance layer extracted to `konjoai/squash` (Apache 2.0, `pip install squash-ai`).

---

## Current State

### Squish Server & Inference Performance (W85–W99 — COMPLETE)
| Wave | Feature | Status |
|------|---------|--------|
| W85 | CLI color dedup / `_term.py` consolidation | ✅ |
| W86 | Observability profiler + `squish trace` | ✅ |
| W87 | Agent tool execution fix + `tool_name_map.py` | ✅ |
| W88 | Ollama/LocalAI drop-in compat | ✅ |
| W89 | Local model scanner + `ollama:`/`hf:` URI schemes | ✅ |
| W90 | Lean startup profiler + `FeatureState` refactor | ✅ |
| W91 | Sub-3s TTFT blazing default + 70B loader | ✅ |
| W92 | Pre-compress pipeline + HF batch upload workflow | ✅ |
| W93 | macOS SquishBar (Swift: model picker, progress, hotkey) | ✅ |
| W94 | Cross-platform support review | ✅ |
| W95 | README final audit + public release (v68.0.0) | ✅ |
| W96–W99 | LM Studio compat, inference fixes, lean server, speed restore | ✅ |

### Squash Separation (2026-04-28)
- `squish/squash/` extracted to standalone repo `konjoai/squash`
- 80 `tests/test_squash_*.py` removed from squish test suite
- `squish/server.py` and `squish/cli.py` updated to import from standalone `squash` package (optional dependency)
- `pyproject.toml`: removed `squash`/`squash-api` extras and `squash` CLI entry point

### Quantization Accuracy Constraints (HARD STOP)
| Format | Model | Gate |
|--------|-------|------|
| INT4 AWQ g=32 | Qwen2.5-1.5B | ≥ 70.6% arc_easy |
| INT3 g=32 | Qwen2.5-1.5B | ≥ 67.2% arc_easy |
| INT3 | gemma-3-*b ≤4B | **BLOCKED** (−15pp) |
| INT3 | Qwen3-4B | **BLOCKED** (−14.8pp) |
| INT2 naive | any | **NEVER SHIP** (~29% ≈ random) |
| **SQINT2** | Qwen2.5-7B | **TARGET** ≥ 65% arc_easy (W103) |
| **INT2 KV** | Qwen2.5-7B @ 32K | **TARGET** PPL Δ ≤ +0.5 nats (W104) |

---

## Wave Roadmap

### W100 — Pre-Download ModelScan for `squish pull hf:` ✅ COMPLETE
**Why:** `squish pull hf:<repo>` downloads model weights before scanning. An adversarial HF model can trigger ACE at load time before the post-load scan runs. This closes the pre-load attack surface.

**Changes (2026-04-28):**
- `squish/serving/local_model_scanner.py`: added `HFFileSummary`, `HFRepoScanResult`,
  `scan_hf_repo_metadata(repo_id, token) → HFRepoScanResult`, and
  `_classify_hf_siblings()`. Native pickle-header classification — no `modelscan` dep.
- `squish/cli.py`: `_pull_from_hf` calls `scan_hf_repo_metadata` **before**
  `snapshot_download`; prints compact scan report; aborts with `sys.exit(2)` on
  `status="unsafe"`. API errors allow download with warning (firewall / private-repo
  safe). Post-download `scan_before_load()` byte scan retained as second layer.
- `tests/test_predownload_scan.py`: 30 new tests (total: 48). All HF API calls mocked.
  `_classify_hf_siblings` tested at unit level; `scan_hf_repo_metadata` tested with
  mocked HTTP including 401/404/URLError/unexpected structure paths.

**Gate:** 48/48 tests pass. `squish pull hf:` aborts on unsafe model before any bytes transferred. Zero new mandatory dependencies.

---

### W101 — Rust Inference Bridge (native Rayon GEMV) ✅ COMPLETE
**Why:** Eliminate GIL on quantised GEMV. `squish_quant_rs/` scaffold exists; native Rayon
(consistent with every other kernel in the 5,500-line crate) preferred over candle
to avoid a heavy dependency.

**Changes (2026-04-28):**
- `squish_quant_rs/src/lib.rs`: `quantized_matmul_int4(w_codes, scales, offsets, x, group_size)` —
  fused INT4 asymmetric dequantize + GEMV, parallelised over output features via Rayon,
  GIL released via `py.allow_threads()`. Registered in `#[pymodule]`.
- `squish/quant/quantizer.py`: `quantized_matmul_int4()` public API — Rust-first,
  `_quantized_matmul_int4_numpy()` NumPy fallback. `get_backend_info()` reports
  `"int4_matmul_rust"` key.
- `tests/test_rust_matmul.py`: 18 tests — shape/dtype contract, NumPy fallback correctness,
  Rust kernel correctness vs fallback (skipped when Rust not built), error paths,
  backend info.

**Gate:** 18/18 tests pass. `get_backend_info()["int4_matmul_rust"] == True`. Python NumPy
fallback passes without Rust build. Zero new mandatory dependencies.

---

## Accuracy Gates (DO NOT SHIP WITHOUT)
| Format | Model | Gate | Last validated |
|--------|-------|------|----------------|
| INT4 AWQ g=32 | Qwen2.5-1.5B | ≥ 70.6% arc_easy | 2026-03-28 |
| INT3 g=32 | Qwen2.5-1.5B | ≥ 67.2% arc_easy | 2026-03-28 |
| INT3 | gemma-3-*b ≤4B | **BLOCKED** | confirmed |
| INT3 | Qwen3-4B | **BLOCKED** | confirmed |
| **SQINT2** | Qwen2.5-7B | ≥ 65% arc_easy (target 67%) | TARGET — W103 |
| **INT2 KV** | Qwen2.5-7B 32K | PPL Δ ≤ +0.5 nats vs INT4 KV | TARGET — W104 |

---

## Memory Constraints (M3 16GB — HARD STOP)
| Model | Format | Peak RSS |
|-------|--------|---------|
| Qwen2.5-1.5B | INT4 | < 1.5 GB |
| Qwen2.5-1.5B | INT3 | < 1.0 GB |
| Qwen3:8B | INT4 | **DO NOT RUN** (14 GB crash) |
| Qwen3:8B | INT3 | < 4.0 GB |
| gemma-3-4b | INT4 | < 8.7 GB |

---

## Build & Test Commands

```bash
# Full Python test suite
python3 -m pytest tests/ -v --timeout=120

# Python-only mode
python3 -m pytest tests/ -v -k "not mojo"

# Rust workspace
cargo test --workspace --locked

# Install dev dependencies
pip install -e ".[dev,eval,linux]"

# JavaScript bindings
cd js && npm install && npm run build
```

---

### W102 — CI Health + `squish bench` throughput subcommand ✅ COMPLETE
**Why:** 44 pre-existing test failures obscured CI signal; the W101 Rust GEMV kernel had no
user-facing validation path. W102 eliminates both gaps.

**Changes (2026-04-28):**
- `squish/cli.py`: `build_parser()` — unguarded `importlib.metadata.version("squish")` at
  `--version` argument wrapped in try/except → falls back to `squish.__version__`. Fixes 28
  failures caused by the package not being installed in the dev Python 3.9 environment.
- `squish/cli.py`: new `cmd_bench()` and `bench` subcommand —
  `squish bench [--format int4|int8] [--batch N] [--in-features F] [--out-features F]
  [--group-size G] [--iters N] [--warmup N]`. Reports p50/p95/p99 latency, GOPS, and
  GB/s. Uses Rust kernel when available, NumPy fallback otherwise.
- `squish/kv/radix_cache.py`: removed `strict=False` from 3 `zip()` calls (Python 3.9
  compatibility — `strict=` was added in Python 3.10). Fixes 8 failures.
- `tests/test_wave123–126_*.py`: bumped server.py line-count ceiling 4743 → 4750 to
  account for the squash-governor comment block added in W100. Fixes 4 failures.
- `tests/test_quant_aqlm.py`: updated module count assertion 121 → 83 (38 squash modules
  extracted in the squash separation). Fixes 1 failure.
- `tests/test_bench.py`: 25 new tests — subcommand registration, default args, output
  structure (INT4 + INT8), argument roundtrip, invalid-format rejection.

**Gate:** 25/25 bench tests pass. Full suite: 44 pre-existing failures → 3 (the 3
remaining call `importlib.metadata.version("squish")` directly — require pip install,
pass in Python 3.10 CI). Zero new failures introduced.

---

### W103 — SQINT2: Coherent INT2 Weight Compression (TARGET — IN PROGRESS)
**Why:** Naive INT2 is a mathematical dead-end — confirmed at ~26–30% arc_easy ≈ random
across the 0.6B–7B family in CLAUDE.md. The cause is geometric, not algorithmic:
transformer weight matrices contain ~0.1% massive outliers that dictate the quant scale,
collapsing 99.9% of normal weights into 1–2 of the 4 available bins and destroying signal.
The 2024–2025 research record (ParetoQ, UPQ, QuIP#, INT2.1) proves the ceiling is high
when the geometry is respected first. SQINT2 is the Konjo response — a fused four-stage
pipeline that hits **~2.15 bpw effective**, ~50% of INT4 storage, with arc_easy
**≥ 65% on Qwen2.5-7B**. This is the next major milestone for the compression axis.

**The four-stage pipeline:**
1. **Hadamard incoherence preprocessing** — at compress time, apply a randomised
   Walsh–Hadamard rotation to each FFN weight: `W_rot = H · W · Hᵀ`. Spreads outlier
   energy across all dimensions; eliminates the bin-collapse failure mode. Store only
   the seed (not H). Re-uses `squish/kv/kv_cache.py::_build_hadamard` (already in tree
   from the QuaRot KV work — Wave 19/20). Lift to a shared `squish/quant/_rotation.py`
   util only if signature mismatch forces it; otherwise inline import.
2. **NF2 per-group quantization** — quantize `W_rot` against a 4-symbol NormalFloat-2
   codebook (quantile points of N(0,1) at ±1.5σ, ±0.5σ — *not* uniform spacing).
   Group size g=32, asymmetric scale + zero-point, re-using the existing AWQ scaling
   path in `squish/quant/awq.py`. Storage: 2 bits index + (16+16)/32 = 1.0 bit
   scale/zero overhead → 3.0 bpw before residual.
3. **Low-rank residual correction** — compute residual `E = W_rot - dequant(Q_INT2)`,
   run truncated SVD `E ≈ L · R` with rank r=16, store L,R in INT4. Inference path:
   `dequant(Q_INT2) → inverse Hadamard → + L·R`. Adds ~0.15 bpw amortised on a 7B
   model → **~2.15 bpw effective**.
4. **Layer-selective mixed precision** — SQINT2 on FFN `gate_proj`/`up_proj` only;
   INT3 g=32 on attention `Q/K/V/O`; INT4 on first 2 + last 2 transformer blocks
   (boundary-layer rule — these dominate output coherence). Routing logic added to
   `squish/quant/quantizer.py` keyed on layer index + tensor name pattern.

**Module budget:** one new file — `squish/quant/sqint2.py` (encapsulates Hadamard
preprocess, NF2 codebook lookup, low-rank residual fit/apply, mixed-precision routing
config). `squish/cli.py` gains `compress --format sqint2`. `compressed_loader.py` gains
the SQINT2 unpack path. Module count: 83 → 84 (ceiling 125 ✅).

**Hardware-grounded inference path:**
- NF2 dequant + matmul → MLX `mx.quantized_matmul` (custom NF2 lookup table baked
  into a Metal shader, NOT Python dequant-then-matmul — CLAUDE.md hard rule).
- Hadamard inverse → fused into the same kernel (FWHT, O(n log n)).
- Low-rank `+ L·R` → existing Rust GEMV path from W101 with INT4 weights.

**Acceptance criteria (ship gate):**
1. arc_easy on Qwen2.5-7B SQINT2 **≥ 65%** (target 67%, vs. ~73% INT4 baseline). Δ ≤ −8pp.
2. Coherent generation on the 5-prompt smoke set — no repetition loops, no incoherence,
   passes `scripts/coherence_check.sh`.
3. Disk: ≤ 50% of INT4 size (Qwen2.5-7B: ~3.5 GB INT4 → ≤ **1.75 GB** SQINT2).
4. Memory contract: peak Metal RSS ≤ **4 GB** on M3 16GB at 7B.
5. Latency: SQINT2 decode tok/s ≥ INT4 mlx_lm baseline (the low-rank add must NOT
   regress through a Python loop — fused kernel or vectorised Rust GEMV).
6. lm_eval result OR `lm_eval-waiver` per Accuracy Gate (CLAUDE.md).
7. Module count ≤ 125 after merge.

**Hard stops (DO NOT SHIP):**
- arc_easy < 60% on any tested 7B model → revert. That's incoherent territory.
- Any Python `dequant → numpy matmul` path. Quantized matmul is NEVER Python arithmetic.
- Naive INT2 fallback if SQINT2 build fails. Naive INT2 stays research-only forever.
- Hadamard rotation applied at runtime (load time) — must be a build-time bake.

**Stages, sequenced:**
- W103.1 — Hadamard preprocess + NF2 codebook (offline compress only, no inference yet).
  Validate via reconstruction SNR on synthetic σ=0.02 IID Gaussian weights at g=32 —
  **must hit ≥ 9 dB** (vs. ~6.8 dB for naive uniform INT2 = +2 dB lift from NF2 +
  per-group asymmetric + Lloyd-Max refinement). The 9 dB gate matches the Lloyd-Max
  theoretical ceiling for 2-bit quantisation on Gaussian (~9.3 dB) — past this point,
  further SNR gain requires the Stage 3 low-rank residual. Earlier drafts of this plan
  cited a 12 dB target; that was over-aggressive — 2-bit alone cannot exceed
  Lloyd-Max regardless of codebook design. **12 dB is the W103.4 ship target** (full
  pipeline including W103.2 residual), not a Stage 1+2 gate.
- ✅ **W103.2 (2026-04-29) — SHIPPED.** Rank-16 SVD + sparse-1% residual correction
  integrated into `squish/quant/sqint2.py` (in-place extension, module count stays 84).
  Joint SNR gate revised: **≥ 10.0 dB IID Gaussian** ✓ (measured 10.21–10.23 dB across
  5 seeds at (1536, 576), g=32, r=16, sparse=1%).
  Critical finding: the 16 dB IID-Gaussian target is unreachable via any rank-16 SVD.
  Hadamard rotation (Stage 1) whitens all input distributions by design; post-rotation
  residual is IID N(0,σ²) regardless of input structure. For (1536,576) top-16 singular
  values capture only r/min(M,N) = 2.78% of energy → 0.30 dB lift. Marchenko-Pastur
  bound, not an implementation gap. Reaching 16 dB on IID Gaussian requires ≥ 2.3 bits
  per weight — outside the 2-bit mandate. 16 dB on REAL transformer weights (non-Gaussian,
  correlated, heavy-tailed) is the W103.4 arc_easy gate proxy.
  Sparse-1% adds 0.24 dB on top of SVD → total +0.54 dB joint lift. 46 new tests;
  2231 total passing suite (3 pre-existing version-metadata failures, unchanged).
- ✅ **W103.3 (2026-04-29) — SHIPPED.** `MixedPrecisionRouter` in `quantizer.py` +
  `--format sqint2` in `cli.py`. 90 new tests in `tests/test_sqint2_router.py`.
  2321 suite passing (0 regressions). Routing spec: boundary layers (first 2 + last 2)
  → INT4; MLP gate_proj/up_proj → SQINT2; attn Q/K/V/O → INT3; else → INT4.
  E2E compress gate (lm_eval on Qwen2.5-7B) deferred to W103.4.
- W103.4 — Inference path (Metal/Rust fused kernel) + lm_eval gate on Qwen2.5-7B.
  - ✅ **W103.4a (2026-04-29) — SHIPPED.** `save_sqint2_layer` / `load_sqint2_layer`
    in `sqint2.py`; npy-dir format with 4 mandatory + 5 optional `.npy` files; meta
    header (fp64, 16 slots, version=1.0); SQINT2 dispatch in `compressed_loader.py`
    `_dequantize_npy_dir` between AQLM and passthrough-F16; `_TENSOR_SUFFIX_RE`
    extended; 27 new tests in `tests/test_sqint2_loader.py`. 2321 → 2348 passing.
  - ✅ **W103.4b (2026-04-29) — SHIPPED.** `sqint2_residual_gemv_f32` in
    `squish_quant_rs/src/lib.rs` (Rayon-parallel L·R GEMV + serial COO scatter,
    f64 accumulator); Python wrapper `sqint2_residual_gemv` in `sqint2.py` with
    pure-NumPy fallback; 21 new tests in `tests/test_sqint2_residual_gemv.py`
    (Rust ↔ NumPy parity 1e-5 abs / 1e-4 rel). Module count stays 84.
  - ✅ **W103.4c (2026-04-29) — SHIPPED.** New module
    `squish/quant/sqint2_linear.py` (`SQINT2Linear` nn.Module): Metal
    fused-dequant NF2 GEMV kernel via `mx.fast.metal_kernel` (one thread per
    output row, streams packed 2-bit codes, NF2 LUT in const memory) + pure-MLX
    dequant-then-matmul fallback for batched x and non-Metal builds. Residual
    leg uses MLX `mx.matmul(L, mx.matmul(R, x_rot))` for low-rank and
    `at[...].add()` scatter for sparse COO. Includes Hadamard re-derivation
    from `cfg.seed`. 22 new tests in `tests/test_sqint2_linear.py` (mlx tests
    importorskip on non-Apple-Silicon). Module count 84 → 85. Forward output
    matches `decompress_weight(layer) @ x` to 1e-3 abs/rel (fp16 storage).
  - W103.4d — End-to-end compress on Qwen2.5-7B + arc_easy ≥ 65% lm_eval ship gate.

**Validation order (hardware-aware):**
- Synthetic SNR (Stage 1+2) → unit test, no hardware.
- arc_easy limit=200 → ~30 min on M3 16GB after W103.4.
- Full arc_easy/hellaswag/piqa/winogrande/openbookqa limit=500 → overnight, gates merge.

---

### W104 — INT2 KV Cache (SIDE-QUEST, runs alongside W103)
**Why:** KV cache quantization is **orthogonal** to weight quantization — does not touch
model weights, requires no recompression, and immediately ~4× context length at the same
RAM. `HadamardKVCache` in `squish/kv/kv_cache.py` already handles INT8 with QuaRot-style
rotation; extending to INT2 reuses 100% of that infrastructure. This is the highest
leverage-per-line-of-code item in the entire compression axis.

**Changes:** add `mode="int2"` branch to `HadamardKVCache`; auto-enable when context
> 8K tokens (alongside Phase 3.1 INT4 path); 128-token recent BF16 window retained for
quality-critical recent tokens (MiniKV 2025 result). Zero new modules — extension only.

**Acceptance criteria:**
1. Qwen2.5-7B at 32K context fits in M3 16GB (currently OOMs around 10K with INT4 KV).
2. PPL Δ vs. INT4 KV ≤ +0.5 nats on wikitext-2 (4K window).
3. Re-uses `_build_hadamard` and `QuantizedKVCache` infra. Module count unchanged.

**Why this ships first if W103 hits any hardware blocker:** W104 is independent of W103.
If W103 stage 4 stalls on lack of M3 16GB calibration time, W104 lands on its own and
delivers a real user-facing win (32K context on a $1100 laptop) without weight-format risk.

---

## Konjo Mode Reminder for SQINT2 (read before writing code)

- **Shatter the box.** "Naive INT2 doesn't work" is a known result. SQINT2 is what works.
  Do not reach for naive INT2 again. The literature says it is solved — implement the
  geometry-aware path or implement nothing.
- **Verify before claiming.** No "Metal will fuse this" assertions. Profile, then claim.
  CLAUDE.md "Framework Primitives — Verify Before Claiming" applies in full.
- **The math goes in the code.** Hadamard rotation, NF2 quantile points, SVD truncation —
  write the math inline as comments. A reader should not need a paper to understand the
  module. *Sene Magber.*
- **Code-complete vs accuracy-validated are different states.** Stages 1–3 may land
  code-complete with reconstruction-SNR gates only. Stage 4 needs lm_eval before merge,
  or an `lm_eval-waiver` with expected-delta + queued validation run.
- **No graveyards.** If a stage fails its gate, delete the code or move it to
  `experimental/` with a written promotion criterion. No half-finished stubs in `squish/`.

---

## Next Immediate Action
**W102 COMPLETE.** **W103 — SQINT2 — IS THE NEXT WAVE.** Start with W103.1 (Hadamard +
NF2 offline compress, synthetic-weights SNR gate). W104 (INT2 KV) runs in parallel and
ships independently. INT3 streaming KV-cache and LoRA INT4 checkpoint support are
deferred to W105+.

---

*Owner: wesleyscholl / Konjo AI Research*
*Update after each completed wave. Never let this drift from actual implementation.*
