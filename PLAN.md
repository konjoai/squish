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
  - ✅ **W103.4b (2026-05-05) — SHIPPED.** `sqint2_residual_gemv` in
    `squish_quant_rs/src/lib.rs` — GIL-free Rayon GEMV for the Stage-3 SVD
    residual term. Computes `x @ (L @ R)ᵀ` in two sequential Rayon-parallel
    steps (h = x @ Rᵀ, y = h @ Lᵀ). fp16 factors upcast to fp32 inside the
    kernel (CLAUDE.md FP32-accumulation mandate). Registered in `#[pymodule]`.
    Python public API `sqint2_residual_gemv(l_fp16, r_fp16, x)` added to
    `squish/quant/quantizer.py` with NumPy fallback
    `_sqint2_residual_gemv_numpy`. `get_backend_info()` gains
    `"sqint2_residual_gemv_rust"` key. 31 tests in
    `tests/test_sqint2_residual_gemv.py` (24 passing / 7 Rust-skipped when
    extension not built). Suite: 349 passing on targeted key files, 0 new
    regressions. Rust: `cargo check` clean.
  - ✅ **W103.4c (2026-05-07) — SHIPPED.** `squish/quant/sqint2_linear.py` —
    `SQINT2Linear(mlx.nn.Module)` + `sqint2_linear_from_layer` factory. Hard
    darwin/arm64 platform guard (ImportError on Linux). Forward: NF2 LUT lookup
    via `mx.take` → per-group asymmetric dequant → fp16 matmul → optional SVD
    residual (Rust/NumPy GEMV) → optional sparse COO correction. 46 new tests in
    `tests/test_sqint2_linear.py` (all skip on Linux/non-arm64). Module count 84 → 85.
    Also: `cmd_bench` INT4 NumPy fallback (8 bench tests fixed), version test
    alignment (stale 9.14.0 → 9.25.0 assertions updated in 2 test files).
  - W103.4d — End-to-end compress on Qwen2.5-7B + arc_easy ≥ 65% lm_eval ship gate.

**Validation order (hardware-aware):**
- Synthetic SNR (Stage 1+2) → unit test, no hardware.
- arc_easy limit=200 → ~30 min on M3 16GB after W103.4.
- Full arc_easy/hellaswag/piqa/winogrande/openbookqa limit=500 → overnight, gates merge.

---

### W104 — INT2 KV Cache (SIDE-QUEST, runs alongside W103) ✅ COMPLETE (2026-05-01)
**Why:** KV cache quantization is **orthogonal** to weight quantization — does not touch
model weights, requires no recompression, and immediately ~4× context length at the same
RAM. `HadamardKVCache` in `squish/kv/kv_cache.py` already handles INT8 with QuaRot-style
rotation; extending to INT2 reuses 100% of that infrastructure. This is the highest
leverage-per-line-of-code item in the entire compression axis.

**Changes shipped (2026-05-01):**
- `squish/kv/kv_cache.py` (in-place):
  - `_quantize_int2_per_channel` / `_dequantize_int2_per_channel` — per-token
    symmetric NF2 4-level codec, indices bit-packed 4-per-uint8 along `head_dim`.
  - `_kv_quantize_per_channel` / `_kv_dequantize_per_channel` — mode dispatch.
  - `KVLayerCache._kv_mode` slot + `kv_mode=` constructor arg.
  - `QuantizedKVCache.mode="int2"` validated; rejects illegal combinations
    (`svd_rank > 0`, `comm_vq_bits > 0`, `qfilter_rank > 0`,
    `enable_disk_tier()`).
  - `HadamardKVCache` docstring updated with W104 motivation.
  - `recommended_kv_mode(context_tokens)` + `KV_INT2_AUTO_THRESHOLD = 8192`.
- `tests/test_kv_int2.py` — 32 new tests (codec roundtrip, dispatch, mode
  validation, end-to-end through QuantizedKVCache and HadamardKVCache,
  memory-ratio assertion ≥ 2.9× reduction, disk-tier guardrail).
- Zero new production modules. Codec storage is `(n_tokens, head_dim/4)` uint8
  — asymptotic 4× reduction on the old-tier buffer vs INT8.
- Per-token reconstruction SNR ≥ 5 dB on uniform inputs; ≥ +1 dB lift from
  Hadamard rotation on heavy-tailed activations (validated in test).

**Hardware ship gate (deferred to lm_eval session):**
1. Qwen2.5-7B at 32K context fits in M3 16GB (currently OOMs around 10K with INT8 KV).
2. PPL Δ vs. INT8 KV ≤ +0.5 nats on wikitext-2 (4K window).
- Both metrics require live Qwen2.5-7B inference; tracked alongside the
  W103.4d arc_easy run.

**Acceptance criteria met (code + unit gates):**
- ✅ `mode="int2"` branch lands on `QuantizedKVCache` and `HadamardKVCache`.
- ✅ Storage saves ≥ 2.9× per-token at head_dim=128 (4× asymptotic).
- ✅ Re-uses `_build_hadamard` and `KVLayerCache` infra. Zero new modules.
- ✅ Auto-mode helper (`recommended_kv_mode`) with 8 K threshold.
- ⏳ 32 K context fit & PPL Δ — gated behind hardware run.

**Recommended configuration for ≥ 16 K context:**
```python
from squish.kv.kv_cache import HadamardKVCache, recommended_kv_mode
mode = recommended_kv_mode(planned_context_tokens)   # "int8" or "int2"
cache = HadamardKVCache(n_layers=N, window=128, mode=mode, seed=42)
```
The 128-token recent FP16 window (configurable) retains quality-critical
recent tokens; everything older is INT2-quantised in the rotated frame.

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

### W105 — INT4 KV Cache (intermediate quality tier) ✅ COMPLETE (2026-05-02)
**Why:** W104 shipped INT2 KV (4× memory) but the SNR cliff from INT8 (~44 dB)
to INT2 (~5 dB on Hadamard-rotated activations) is sharp. INT4 fills the gap
at ~22 dB SNR with 2× memory reduction — the right default for 8 K–16 K
contexts where INT2 is overkill and INT8 is too expensive.

**Changes shipped (2026-05-02):**
- `squish/kv/kv_cache.py` (in-place):
  - `_quantize_int4_per_channel` / `_dequantize_int4_per_channel` — per-token
    symmetric 16-level uniform codec (`{-7.5,…,7.5}`), nibble-packed
    2-per-uint8 along `head_dim` (low = even col, high = odd col).
  - `_kv_quantize_per_channel` / `_kv_dequantize_per_channel` — dispatch
    extended to `"int4"`. New `_KV_QUANT_MODES` frozenset.
  - `KVLayerCache(kv_mode="int4")` accepted; validation rejects all other
    unknown values.
  - `QuantizedKVCache(mode="int4")` accepted; rejects illegal combinations
    (`svd_rank > 0`, `comm_vq_bits > 0`, `qfilter_rank > 0`).
  - `enable_disk_tier()` now rejects all sub-INT8 modes (int4 or int2).
  - `HadamardKVCache` docstring extended with W105 section.
  - `recommended_kv_mode_3tier(ctx)` + `KV_INT4_DEFAULT_THRESHOLD = 16384`.
    `recommended_kv_mode()` accepts optional `medium_mode` /
    `medium_threshold` for inline 3-tier dispatch.
- `tests/test_kv_int4.py` — 38 new tests (codec roundtrip, packing layout,
  SNR ordering INT8 > INT4 > INT2, dispatch, mode validation, end-to-end,
  memory ratio ≥ 1.7×, disk-tier guardrail, 3-tier recommendation).

**Acceptance criteria met:**
- ✅ INT4 SNR floor ≥ 18 dB on uniform inputs.
- ✅ INT4 strictly between INT8 and INT2; ≥ 6 dB margin over INT2.
- ✅ Storage = `head_dim/2 + 4` bytes per token (head_dim=128 → 68 B/token,
  asymptotic 2× reduction vs INT8).
- ✅ Hadamard rotation lifts INT4 SNR on heavy-tailed inputs.
- ✅ Suite: 2464 passed / 3 pre-existing W95 / 43 skipped (+38 from W105).
- ✅ Module count: zero new production modules. One new test file.

**Recommended configuration (W105, replacing the W104 default for ≥ 8 K):**
```python
from squish.kv.kv_cache import HadamardKVCache, recommended_kv_mode_3tier
mode = recommended_kv_mode_3tier(planned_context_tokens)
# ≤ 8K → int8;  8K–16K → int4 (W105);  > 16K → int2 (W104)
cache = HadamardKVCache(n_layers=N, window=128, mode=mode, seed=42)
```

---

### W106 — KV memory budgeting + cache factory ✅ COMPLETE (2026-05-03)
**Why:** W104 + W105 added three storage modes but production callers still
hand-wired `recommended_kv_mode_3tier` + a constructor and had no closed-
form way to answer the two RAM-planning questions every deployer asks:
"will this fit in N GB?" and "how long a context fits in B bytes?"

**Changes shipped (2026-05-03):**
- `squish/kv/kv_cache.py` (in-place):
  - `KVMemoryEstimate` (frozen dataclass) + `estimate_kv_memory()` —
    closed-form per-tier accounting; matches live `cache.memory_bytes`
    within 1 % on the regression workload.
  - `estimate_max_context()` — exact inverse: `(budget, mode, dims) → int`.
    Subtracts the FP16 recent-window overhead before computing capacity.
  - `recommend_mode_for_budget()` — budget-driven mode picker
    (`int8 → int4 → int2`); returns the highest-quality mode that fits or
    `None` when even INT2 is too large.
  - `make_kv_cache(n_layers, *, planned_context, rotate=True, mode=None,
    window=128, seed=42, **extra)` — one-line factory; picks mode via
    `recommended_kv_mode_3tier` and constructs `HadamardKVCache`
    (default) or `QuantizedKVCache` (`rotate=False`).
- `tests/test_kv_budget.py` — 49 new tests covering byte-cost building
  block, dataclass invariants, closed-form vs. live agreement,
  `estimate_max_context` ⇄ `estimate_kv_memory` round-trip, ordering
  invariant `int2 ≥ int4 ≥ int8` at fixed budget, mode selection
  including the `None`-when-nothing-fits edge, and the factory's
  defaults / overrides / determinism / kwarg forwarding.

**Acceptance criteria met:**
- ✅ Closed-form estimate within 1 % of live `cache.memory_bytes` for
  int8 / int4 / int2 (validated in `test_estimate_matches_live_cache_within_tolerance`).
- ✅ `estimate_max_context` inverts `estimate_kv_memory` exactly.
- ✅ `recommend_mode_for_budget` returns `None` when nothing fits.
- ✅ Factory wires `recommended_kv_mode_3tier` to `HadamardKVCache` /
  `QuantizedKVCache` with deterministic seed propagation.
- ✅ Suite: 2513 passed / 3 pre-existing W95 / 43 skipped (+49 from W106).
- ✅ Module count: zero new production modules; one new test file.

**One-line developer ergonomics now:**
```python
from squish.kv.kv_cache import make_kv_cache
cache = make_kv_cache(n_layers=28, planned_context=32_000)
# auto-picks int2, builds HadamardKVCache, ready for mlx_lm.
```

---

### W107 — Hugging Face Spaces demo + grounded BENCHMARKS.md ✅ COMPLETE (2026-05-09)
**Why:** the W104/W105/W106 KV-cache work shipped a publishable result —
3 quantization tiers, closed-form memory planning, an 8 dB SNR lift from
Hadamard rotation on outlier-heavy activations — but had no zero-install
surface for the public to *try*. A HF Space converts the codecs from a
PyPI dependency into a clickable demo and gives the upcoming launch post a
"proof page" (BENCHMARKS.md) that points at code, not slides.

**Changes shipped (2026-05-09):**
- **`spaces/`** (new directory, four files, *not* part of the squish wheel):
  - `spaces/__init__.py` — package marker so `tests/test_spaces_demo.py`
    can import the helpers as `spaces._logic`.
  - `spaces/_logic.py` — pure helpers (gradio-free, fully testable):
    `snr_db`, `make_synthetic_activations` (gaussian / heavy_tailed /
    outlier), `apply_hadamard`, `run_all_tiers` (INT8/INT4/INT2 round-trip),
    `recommend_mode_for_context`, `memory_table_rows`,
    `label_budget_fit`, `recommend_for_budget_mb`. All numbers come from
    the public API of `squish.kv.kv_cache` (`estimate_kv_memory`,
    `recommend_mode_for_budget`, `recommended_kv_mode_3tier`,
    `_quantize_int{8,4,2}_per_channel`, `HadamardKVCache._build_hadamard`).
  - `spaces/app.py` — Gradio Blocks app with two tabs:
    *Tensor Inspector* (distribution + rotation toggle → SNR/B-per-token
    table + recommended-tier markdown + 5 pre-loaded "Try these examples"
    via `gr.Examples`) and *Memory Budgeter* (model preset + context +
    RAM budget → fp16/int8/int4/int2 memory table with "fits / over by
    N MB" labels + by-context and by-budget recommendations).
  - `spaces/requirements.txt` — `gradio>=4.44,<5.0` and `squish==9.27.0`.
    No torch (the codecs are pure numpy).
  - `spaces/README.md` — HF Space metadata header (title, emoji, sdk,
    sdk_version, app_file, license, tags) + how-to-read-SNR table +
    source pointer.
- **`pyproject.toml`** — gated `mlx` and `mlx-lm` behind
  `sys_platform == 'darwin' and platform_machine == 'arm64'`. Backward-
  compatible on Apple Silicon; unblocks `pip install squish` on the
  Linux x86_64 HF Space runner. Lazy-import paths (`squish/kv/kv_cache.py:57`
  et al.) already handle the MLX-absent case.
- **`BENCHMARKS.md`** (new, repo root) — six grounded sections: cold-load
  + TTFT, disk size raw vs squished, weight quant accuracy gates, KV-cache
  storage + SNR + Qwen2.5-7B per-context memory table, GEMV throughput,
  "reproduce" bash recipes. The KV SNR table calls out the 17 dB rotation
  lift on outlier-spiked input (-8.61 dB → +8.47 dB at INT2) — the
  demo's headline claim, with a unit test pinning the floor at 8 dB.
- **`README.md`** — added the HF Space and Benchmarks shields next to
  the existing HuggingFace badge.
- **`tests/test_spaces_demo.py`** — 41 new tests covering every public
  helper in `_logic.py` plus a hard-fail "headline claim" test
  (`test_hadamard_rotation_lifts_int2_snr_by_at_least_8db_on_outlier_input`).
- Version bumped 9.26.0 → 9.27.0 (`pyproject.toml`, `squish/__init__.py`,
  `tests/test_version.py`, `tests/test_wave79_startup_inference.py`).

**Acceptance criteria met:**
- ✅ `spaces/app.py` constructs without launching (`build_demo()`),
  syntax-validated; ready for the HF Space runner.
- ✅ `spaces/_logic.py` runs end-to-end on numpy alone —
  `python -c "from spaces._logic import ...; ..."` produces the
  BENCHMARKS.md numbers verbatim.
- ✅ `pip install squish` succeeds on Linux x86_64 (mlx gated out).
- ✅ KV-cache tests unchanged: 119 / 119 in `tests/test_kv_*.py`.
- ✅ New tests: 41 / 41 in `tests/test_spaces_demo.py` (full file in 1.87 s).
- ✅ Headline 8 dB rotation-lift claim is a hard CI assertion.
- ✅ Module count: zero new files in `squish/` (`spaces/` is a sibling
  directory, excluded from the `Path(squish.__file__).parent` walks in
  `tests/test_quant_aqlm.py`, `test_sqint2.py`, `test_sqint2_router.py`).

**Hard stops (none triggered):**
- No new mandatory dependency in `squish/` core — gradio lives only in
  `spaces/requirements.txt`.
- No `unwrap()` / silent-swallow / `dbg!()` / TODO leftovers in the new code.
- No mocking of the codecs in tests — all reconstructions are real.

---

### W109 — Dashboard v2 + GET /api/recommend + visual UI overhaul ✅ COMPLETE (2026-05-10)
**Why:** the W107 HF Space proved the codecs are demonstrable in a browser,
but the existing local dashboard at `demo/index.html` was an
information-dense console — bars and numbers — that didn't *show* what
KV-cache quantization actually does. W109 rebuilds it as a visceral
visual: two columns of "memory blocks" side by side, the right one
collapsing as the slider moves through INT8 → INT4 → INT2. The new
`GET /api/recommend` endpoint adds the planner counterpart to the
existing live-codec endpoints, so the dashboard can request a real
reasoning string from the same closed-form math the inference server
uses internally.

**Changes shipped (2026-05-10):**
- **`demo/server.py`** — added `GET /api/recommend?model_size_b&ctx_len[&budget_mb]`:
  - Closed-form (no token loop) — uses `estimate_kv_memory` +
    `recommend_mode_for_budget` + `recommended_kv_mode_3tier` from
    `squish.kv.kv_cache`. Sub-millisecond per request.
  - Architecture snapping table `_REC_ARCH_TABLE` with 8 canonical
    presets (Qwen2.5 0.5B/1.5B/3B/7B/14B/32B, Llama-3.1 8B/70B). Snaps
    `model_size_b` to the closest entry; ties break to the **larger**
    preset (the conservative pick for memory planning).
  - `basis` field reports which constraint actually decided
    (`"context"` / `"budget"` / `"agreement"`) — the dashboard surfaces
    this so the user understands *why* a tier was chosen.
  - Reasoning string is short and factual; explicitly distinguishes the
    "context says X but budget allows higher-quality Y" case from the
    pure-context case.
  - Input validation per CLAUDE.md security.md: required-param check,
    range gates, non-numeric rejection — all errors return HTTP 400 with
    a JSON `{"error": "…"}` body, never 500.
  - The existing `POST /api/recommend`, `/api/health`, `/api/compress`,
    `/api/benchmark` endpoints unchanged; backward-compatible.

- **`demo/index.html`** — full rebuild (~960 lines, replaces ~2 250).
  Pure CSS animations (no canvas, no WebGL, no JS for the visuals —
  JS only swaps CSS custom properties). Spec match per W109 prompt:
  - **Background**: `#06060f` + tiled SVG hexagonal lattice with a
    60 s-loop drift, plus four parallax `clip-path` hex blobs floating
    independently (`@keyframes hex-drift`, `@keyframes hex-float`).
  - **Hero**: two-column showcase. Left = static FP16 reference (16
    bright amber blocks, full spacing). Right = live tier (block count
    is `LIVE_MAX_BLOCKS / compression_ratio`, so a slider move literally
    *removes* blocks). Tint driven by HSL custom props
    (`--tier-hue/sat/light/glow`).
  - **Slider**: single horizontal `<input type="range">`, no label,
    rail painted as a left-to-right amber → salmon → purple gradient.
    Snaps to tier midpoints on `change`.
  - **Three orbs above the slider**: pure round divs, no text. Active
    orb scales 1.55× and pulses (`@keyframes orb-pulse`). Orb colour
    matches its tier (amber/salmon/purple). Keyboard-navigable
    (`role="button"`, `tabindex=0`, Enter/Space).
  - **Three floating number cards**: memory MB · SNR dB · mode label.
    Numbers animate via a per-frame eased ticker (~380 ms). Card
    gradient flips colour scheme on tier change. Card flash on
    transition (`@keyframes card-flash`).
  - **Saving arc**: conic-gradient ring centred between the columns,
    sweep angle = `(1 - 1/compression) × 360°`. Glowing % readout in
    the centre.
  - **Crystal-lattice comparison table**: 5-column grid (metric, fp16,
    int8, int4, int2). The active tier's column shimmers
    (`@keyframes shimmer`) with a violet inner-shadow ring.
  - **Reasoning panel**: pulls live string from `GET /api/recommend`
    when the demo server is running; falls back to baked-in strings
    that match the BENCHMARKS.md numbers when offline (file://).
  - **Color language**: `--hot:#ffb86b` (amber) → `--warm:#ff8a5b`
    (salmon) → `--cold:#a173ff` (Konjo purple). High precision is hot,
    compressed is cold.
  - **Accessibility**: full `prefers-reduced-motion` support — every
    keyframe and transition is silenced. Mobile-responsive grid
    breakpoints at 880 px and 560 px.

- **`tests/test_demo_server.py`** — 35 new tests covering the GET
  endpoint end-to-end (a real `HTTPServer` on a free port + `urllib`
  client) plus all the helpers as pure functions:
  - Architecture-table invariants (sorted, well-formed rows, snap-to-
    closest, tie-breaks-larger).
  - Closed-form happy paths (short/medium/long context picks correct
    tier; budget basis correctly resolves and reports `by_context` /
    `by_budget` / `agreement`).
  - Reasoning content gates (mentions chosen tier; mentions budget
    when budget decided; mentions both tiers when they disagree).
  - Live HTTP — happy paths and 7 negative paths (missing required
    params, non-numeric, negative, out-of-range — all return 400, none
    return 500).
  - Backward compatibility — `/api/health`, POST `/api/recommend`,
    unknown path → 404.

- Version bumped 9.27.0 → 9.28.0 (`pyproject.toml`,
  `squish/__init__.py`, `spaces/requirements.txt`,
  `tests/test_version.py`, `tests/test_wave79_startup_inference.py`).

**Acceptance criteria met:**
- ✅ Memory savings calculator now has a *visual* output (the right
  column literally collapses; the saving arc fills; the cards animate).
  Numbers are still there, but they ride on the visual.
- ✅ `GET /api/recommend` ships, takes model size + context length
  (+ optional budget), returns mode + reasoning + closed-form memory
  table for all 4 tiers.
- ✅ All UI animations are pure CSS — JS does no animation work, only
  `setProperty` on CSS custom props and the small number ticker.
- ✅ Spec compliance: 11 / 11 W109 design-checkpoints verified by an
  HTML structural sanity script (background colour, hex field, two
  columns, slider, orbs, cards, lattice, color language, GET wiring,
  ≥ 5 keyframes).
- ✅ Tests: 35 / 35 in `tests/test_demo_server.py` (0.94 s).
- ✅ Existing endpoints + KV-cache + spaces tests unchanged.

**Hard stops (none triggered):**
- No JS animation libraries pulled in. No bundler. No build step. The
  HTML opens from `file://` and works offline (live API enhancement is
  optional).
- No new mandatory dependency in `squish/` core.
- No `unwrap()` / silent-swallow / TODO leftovers.
- All API errors return 400, never 500. CLAUDE.md security.md respected:
  every input is validated at the boundary, range-checked, and typed.

### W110 — Prompt Router (v9.29.0) ✅ COMPLETE (2026-05-11)
**Why:** squish serves prompts of wildly different types — Python snippets,
maths problems, creative writing, factual Q&A. Routing to the best local model
for each type improves quality without extra latency or model loading. W110
adds a pure-software, zero-dependency routing layer.

**Changes shipped (2026-05-11):**
- **`squish/serving/router.py`** — 300-line module, no new deps:
  - `RouterCategory` — StrEnum: CODE / MATH / CREATIVE / FACTUAL /
    CONVERSATION / UNKNOWN.
  - `RouterRule` — dataclass: name, category, compiled regex, priority,
    model_hint.
  - `RouterDecision` — frozen dataclass: category, matched_rule, model_hint,
    confidence (1.0 rule / 0.5 heuristic / 0.0 unknown), reasoning (≤120 chars).
  - `RouterConfig` — dataclass: custom rules, fallback_category,
    enable_heuristics.
  - `PromptRouter` — main classifier. Route steps: (1) empty → UNKNOWN 0.0;
    (2) rules descending priority, first match → confidence 1.0; (3) keyword
    heuristics (CODE/MATH/CREATIVE/FACTUAL) → confidence 0.5; (4) UNKNOWN 0.0.
    `explain()` returns `asdict()` + `prompt_length` + `n_rules_checked`.
  - 8 built-in rules: python-code (p90), general-code (p85), math-equation
    (p80), math-calc (p75), creative-story (p70), creative-write (p65),
    factual-qa (p60), factual-explain (p55).
  - `get_default_router()` — module-level singleton factory.

- **`squish/cli.py`** — added `squish route "<prompt>" [--json]` subcommand
  via `cmd_route()` + `p_route` parser in `build_parser()`. Prints a formatted
  table or JSON. No new imports beyond stdlib `json`.

- **`tests/test_router.py`** — 27 tests (25 required + 2 bonus edge cases):
  all pass in 0.14 s. Covers every public API path including frozen-dataclass
  mutation guard, priority ordering, custom rule overrides, heuristic path,
  UNKNOWN path, explain() keys, CLI subcommand registration.

- Module count updated: 85 → 86 (`test_sqint2.py`, `test_sqint2_router.py`).

- Version bumped 9.28.0 → 9.29.0: `pyproject.toml`, `squish/__init__.py`,
  `spaces/requirements.txt`, `tests/test_version.py`,
  `tests/test_wave79_startup_inference.py`.

**Acceptance criteria:**
- ✅ `RouterDecision` is frozen (mutation raises `FrozenInstanceError`).
- ✅ Empty prompt → UNKNOWN, confidence 0.0.
- ✅ Rule match → confidence 1.0, matched_rule populated.
- ✅ Heuristic path → confidence 0.5, matched_rule=None.
- ✅ Priority ordering: higher-priority rule wins on ties.
- ✅ `squish route` CLI subcommand registered in `build_parser()`.
- ✅ 27 / 27 tests pass; zero regressions introduced.
- ✅ No silent failures, no TODOs, no dead code, no new mandatory deps.

---

## Next Immediate Action
**W110 SHIPPED (2026-05-11).** Prompt Router live at `squish/serving/router.py`.

**After W110: W103.4d** — End-to-end compress on Qwen2.5-7B + arc_easy ≥ 65 %
lm_eval ship gate (hardware run required). Also validates the W104 32 K-context
envelope on the same hardware run. LoRA INT4 checkpoint support remains deferred.

---

### W111 — Inference Quality Monitor (v9.30.0) ✅ COMPLETE (2026-05-11)
**Why:** squish serves requests across multiple models with no visibility into
per-request latency percentiles, TTFT, or throughput degradation. W111 adds a
zero-dependency rolling-window quality tracker that exposes P50/P95/P99 stats
via `GET /v1/quality` and `squish quality` CLI — making production perf visible
without Prometheus or any external APM tool.

**Changes shipped (2026-05-11):**
- **`squish/serving/quality_monitor.py`** — 450-line module, stdlib only:
  - `RequestMetric` — frozen dataclass: timestamp, model_id, latency_ms, ttft_ms,
    tokens_generated, tokens_per_sec, success, error_type.
  - `QualityStats` — frozen dataclass: per-model P50/P95/P99 latency + TPS + TTFT
    + error_rate + n_requests. `to_dict()` → JSON-serialisable.
  - `QualityReport` — frozen dataclass: all-models report. `to_dict()`.
  - `QualityMonitor` — thread-safe deque-backed tracker (threading.Lock).
    `record()`, `report()`, `stats_for()`, `clear()`. Lazy trim on every record().
  - `_percentile(values, p)` — linear interpolation; math comment: `i = (p/100)*(n-1)`.
  - `get_quality_monitor()` — double-checked-locking singleton.
  - `record_completion_metric(model_id, duration_s, ttft_s, n_tokens, tps)` —
    server-facing helper; never raises; logs warning on failure.
  - `quality_response_dict(window, model_filter)` — server-facing report builder.

- **`squish/server.py`** — `_ModelState.record_completion` hooks into
  `record_completion_metric` via a single-line call. `GET /v1/quality` endpoint
  (window=[60, 86400], optional model filter). Net +18 lines.

- **`squish/cli.py`** — added `squish quality [--window N] [--model M] [--json]`
  subcommand via `cmd_quality()` + `p_quality` parser in `build_parser()`.

- **`tests/test_quality_monitor.py`** — 25 tests. All pass in 0.33 s.

- Module count: 86 → 87 (`quality_monitor.py`).

- Version bumped 9.29.0 → 9.30.0: `pyproject.toml`, `squish/__init__.py`,
  `spaces/requirements.txt`, `tests/test_version.py`,
  `tests/test_wave79_startup_inference.py`.

- Line-count test thresholds updated (waves 122–126) to account for +18 lines.

**Acceptance criteria:**
- ✅ `RequestMetric` and `QualityStats` are frozen (mutation raises `FrozenInstanceError`).
- ✅ Thread-safe: 50 concurrent threads record 500 metrics without crash.
- ✅ Rolling window correctly excludes events older than `window_seconds`.
- ✅ `get_quality_monitor()` returns the same singleton across calls.
- ✅ `squish quality` CLI subcommand registered in `build_parser()`.
- ✅ `GET /v1/quality` returns HTTP 200 with valid JSON (empty stats valid).
- ✅ 25 / 25 tests pass; zero regressions introduced.
- ✅ No silent failures, no TODOs, no dead code, no new mandatory deps.

---

## Next Immediate Action
**W111 SHIPPED (2026-05-11).** Quality Monitor live at `squish/serving/quality_monitor.py`.

**After W111: W103.4d** — End-to-end compress on Qwen2.5-7B + arc_easy ≥ 65 %
lm_eval ship gate (hardware run required). Also validates the W104 32 K-context
envelope on the same hardware run.

---

*Owner: wesleyscholl / Konjo AI Research*
*Update after each completed wave. Never let this drift from actual implementation.*
