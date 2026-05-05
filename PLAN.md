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
  - W103.4c — Metal NF2 fused-dequant GEMV kernel + `SQINT2Linear` mlx Module.
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

## Next Immediate Action
**W103.4b SHIPPED (2026-05-05).** `sqint2_residual_gemv` Rust GEMV is now on
`main`. The Rust path is gated behind `hasattr(_squish_quant, "sqint2_residual_gemv")`
— NumPy fallback is always available.

**Next: W103.4c** — Metal NF2 fused-dequant GEMV kernel + `SQINT2Linear` MLX Module.
This requires Apple Silicon with MLX (`platform.machine() == "arm64"`, mlx installed).
Implementation scope:
  - `squish/quant/sqint2_linear.py` (new file, gated `if sys.platform == "darwin"`)
    `SQINT2Linear(mlx.nn.Module)` — holds packed indices + scales + zp as `mx.array`;
    forward pass: NF2 dequant → matmul → add residual via `sqint2_residual_gemv`.
  - Metal shader or `mx.quantized_matmul` with custom NF2 lookup table (baked LUT).
  - Gate: `tests/test_sqint2_linear.py` (skip on Linux / non-arm64).

**After W103.4c: W103.4d** — End-to-end compress on Qwen2.5-7B + arc_easy ≥ 65%
lm_eval ship gate (hardware run required). Also validates the W104 32K-context
envelope on the same hardware run. LoRA INT4 checkpoint support remains deferred.

---

*Owner: wesleyscholl / Konjo AI Research*
*Update after each completed wave. Never let this drift from actual implementation.*
