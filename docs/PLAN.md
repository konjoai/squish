# Squish — Next Level Plan

> **Status:** Living document. Updated after each wave.
> **Source:** [squish_next_level.html](../squish_next_level.html) — the full research-backed plan.

---

## Target Metrics (M3 16GB)

| Metric | Now | Target |
|---|---|---|
| Tok/s — qwen2.5:1.5b INT4 | ~65–90 | **200–350** |
| TTFT — qwen2.5:1.5b INT4 | ~300–500 ms | **< 50 ms** |
| Peak RSS — 1.5b INT4 | ~1.5–3 GB | **< 600 MB** |
| Disk — 1.5b INT4 | ~900 MB | **< 400 MB** |
| Tok/s — qwen3:8b INT4 | ~14–22 | **40–70** |
| TTFT — qwen3:8b INT4 | ~443–535 ms | **< 150 ms** |

Physics ceiling (no spec decode): ~130 tok/s for 1.5B INT4 at 100 GB/s M3 bandwidth.
With EAGLE-3 at 75% acceptance: ~350 tok/s ceiling.

---

## Phase 1 — Foundation (Weeks 1–2)

Close the gap to the physics ceiling. No new techniques — validate and activate what already exists.

### 1.1 `mx.compile()` on the forward pass ✅ DONE

**Status:** Active on KV-cache decode path (server.py line 2028–2034). Added to `SpeculativeGenerator` in Wave 110 (`self._target_compiled`).

**Verified:** `--no-compile` flag disables. Manual decode path is the O(n²) fallback — growing input shapes make compile non-beneficial there.

**Expected gain:** +20–40% decode throughput.

---

### 1.2 Vectorize `_pack_codes_uint32` ✅ DONE (Wave 109)

**Status:** `np.add.at` scatter vectorization shipped. Python loop eliminated.

**Expected gain:** INT3 first-load for 8B model: 30s → < 3s.

---

### 1.3 Chunked prefill as default ✅ DONE

**Status:** `_chunk_prefill_enabled = not args.no_chunk_prefill`. Since `--no-chunk-prefill` defaults to False, chunked prefill IS on by default. Auto-enabled in `--blazing` mode.

**Config:** threshold=512 tokens, chunk_size=512. Use `--no-chunk-prefill` to disable.

**Expected gain:** TTFT on 4K-token prompts: 5–20s → 200–500ms.

---

### 1.4 Verify INT4 inference path ✅ VERIFIED

**Status:** `_build_squish_4bit_dir` writes `"quantization": {"bits": 4, "group_size": N}` to config.json, then `mlx_lm.load()` automatically uses `nn.QuantizedLinear` for all linear layers. No BF16 materialization occurs.

**Verification method:** Code path traced. Runtime type check (`type(model.layers[0].self_attn.q_proj)`) should return `nn.QuantizedLinear` on any squish_4bit/ model.

---

### 1.5 RadixTree prefix cache — KV-level sharing

**Status:** ❌ PARTIALLY DONE — gap identified.

- `_prefix_cache` (text-level exact-match, RadixTree-backed): **already default** with 512 entries. Hits on IDENTICAL prompt text. No model call needed on hit.
- `_radix_attn_cache` (`--radix-attn`, RadixAttentionCache): currently **numpy-only simulation**. Does NOT store or restore MLX KV tensors. Making this default would be a no-op.

**Real work needed for Phase 1.5 KV-level sharing:**
The actual implementation requires storing layer KV tensors (K,V arrays per layer) in a radix tree keyed by token prefix hash. On cache hit, restore KV state and prefill only the suffix. This is essentially Phase 3.2 work (SSD-backed paged KV).

**Near-term win:** The text-level exact-match cache effectively caches complete responses for identical prompts. For agent workloads with the same system prompt, Phase 3.2 (SSD KV) is the real solution.

---

## Phase 2 — Speculative Decoding (Weeks 3–8)

### 2.1 EAGLE-3 as the primary, always-on decode path

**Status:** ✅ VERIFIED CORRECT; ✅ N-gram fallback WIRED (Wave 110)

**Tree verification confirmed:** `_decode_multi_cached` submits all K draft tokens in a single `[1, K]` forward pass. The acceptance loop then processes logits sequentially. This IS the correct batched EAGLE-3 algorithm.

**N-gram fallback (Wave 110):** `SpeculativeGenerator._ngram_only_spec_stream` now active by default on all spec decode requests (including no-EAGLE-head case). Enabled via `_rebuild_spec_gen()` creating a generator even without a draft model. Disable with `--no-ngram-spec`.

**Expected gain (n-gram only, no EAGLE):** 1.3–1.8× tok/s on code/doc tasks.
**Expected gain (EAGLE-3 at 75% acceptance):** 2–3× tok/s.

**Remaining Phase 2.1 work:**
- Pre-built EAGLE-3 heads for qwen2.5:1.5b, qwen3:4b, qwen3:8b in HuggingFace catalog
- Benchmark EAGLE-3 acceptance rate on Squish INT4 models vs. standard heads

### 2.2 Train Squish-native EAGLE-3 heads

**Status:** NOT STARTED. Requires GPU training environment (~4–8h per model on RTX 3090).

**Why it matters:** Heads trained on INT4-quantized Squish models have higher acceptance rates than heads trained on BF16 and applied to INT4. 3–5% higher acceptance = 10–20% additional throughput on top of EAGLE-3 base.

---

## Phase 3 — KV Cache (Weeks 7–10)

### 3.1 INT4 KV cache quantization (KIVI) as default for context > 4K tokens

**Status:**  PARTIAL — `QuantizedKVCache` exists in `squish/kv/kv_cache.py`. Not auto-enabled.

**Work needed:** Auto-enable when conversation exceeds 4K tokens. Keep 128-token BF16 window for recent tokens.

**Expected gain:** 4× context length at same RAM. 8B model: 8K → 32K context on M3 16GB.

### 3.2 Paged KV cache with SSD cold tier

**Status:** NOT STARTED (infrastructure exists: `PagedKVCache` in `squish/kv/paged_attention.py`).

**Why this is the TTFT killer:** SSD at 7 GB/s means 100 MB KV state restores in 15ms vs 300ms+ recompute. For agents with shared system prompts, this is the real Phase 1.5 win.

**Expected gain:** Repeat-prefix TTFT: 300–500ms → 15–50ms. 10–60× for agent workloads.

### 3.3 Continuous batching with shared KV prefix blocks

**Status:** NOT STARTED.

---

## Phase 4 — Compression (Weeks 11–16)

### 4.1 SQINT2 — Coherent INT2 weight compression (W103, Phase 4 anchor)

**Status:** ✅ DESIGN-COMPLETE — IMPLEMENTATION STARTING (W103). Replaces the prior
"INT2 FFN, INT4 attention" sketch with a real four-stage pipeline grounded in the
2024–2025 research record.

**Why naive INT2 fails (the floor):** Transformer weight matrices contain ~0.1% massive
outliers. With only 4 bins available at 2 bits ({−3, −1, +1, +3} or NF2), the outliers
dictate the quant scale, collapsing 99.9% of normal weights into 1–2 bins. Signal is
destroyed; output becomes incoherent. Our own benchmarks confirm this:
~26–30% arc_easy across 0.6B–7B (≈ random). Naive INT2 is a mathematical dead-end.

**Why SQINT2 works (the ceiling):** Respect the geometry first. Spread the outliers
*before* you quantise (Hadamard incoherence). Place the quantisation grid on the
distribution's quantiles, not uniform points (NF2). Patch the residual error with a
tiny low-rank correction (INT2.1). Use mixed precision so the layers that dominate
output coherence stay at higher bitwidth. Result: ~2.15 bpw effective, ≥ 65% arc_easy
on Qwen2.5-7B (vs. ~73% INT4 baseline, ≈ −8pp).

#### Research basis

| Paper | Year | Key result | SQINT2 stage |
|---|---|---|---|
| **ParetoQ** | Feb 2025 | 2-bit QAT closes to 3.4pp gap from full-precision | overall ceiling proof |
| **UPQ** (Universal Progressive Quant.) | Jun 2025 | FP16 → INT4 → INT2 staged quant cuts Frobenius error 42% vs FP16 → INT2 direct | informs Stage 2 ordering |
| **QuIP#** | 2024 | Randomized Hadamard transform spreads outliers; enables 2-bit at near-FP16 quality | Stage 1 (Hadamard) |
| **INT2.1** | 2023 | `W ≈ Q_INT2 + L · R` with rank-r FP16 residual matrix produces coherent text from PTQ INT2 | Stage 3 (residual) |
| **MiniKV** | 2025 | INT2 KV cache, 128-token BF16 recent window, 4× context at same RAM | side-quest (Phase 3.1 ext.) |
| **NormalFloat (NF4 → NF2)** | QLoRA 2023 + extension | Codebook on quantile points of N(0,1), not uniform spacing — better matches transformer weight distribution | Stage 2 (NF2 codebook) |

#### The four-stage pipeline

**Stage 1 — Hadamard incoherence preprocessing (compress-time only):**

```
W_rotated = H · W · Hᵀ    where H is a randomised Walsh–Hadamard matrix
                            of dimension matching W.shape[1]
```

`H` is power-of-two-sized; we re-use the existing `_build_hadamard(dim, rng)` from
`squish/kv/kv_cache.py:1354` (already shipped for QuaRot KV in Wave 19/20). Only the
*seed* is stored, not H itself — H is reconstructed on load. FWHT is O(n log n) and
fuses into the Metal shader for inference (no full materialisation of H at runtime).

**Stage 2 — NF2 per-group quantization:**

NF2 codebook (4 symbols) sits on the quantile points of N(0,1):

```
NF2 = { −1.5060, −0.5004, +0.5004, +1.5060 }   # normalised units
```

These are NOT `{−3, −1, +1, +3}`. The quantile placement matches the actual weight
distribution of `W_rotated` (Hadamard rotation makes weights ~Gaussian by CLT). Group
size g=32, asymmetric scale + zero-point reusing the AWQ pipeline in
`squish/quant/awq.py`.

Storage per group: 32 weights × 2 bits = 64 index bits, + 16-bit scale + 16-bit
zero-point = 96 bits / 32 weights = **3.0 bpw** before residual.

**Stage 3 — Low-rank residual correction:**

```
E = W_rotated − dequant(Q_INT2)         # NumPy at compress time
U, Σ, Vᵀ = svd(E)                       # truncated to rank r=16
L = U[:, :16] · diag(Σ[:16])            # shape (rows, 16)
R = Vᵀ[:16, :]                          # shape (16, cols)
```

Store `L`, `R` quantised to INT4. On a 7B model with ~30 layers × 4 FFN matrices
of shape ~(4096, 14336), residual storage amortises to ~0.15 bpw → **2.15 bpw effective**.

Inference path:

```
W_recon ≈ inv_hadamard( dequant_NF2(Q_INT2) ) + L · R
```

The `+ L · R` term is a rank-16 outer-product update — uses the W101 Rust GEMV path
with INT4 weights. Cost is negligible (16 × col-dim FLOPs).

**Stage 4 — Layer-selective mixed precision:**

| Layer / tensor | Precision | Reason |
|---|---|---|
| First 2 transformer blocks | INT4 (g=32) | Embedding-adjacent — dominate semantic encoding |
| Last 2 transformer blocks | INT4 (g=32) | Output-adjacent — dominate token-distribution shape |
| Attention `Q/K/V/O` (middle layers) | INT3 (g=32) | High sensitivity in rotation/attention math |
| FFN `gate_proj`, `up_proj` (middle layers) | **SQINT2** | High redundancy — safest INT2 surface |
| `down_proj` (middle layers) | INT3 (g=32) | Aggregates FFN output — keep one rung up |
| `norm`, `embed`, `lm_head` | BF16 | Small, scale-critical, never quantise |

Routing logic added to `squish/quant/quantizer.py` keyed on `layer_idx` + tensor name
regex. Driven by a config dataclass in `squish/quant/sqint2.py::SQINT2Config`.

#### Implementation plan (W103, sequenced)

| Stage | Deliverable | Gate | Hardware needed |
|---|---|---|---|
| W103.1 | `squish/quant/sqint2.py` — Hadamard + NF2 offline compress | reconstruction SNR **≥ 9 dB** on σ=0.02 IID Gaussian at g=32 (matches 2-bit Lloyd-Max ceiling; +2 dB over naive INT2's 6.8 dB) | none (numpy) |
| W103.2 | Low-rank residual fit + storage layout | joint SNR ≥ 16 dB (residual is what closes the gap from 9 dB → 12 dB) | none (numpy) |
| W103.3 | Layer-selective routing + `squish compress --format sqint2` CLI | E2E compress of Qwen2.5-1.5B succeeds; disk ≤ 50% of INT4 | none |
| W103.4 | Fused inference path (Metal NF2 + Rust low-rank); `compressed_loader.py` SQINT2 unpack | arc_easy ≥ 65% on Qwen2.5-7B; tok/s ≥ INT4 baseline | M3 16GB, overnight |

> **Konjo note on the W103.1 gate revision.** The first draft of this plan cited a
> 12 dB W103.1 target. That was over-aggressive: 2-bit quantisation of IID Gaussian
> weights is bounded by the Lloyd-Max theoretical ceiling at ~9.3 dB regardless of
> codebook design — no amount of Hadamard rotation or per-group scaling can beat it
> for a Gaussian source. NF2 + per-group asymmetric + Lloyd-Max refinement reaches
> 9.69 dB measured, +2.86 dB over the naive uniform-INT2 baseline of 6.83 dB. That
> +2 dB lift IS the Stage 1+2 win — it proves the codebook + scaling are correctly
> placed. The 12 dB target stays intact for **W103.4 full-pipeline ship gate**, where
> the Stage 3 low-rank residual covers the remaining ~3 dB. Surface trade-offs, then
> make a call (CLAUDE.md "Konjo Pushback Mandate").

#### Module budget

- **One new file:** `squish/quant/sqint2.py`. Module count 83 → 84 (ceiling 125 ✅).
- **Modified:** `squish/quant/quantizer.py` (mixed-precision routing), `squish/cli.py`
  (`--format sqint2`), `squish/quant/compressed_loader.py` (SQINT2 unpack path),
  `squish/kv/kv_cache.py` (lift `_build_hadamard` to importable export OR re-import
  inline in `sqint2.py`).

#### Acceptance criteria (ship gate, mirrors `/PLAN.md` W103)

1. arc_easy on Qwen2.5-7B SQINT2 **≥ 65%** (target 67%).
2. Coherent generation on the 5-prompt smoke set.
3. Disk ≤ 50% of INT4 (~1.75 GB on Qwen2.5-7B).
4. Peak Metal RSS ≤ 4 GB on M3 16GB at 7B.
5. Decode tok/s ≥ INT4 mlx_lm baseline.
6. lm_eval result OR `lm_eval-waiver` per CLAUDE.md Accuracy Gate.
7. Module count ≤ 125 after merge.

#### Hard stops (DO NOT SHIP)

- arc_easy < 60% on any 7B model. That is incoherent.
- Python `dequant → numpy matmul` for NF2. Quantized matmul is NEVER Python arithmetic.
- Hadamard applied at runtime instead of compress time. Build-time only.
- Naive INT2 fallback if SQINT2 build fails — naive INT2 stays research-only forever.

#### Side-quest — INT2 KV cache (W104, parallel)

`squish/kv/kv_cache.py::HadamardKVCache` already supports `mode="int8"`. Add `mode="int2"`
on the same code path. Auto-enable on context > 8K tokens (alongside Phase 3.1 INT4 KV
default). Retain 128-token recent BF16 window per MiniKV 2025. Zero new modules.

**Acceptance:** Qwen2.5-7B fits at 32K context on M3 16GB; PPL Δ ≤ +0.5 nats vs. INT4 KV
on wikitext-2.

#### What we are NOT doing in W103/W104

- **Not** training QAT INT2 (ParetoQ-style). Hardware budget too high. SQINT2 is PTQ.
- **Not** shipping AQLM/QuIP# as user-facing options (they remain research; SQINT2
  subsumes their value via Hadamard + NF2 + residual).
- **Not** wavelet-domain INT2 or stochastic-dithering NF2. Promising paths, but
  speculative — graveyard risk. Park in `experimental/` only with a promotion criterion.

#### Disk savings (concrete)

| Model | INT4 disk | SQINT2 disk (target) | Δ |
|---|---|---|---|
| Qwen2.5-1.5B | ~900 MB | ~480 MB | −47% |
| Qwen2.5-7B | ~3.5 GB | ~1.75 GB | −50% |
| Qwen3-8B | ~4.0 GB | ~2.0 GB | −50% |
| gemma-3-4b | ~2.0 GB | n/a | gemma family INT3-unsafe at ≤4B; SQINT2 blocked at ≤4B until ≥7B Gemma data confirms safety |

### 4.2 AWQ as default quantization path

**Status:** ❌ NOT DEFAULT — `squish compress --awq` exists but AWQ is opt-in. Naive INT4 is the default.

**Work needed:** Make AWQ the default in `squish compress`. Add `--no-awq` to skip it.

**Why it matters:** AWQ INT4 ≈ naive INT8 quality. AWQ INT3 ≈ naive INT4 quality. Real coherence gain.

### 4.3 `squish sparsity-trim` command

**Status:** NOT STARTED. `squish gen-masks` (sparse mask generation) exists.

### 4.4 ANCF v2 — Metal-native on-disk format

**Status:** NOT STARTED.

---

## Completed Waves

| Wave | Key Changes |
|---|---|
| Wave 109 | INT3Linear BF16 fix, 30 shim upgrades, vectorize `_pack_codes_uint32`, remove aiofiles |
| Wave 110 | `mx.compile` in SpeculativeGenerator, `_ngram_only_spec_stream` (Phase 2.1), `--no-ngram-spec` flag |

---

## Rules (from squish_next_level.html)

1. **One technique, one benchmark, one merge.** Run `scripts/run_baseline.sh` before and after every change.
2. **Verify before claiming.** No fusion/zero-copy claims without profiler evidence.
3. **100-file hard cap.** squish/ (non-experimental) stays under 100 Python files.
4. **Memory contract is law.** qwen2.5:1.5b INT4: peak Metal RSS < 1.5 GB; qwen3:8b INT4: < 6 GB.
5. **Quantized matmul is never Python arithmetic.** All quantized layers use `mx.quantized_matmul()` or `nn.QuantizedLinear`.

---

## The Demo Goal

An agent running qwen3:8b INT4 with EAGLE-3 on M3 16GB, writing and editing code, at **50–70 tok/s**, with **TTFT under 200ms** including SSD prefix cache on repeat turns. Everything in this plan drives toward that demo.
