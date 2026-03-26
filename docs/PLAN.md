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

### 4.1 Mixed precision: INT2 FFN, INT4 attention

**Status:** NOT STARTED.

**Expected disk reduction:** 1.5B model: 900 MB → ~550 MB.

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
