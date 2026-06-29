# Prefix-reuse fidelity & the chunked-prefill fix

## The problem
squish's in-memory prompt-prefix KV reuse (`squish/kv/prompt_prefix_cache.py`,
used by the default prompt-lookup decode path) lets a request that extends a recent
prompt skip re-prefilling the shared prefix. It is a real TTFT win, but for
**partial** reuse the greedy output was **not byte-identical** to a cold (no-reuse)
run: in a controlled bf16 test ~60% of 40-token generations forked to a different
(but equally valid) token.

## Root cause — bf16 rounding, not a bug
Proven empirically (not inferred):
- Equal-length tails still diverged (~0.5 logit); **identical** prompts were bit-exact
  (0.0). So it is the *suffix*, not a positional/off-by-one bug.
- Reuse prefills the new suffix in a **small variable-length forward**, whereas a cold
  run prefills it inside the **full-prompt forward**. The math is identical, but the
  two matmul *shapes* round differently in bf16 (~0.5 logit), enough to flip a genuine
  **near-tie** (top-2 margin ~0.1). After one flip, autoregressive generation cascades.
- It affects every partial-reuse path; only **exact-repeat (100%) reuse** is inherently
  lossless (no suffix to prefill).

fp32 compute eliminates it (Δlogit 0.5 → 2e-5; 60% → 0% divergence) but costs ~2× and
won't fit a 7B in fp32 weights on 16 GB — rejected.

## The fix — absolute-position-aligned chunked prefill (no fp32)
Prefill the suffix in **fixed-size, absolute-position-aligned chunks** of size
`_PREFILL_CHUNK` (`prompt_prefix_cache.py`), and align the reuse boundary **down** to a
chunk multiple. A token at absolute position `p` is then always computed in the chunk
`[⌊p/C⌋·C, …)` with the *same* matmul shape and the *same* prior KV whether or not its
prefix was cached → **byte-identical to cold**. This is how vLLM/PagedAttention get
exact prefix reuse; squish keeps full cross-chunk attention (shared accumulating cache),
so there is no attention-sink quality loss. Correctness validated: 0 divergence over 40
tokens across shared lengths on and off the chunk grid (`tests/test_prompt_prefix_cache.py`),
and on the int4 server across the previously-failing cases.

A guard caps reuse at the prior request's **prompt** length (decode/spec tokens are
written off the chunk grid and must not be reused) — see `PromptPrefixCache.store/borrow`.

## Chunk-size (C) tradeoff — measured (724-token int4 prompt, M3)
Two opposing effects: smaller C wastes fewer tokens at the alignment boundary (more
reuse) but processes the re-prefill in more, smaller forwards. Single-shot, noisy:

| C   | 50% overlap (reuse s / speedup) | 90% overlap (reuse s / speedup) | lossless |
|-----|---------------------------------|---------------------------------|----------|
| 256 | 8.39 / 1.46×                    | 3.93 / 2.73×                    | ✓        |
| 128 | 5.65 / 1.81×                    | 2.42 / 5.00×                    | ✓        |
| 64  | 6.10 / 1.72×                    | 2.10 / 5.06×                    | ✓        |
| 48  | 7.54 / 1.75×                    | 1.73 / 6.59×                    | ✓        |
| 24  | 8.24 / 2.09× (cold ballooned)   | 2.30 / 6.20×                    | ✓        |

Findings:
- **C=256 is clearly worst** — at 724 tokens its ≤255-token boundary waste throws away a
  large fraction of the reuse (re-prefills 212 tokens at 90% vs 84 for C=128).
- **Per-chunk overhead is negligible** until very small C: cold prefill time barely moves
  256→64, then balloons at C=24 (tiny forwards underutilize the GPU). So shrinking below
  ~48 backfires.
- **Best C is overlap-dependent:** larger C (128) wins at *low* overlap (big re-prefill →
  chunk overhead matters); smaller C (48–64) wins at *high* overlap (boundary waste matters).
- **Sweet spot: C = 64–128.** Correctness holds for any C (bit-exactness is shape-identity,
  independent of C). Default chosen in `prompt_prefix_cache.py::_PREFILL_CHUNK`.

## 64 vs 128 head-to-head (724-token int4, 19-overlap grid, single rep — noisy)
Reuse latency (s, lower = better), averaged by overlap band to cut single-rep noise:

| overlap band | C=64 | C=128 | winner |
|--------------|------|-------|--------|
| low  (5–25%) | 9.40 | 11.85 | **C=64** (~20% faster) |
| mid  (30–65%)| 7.51 | 6.96  | C=128 (~7%) |
| high (70–95%)| 2.68 | 2.78  | ~tie (C=64 edge) |

C=64 wins 2 of 3 bands, decisively at low overlap (where C=128's larger boundary can drop
reuse to zero). **Default = C=64.** (Caveat: single-rep, ~40% baseline variance; a clean
multi-rep run would tighten the mid-band call. The head-to-head's *speedup* column was
discarded — a baseline bug measured exact-repeat reuse instead of a cold prefill; the
reuse latencies above are valid.)

Reproduce: vary `_PREFILL_CHUNK`, run a cold-vs-reuse timing on the int4 server (see the
ad-hoc harness pattern in `benchmarks/prefix_reuse_curve.py`).
