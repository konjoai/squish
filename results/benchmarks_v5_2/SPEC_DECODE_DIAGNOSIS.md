# v5.2 — Speculative-decoding diagnosis

**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Date:** 2026-06-02
**Question:** Why does loading a draft model produce no measured warm-tok/s gain?

## TL;DR

Speculative decoding is **fully implemented and mathematically correct** in
`squish/speculative/speculative.py` (`SpeculativeGenerator._stateful_spec_stream`,
Leviathan et al. 2023 draft/verify + rejection sampling). It is **wired into the
server** (`server.py` loads the draft model at startup and builds the generator).

It does not help the benchmark because of **one line**:

```python
# squish/server.py:1597
if _draft.generator is not None and temperature > 0.0:
```

The benchmark (`bench_v5_1.py`) decodes at **temperature = 0 (greedy)**. The gate
excludes `temperature == 0`, so **every benchmark request bypasses the spec path
entirely** and runs the standard single-token autoregressive loop. The draft
model is loaded, resident, and never called during the measured run.

This is not "spec decode is broken" — it is "spec decode is switched off for
exactly the workload we benchmark."

## 1. What's loaded and wired (works today)

| Component | Location | Status |
|-----------|----------|--------|
| Package exports (`load_draft_model`, `SpeculativeGenerator`, …) | `squish/speculative/__init__.py` | ✅ exported — the v4 import bug is fixed |
| `--draft-model` / `--draft-compressed` argparse | `server.py:4136` | ✅ present |
| Draft load at startup | `server.py:5060` → `load_draft_model()` (`server.py:1275`) | ✅ both models kept resident |
| Generator construction | `server.py:1308` `_rebuild_spec_gen()` → `SpeculativeGenerator(...)` | ✅ built with draft model + tokenizer |
| Draft+verify loop | `speculative.py:884` `_stateful_spec_stream` | ✅ correct (see §3) |
| Separate KV caches | `speculative.py` `_draft_cache` / `_target_cache` via `_try_make_model_cache` | ✅ independent caches |
| Prometheus `squish_spec_draft_loaded` | `server.py:3858` | ✅ reports loaded state |

## 2. What's missing entirely

1. **temp=0 activation.** The decode path (`server.py:1597`) requires
   `temperature > 0.0`. Greedy decoding — the entire benchmark and the Phase-3
   correctness test — never enters spec decode. **This is the primary defect.**
   The inline comment ("greedy draft on temp==0 benchmarks offers less benefit
   and adds overhead") is the explicit, deliberate cause.

2. **`--draft-depth` flag.** The brief asks for K configurable via
   `--draft-depth`. No such flag exists in `server.py`. K is fixed at
   `_DEFAULT_K = 4` (`speculative.py:55`) because `_rebuild_spec_gen` constructs
   `SpeculativeGenerator(...)` without passing `k=`. (K=4 is already the desired
   default, so this is a configurability gap, not a correctness one.)

3. **A benchmark config that loads a draft model.** `bench_v5_1.py` has no
   `squish_block_spec` row — no existing benchmark config passes `--draft-model`,
   so spec decode has never been exercised by the harness even at temp>0.

## 3. What the inference loop does at decode time (current vs. intended)

**Current (temp=0, i.e. the benchmark):** `server.py` skips the spec block at
:1597, skips the Jacobi block at :1640 (draft loaded ⇒ `_draft.generator is not
None` ⇒ Jacobi's `_draft.generator is None` guard fails too — Jacobi is also
off), and falls through to the **standard autoregressive decode loop**: one
target forward per token, no draft involvement. tok/s is pure 7B decode.

**Intended (and already implemented for temp>0):** `_stateful_spec_stream`
(`speculative.py:884`) runs the textbook loop:

1. Prefill draft + target caches once.
2. Draft proposes K tokens (n-gram table fills free slots first, neural draft
   fills the rest) — K cheap 0.5/1.5B forwards.
3. Target verifies all K in **one** batched forward (`_decode_multi_cached`).
4. Sequential accept/reject with rejection sampling
   `accept iff u < min(1, p_target/p_draft)`; on reject, resample the divergence
   token from `normalize(max(0, p_target − p_draft))`; on all-accept, append a
   bonus token from the target. (`speculative.py:983-1014`)
5. Roll both caches to `base + n_accepted`, run the final token through both to
   realign, repeat.

### Correctness at temp=0 (why the fix is safe)

`_softmax_np(row, temp)` divides logits by `max(temp, 1e-8)` (`speculative.py:135`).
At `temp=0` this sharpens every distribution to a **one-hot at the argmax**.
Walking the accept/reject math with one-hot `p_draft`/`p_target`:

- draft token == target argmax ⇒ `p_target[d]=1`, accept with prob 1;
- draft token != target argmax ⇒ `p_target[d]=0`, reject, and the resample
  `max(0, p_target − p_draft)` is one-hot at the **target argmax**.

So at temp=0 the emitted sequence is **exactly the target's greedy argmax at
every position** — bit-identical to the non-spec greedy path, and fully
deterministic (the `np.random.random()` draw is irrelevant when probs ∈ {0,1}).
This is what makes the Phase-3 identical-output test viable: we can flip spec on
at temp=0 and the output must not change.

## 4. Draft-model availability constraint (affects Phase 4)

The brief's smoke test and Phase-4 matrix call for `Qwen2.5-0.5B-int4` and
`Qwen2.5-3B-int4`. **Neither exists in `~/models/`.** Inventory:

```
Qwen2.5-1.5B-Instruct-{bf16,int2,int3,int4,int4-awq,mixed-attn}
Qwen2.5-7B-Instruct-{bf16,int3,int4}
```

Only **Qwen2.5-1.5B-Instruct-int4** shares the Qwen2.5 tokenizer family with the
7B target. Per the scope guard ("don't quantize new draft models", "if draft and
target use different tokenizers, fail loud") the Phase-4 three-row matrix
collapses to a **single workable pairing: 7B-int4 target + 1.5B-int4 draft.**
The 0.5B and 3B rows are documented as unavailable rather than fabricated.

The brief's smoke command (`--draft-model qwen2.5-0.5b-int4`) is therefore
re-run with `Qwen2.5-1.5B-Instruct-int4`.

## 5. The fix (Phase 2 scope)

Minimal, isolated to the activation gate + configurability — **no change to the
draft/verify algorithm, the caches, or the block-cache code**:

1. Allow the spec path at temp=0 **only when a real draft model / EAGLE head is
   loaded** (not for the n-gram-only default path, to keep the v5.1.1
   `squish_block` / `squish_recommended` rows unchanged for comparison):
   ```python
   _has_draft = _draft.model is not None or _draft.eagle_head is not None
   if _draft.generator is not None and (temperature > 0.0 or _has_draft):
   ```
2. Add `--draft-depth` (default 4) and thread it to `SpeculativeGenerator(k=…)`.
3. Add a `squish_block_spec` benchmark config (block cache + `--draft-model
   …1.5B-int4`) so Phase 5 can isolate the spec contribution vs. `squish_block`.

## Conclusion

Nothing in the spec-decode algorithm needs rewriting. The feature is correct and
resident but **deliberately disabled for greedy decoding**, which is the only
mode the benchmark measures. The v5.2 work is to (a) enable it at temp=0 behind a
real-draft guard, (b) prove output identity, (c) measure acceptance and net
tok/s with the one available draft pairing, and (d) keep or revert on the
decision gate.
