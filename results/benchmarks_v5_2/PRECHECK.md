# v5.2 Spec-Decode — PRECHECK

**Date:** 2026-06-02 · **Hardware:** Apple M3, 16 GB, Darwin 25.5.0
**Target:** Qwen2.5-7B-Instruct-int4 · **Draft:** Qwen2.5-1.5B-Instruct-int4
**Decision: REVERT** — draft/verify stays opt-in via `temperature > 0.0`.

## What was investigated
Spec decode was fully implemented but never fired at greedy `temp=0` (the warm
benchmark path). v5.2 enabled it behind a draft-resident guard, fixed the
latent breakages that surfaced, then measured whether it earns net throughput.

## Fixes made (kept — harmless, only active when `--draft-model` is passed)
- **bf16 → numpy:** 7B emits bf16 logits; cast `.astype(mx.float32)` before
  `np.array()` at all 5 conversion sites in `speculative.py`.
- **Vocab-width align:** 7B lm_head pads to 152064, 1.5B to 151936; slice
  target rows + draft probs to the common min width before compare.
- **Greedy-match verify branch:** at `temp=0`, accept iff `draft == target
  argmax`, else emit target argmax (deterministic) instead of stochastic
  rejection sampling.
- **`--draft-depth` (K) flag** plumbed through `_DraftState` → `_rebuild_spec_gen`.

## Why REVERT (all three gates tripped)
| Gate | Result |
|------|--------|
| Net tok/s ≥ 1.5× | **No** — best short-ctx 0.87×, p4000 0.16× |
| Outputs bit-identical | **No** — int4 logit ties flip batched-verify vs sequential |
| Improvement ≥ 1.3% | **No** — net negative at every measured config |

### Short context (75 tok, baseline 17.19 tok/s)
| config | acc | tok/s | net× | identical |
|--------|-----|-------|------|-----------|
| ngram=0 K=1 | 0.811 | 11.79 | 0.69 | False |
| ngram=0 K=2 | 0.747 | 14.88 | 0.87 | False |
| ngram=0 K=4 | 0.606 | 14.78 | 0.86 | False |
| ngram=0 K=6 | 0.451 | 11.74 | 0.68 | True |
| ngram=8 K=4 | 0.323 |  9.50 | 0.55 | False |

### Long context — p4000, the article's target metric (4039 tok, baseline 7.13 tok/s, clean)
| config | acc | tok/s | net× | identical |
|--------|-----|-------|------|-----------|
| ngram=0 K=2 | 0.633 | 1.16 | **0.16** | False |
| ngram=0 K=4 | 0.417 | 0.44 | **0.06** | True |

**Acceptance is healthy (0.63 at K=2) but net throughput collapses** because the
verify path's per-cycle cost scales with context length on M3 int4 — exactly the
regime (long, attention-bound) the warm benchmark targets.

## Root causes (not bugs to fix — fundamental)
1. **int4 logit ties** — quantized lm_head yields exactly-equal max logits;
   batched-verify `[1,K]` and sequential-decode `[1,1]` resolve near-ties
   differently, so greedy output is not bit-identical. Inherent to int4.
2. **Long-context verify cost** — re-forwarding the K-token verify batch over a
   4000-token KV state dominates; the draft savings can't pay for it.

## Reproduce
```bash
PYENV_VERSION=squish python benchmarks/ollama_vs_squish/phase4_v5_2.py      # short ctx sweep
PYENV_VERSION=squish python benchmarks/ollama_vs_squish/p4000_spec_v5_2.py  # long ctx
PYENV_VERSION=squish python benchmarks/ollama_vs_squish/correctness_v5_2.py # identity gate
```

## Note on measurement hygiene
The first p4000 pass overlapped a separately-running squish server. A clean
re-run with no contention moved the baseline only 7.53 → 7.13 tok/s (5.6%, within
noise) and left the net× collapse intact, so the contention did not change the
decision. Absolute numbers above are from the clean run.
