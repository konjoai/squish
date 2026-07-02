# Prefix-reuse speedup curve — post chunk-aligned-fix results

Model: `Qwen2.5-7B-Instruct-int4` · reps=3 · gen=8 tokens · git-diff summarization prompts
Contexts: 512 / 1024 / 2048 / 4096 tokens · overlap 5–95% (by 5)

Re-run of `benchmarks/prefix_reuse_curve.py` on the **merged** chunk-aligned prefill fix
(PR #188, `_PREFILL_CHUNK=64`). Every number below is from the fixed code — the prior
run (`results/prefix_reuse_curve/20260629T021132`, `...065700`) predates the fix and used
a different (lossy) partial-reuse path, so those numbers do not carry forward.

## Headline

- **100% lossless across all 76 points** (4 contexts × 19 overlap steps) — byte-identical
  greedy output vs. a cold (no-reuse) run at every overlap level, not just exact-repeat.
  This confirms the chunk-aligned fix: pre-fix, ~60% of partial-reuse generations forked.
- Speedup scales with both overlap% and context size — larger contexts amortize the
  fixed decode/gen cost over a bigger prefill, so the *reuse* saving reads as a cleaner
  win at 4096 than at 512.
- **Ceiling (95% overlap):** 5.9× (512) → 6.95× (1024) → 11.0× (2048) → **14.7× (4096)**.
- **Crossover to >2×** happens earlier as context grows: ~55% overlap at 512/1024,
  ~50% at 2048, ~50–55% at 4096.

## Results by context

### context ≈ 512 tokens
| overlap% (real) | cold s | reuse s | speedup |
|---|---|---|---|
| 5 (12.3) | 3.60 | 3.13 | 1.15× |
| 25 (30.7) | 5.48 | 4.15 | 1.32× |
| 50 (53.7) | 5.50 | 3.54 | 1.55× |
| 75 (77.0) | 5.59 | 2.03 | 2.75× |
| 95 (95.3) | 5.85 | 1.00 | 5.86× |

### context ≈ 1024 tokens
| overlap% (real) | cold s | reuse s | speedup |
|---|---|---|---|
| 5 (8.6) | 11.13 | 10.40 | 1.07× |
| 25 (27.8) | 11.16 | 8.57 | 1.30× |
| 50 (51.9) | 11.27 | 6.12 | 1.84× |
| 75 (76.0) | 11.25 | 3.48 | 3.23× |
| 95 (95.2) | 12.08 | 1.74 | 6.95× |

### context ≈ 2048 tokens
| overlap% (real) | cold s | reuse s | speedup |
|---|---|---|---|
| 5 (6.8) | 23.77 | 22.53 | 1.06× |
| 25 (26.4) | 24.67 | 18.47 | 1.34× |
| 50 (50.9) | 25.06 | 12.71 | 1.97× |
| 75 (75.5) | 24.32 | 6.93 | 3.51× |
| 95 (95.1) | 24.91 | 2.26 | 11.0× |

### context ≈ 4096 tokens
| overlap% (real) | cold s | reuse s | speedup |
|---|---|---|---|
| 5 (5.9) | 49.36 | 46.21 | 1.07× |
| 25 (25.7) | 49.26 | 37.98 | 1.30× |
| 50 (50.5) | 41.67 | 21.86 | 1.91× |
| 75 (75.2) | 41.70 | 11.45 | 3.64× |
| 95 (95.0) | 41.18 | 2.79 | 14.74× |

Full per-point tables (all 19 overlap steps, not just quartiles): `curve.md` in
`results/prefix_reuse_curve/20260629T225024/` (gitignored raw artifact — regenerate with
the command below).

## Losslessness framing (for writeups)
"Exact at 100% reuse; **byte-identical for partial reuse** after the chunk-aligned
(C=64) prefill fix — validated at every overlap level from 5–95%, not just exact-repeat."

## Reproduce
```bash
cd ~/squish && PYTHONPATH=~/squish .venv/bin/python -m benchmarks.prefix_reuse_curve \
  --contexts 512,1024,2048,4096
```

## Next
Full ollama-vs-squish context/reuse matrix (`benchmarks/ollama_vs_squish/matrix/`) is the
remaining piece — in progress, see `BENCH_HANDOFF.md` for status and resume instructions.
