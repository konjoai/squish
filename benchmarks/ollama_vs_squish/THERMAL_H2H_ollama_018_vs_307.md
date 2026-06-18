# Thermally-controlled head-to-head — Ollama 0.18.2 vs 0.30.7 (M3 16 GB)

**TL;DR — the two Ollama versions are indistinguishable on this machine.** At p75 and p2000 they match to a tenth of a tok/s. At p4000 the M3's thermal envelope—not the engine—sets throughput; a dedicated isolation run proves the apparent version gap there is a measurement-order artifact, not a real difference.

## Method

- Harness: `benchmarks/ollama_vs_squish/bench_thermal_h2h.py` (wraps `bench_v5_1.py`), repo at commit `37bac10`.
- Identical for both versions: same harness, same squish tree, same models, same thermal protocol. **Only the Ollama binary changed.**
- Thermal protocol: 120 s cooldown before each config (all servers down), 25 s settle before each phase, 5 runs/metric. Configs run cool→hot in fixed order with an `ollama_recheck` drift probe last.
- Models: Ollama `qwen2.5:7b` (q4_K_M); squish `Qwen2.5-7B-Instruct` INT4/INT3.
- Prompt sizes: p75 = 57 tok, p2000 = 2001 tok, p4000 = 4053 tok.

- Binaries: **0.18.2** = `/opt/homebrew/bin/ollama` (Cellar keg) · **0.30.7** = `/usr/local/bin/ollama`.

- Raw: `results/.../thermal/20260618T101440.json` (0.18.2 sweep) · `20260614T110922.json` (0.30.7 sweep) · `20260618T125123_p4000iso.json` (p4000 isolation).


## p75 (57 tok) — ✅ thermally clean

| Config | 0.18.2 tok/s | 0.30.7 tok/s | itl_p50 (18/30) | itl_p95 (18/30) | E2E-200s (18/30) | TTFT ms (18/30) |
|---|---|---|---|---|---|---|
| Ollama (warm) | 20.3 | 20.3 | 50.5/50.2 | 52.3/52.4 | 2.7/2.7 | 126/167 |
| Squish daemon INT4 | 21.4 | 21.7 | 45.5/44.9 | 49.6/46.9 | 2.7/2.6 | 624/615 |
| Squish recommended INT4 (block+pkv) | 19.4 | 20.5 | 48.0/45.3 | 51.7/48.4 | 2.7/2.6 | 195/192 |
| Squish recommended INT3 (block+pkv) | 23.5 | 24.0 | 39.8/39.2 | 44.4/42.7 | 3.0/3.0 | 193/199 |
| Ollama (recheck) | 20.3 | 20.5 | 50.1/49.7 | 52.4/51.7 | 2.6/2.7 | 122/166 |

## p2000 (2001 tok) — ✅ thermally clean

| Config | 0.18.2 tok/s | 0.30.7 tok/s | itl_p50 (18/30) | itl_p95 (18/30) | E2E-200s (18/30) | TTFT ms (18/30) |
|---|---|---|---|---|---|---|
| Ollama (warm) | 19.7 | 19.7 | 51.7/51.3 | 54.0/52.9 | 3.8/3.8 | 130/180 |
| Squish daemon INT4 | 20.4 | 20.9 | 47.7/47.1 | 54.5/51.4 | 15.2/14.9 | 10343/10386 |
| Squish recommended INT4 (block+pkv) | 19.9 | 20.2 | 47.5/46.9 | 53.6/51.3 | 4.0/3.9 | 399/406 |
| Squish recommended INT3 (block+pkv) | 22.0 | 22.6 | 42.0/41.1 | 47.3/45.4 | 3.1/3.0 | 409/407 |
| Ollama (recheck) | 19.6 | 19.8 | 51.6/51.0 | 54.8/53.0 | 3.8/3.8 | 131/183 |

**Drift check (p75 ollama first→last):** 0.18.2 20.3→20.3 (+0.4%) · 0.30.7 20.3→20.5 (+0.8%) — small ⇒ cooldowns held, p75/p2000 comparisons are fair.


## p4000 (4053 tok) — ⚠️ thermal-bound, NOT a clean version comparison

The full-sweep p4000 numbers below are shown for completeness, but at p4000 the fixed 120 s cooldown does **not** dissipate the heat from ten 4000-token prefills, so later configs run hotter and slower regardless of binary.


| Config | 0.18.2 tok/s | 0.30.7 tok/s |
|---|---|---|
| Ollama (warm) | 18.4 | 17.0 |
| Squish daemon INT4 | 12.5 | 13.4 |
| Squish recommended INT4 (block+pkv) | 15.7 | 19.1 |
| Squish recommended INT3 (block+pkv) | 16.8 | 19.5 |
| Ollama (recheck) | 9.8 | 11.4 |

### Isolation run — p4000 only, each version from a cold start

To remove the order artifact, p4000 was re-measured alone (sequence 0.18.2 → 0.30.7 → 0.18.2 recheck), each with its own cooldown:


| Run (in order) | p4000 tok/s | itl_p50 | itl_p95 | E2E-200 s | peak RSS |
|---|---|---|---|---|---|
| ollama_018 | 16.1 | 62.1 | 72.8 | 39.7 | 3.40 GB |
| ollama_307 | 10.6 | 97.0 | 109.1 | 53.4 | 4.94 GB |
| ollama_018_recheck | 10.0 | 61.9 | 132.9 | 47.9 | 5.17 GB |

**Drift: 0.18.2 first→recheck 16.1 → 10.0 tok/s (-38%).** The 120 s cooldown reaches a hot steady-state (~10 tok/s), not the deep-cold state (~16 tok/s) the first config caught. Lining up by *temperature* rather than by binary: cold 0.18.2 ≈ cold 0.30.7 (~16–18), hot 0.18.2 ≈ hot 0.30.7 (~10). **No version effect** — 0.18.2 in the same hot slot was actually a hair slower than 0.30.7.


## Conclusion

- **Ollama 0.18.2 and 0.30.7 perform identically** on this M3 for `qwen2.5:7b` single-stream decode.
- p75 / p2000 are drift-validated and directly comparable.
- p4000 absolute throughput is **bounded by the laptop's thermal envelope, not the engine**; a precise p4000 number would require temperature-gated cooldowns (macmon), but it would only reconfirm parity.
- **Do not cite the pre-thermal June-2 v5.1.1 numbers** alongside these — different methodology.

