#!/usr/bin/env python3
"""Clean single-session head-to-head: ollama 0.18.2 vs current fixed+pinned squish.

Points the v5.1 harness at the REAL 0.18.2 binary (homebrew Cellar; the app
binary auto-updated to 0.30.7), trims to the headline configs, and runs the
canonical v5.1 methodology so results are directly comparable to the original
ollama-0.18.2 run (results/benchmarks_v5_1_1/runs/20260613T171549).

Order is ollama-first (coolest) — same as the original run, and conservative
for squish (it runs hotter).
"""
import bench_v5_1 as B

# The true 0.18.2 binary (the /usr/local/bin app symlink is now 0.30.7).
B.OLLAMA_BIN = "/opt/homebrew/bin/ollama"

# Headline configs only — ollama vs the production squish configs.
_KEEP = {
    "ollama",
    "squish_daemon",
    "squish_recommended_int4",
    "squish_recommended_int3",
}
for _k in list(B.CONFIGS):
    if _k not in _KEEP:
        B.CONFIGS.pop(_k)

if __name__ == "__main__":
    B.main()
