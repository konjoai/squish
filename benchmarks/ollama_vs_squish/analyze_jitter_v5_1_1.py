#!/usr/bin/env python3
"""v5.1.1 jitter extractor — per-token inter-token gaps vs token index.

Reads a bench_v5_1 raw.json and, for a chosen config + prompt size, takes the
median-e2e run's chunk_timestamps and emits the inter-token gap series so we
can spot periodic spikes (cache housekeeping), monotonic drift (thermal), or
one-time stalls at block boundaries.  Read-only analysis — no inference here.

Usage:
    python analyze_jitter_v5_1_1.py <raw.json> [config] [phase]
Defaults: config=squish_recommended_int4  phase=p2000
"""
from __future__ import annotations

import json
import statistics as stats
import sys
from pathlib import Path


def gaps_for_run(run: dict) -> list[float]:
    ts = run.get("chunk_timestamps") or []
    return [(ts[i + 1] - ts[i]) * 1000.0 for i in range(len(ts) - 1)]


def pick_median_run(runs: list[dict]) -> dict | None:
    scored = [(r.get("total_s") or 0.0, r) for r in runs if r.get("chunk_timestamps")]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[len(scored) // 2][1]


def main() -> None:
    raw_path = Path(sys.argv[1])
    config = sys.argv[2] if len(sys.argv) > 2 else "squish_recommended_int4"
    phase = sys.argv[3] if len(sys.argv) > 3 else "p2000"
    data = json.loads(raw_path.read_text())
    cfg = data["configs"][config]
    runs = cfg["phases"][phase]["e2e_runs"]
    run = pick_median_run(runs)
    if run is None:
        print(f"no usable run for {config}/{phase}")
        return
    gaps = gaps_for_run(run)
    # gaps[0] is TTFT→token2 (first decode gap); decode-only is gaps[1:]
    decode = gaps[1:]
    print(f"# {config} / {phase} — median-e2e run (total_s={run.get('total_s'):.2f})")
    print(f"tokens={len(gaps)+1}  decode_gaps={len(decode)}")
    if decode:
        sd = sorted(decode)
        print(f"decode gap ms: p50={sd[len(sd)//2]:.1f} "
              f"p95={sd[int(len(sd)*0.95)]:.1f} "
              f"p99={sd[min(int(len(sd)*0.99), len(sd)-1)]:.1f} "
              f"max={sd[-1]:.1f} mean={stats.mean(decode):.1f} "
              f"stdev={stats.pstdev(decode):.1f}")
        # First vs last quartile mean → thermal drift signal.
        q = max(1, len(decode) // 4)
        print(f"first-quartile mean={stats.mean(decode[:q]):.1f} ms  "
              f"last-quartile mean={stats.mean(decode[-q:]):.1f} ms  "
              f"(drift={stats.mean(decode[-q:]) - stats.mean(decode[:q]):+.1f} ms)")
        # Spike positions: gaps > 2× median.
        med = sd[len(sd)//2]
        spikes = [(i + 1, g) for i, g in enumerate(decode) if g > 2 * med]
        print(f"spikes >2×median ({2*med:.0f} ms): {len(spikes)}")
        for idx, g in spikes[:25]:
            print(f"  token#{idx:>3}  gap={g:.0f} ms  "
                  f"(block_boundary={'YES' if idx % 64 == 0 else 'no'})")
        # Per-token series (compact) for plotting/inspection.
        print("series_ms=" + ",".join(f"{g:.0f}" for g in decode))


if __name__ == "__main__":
    main()
