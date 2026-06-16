#!/usr/bin/env python3
"""Focused A/B for the Category-1 serving fixes.

Measures only the regressed short-context category — squish_daemon warm tok/s
and inter-token latency (p50/p95) at p75 and p500 — against a stored baseline.
This is the cheap verification that the queue-decouple + GC-guard + SSE-coalesce
changes flip the throughput + tail-latency losses, without re-running the full
7-config v5.1 matrix.

Usage:
    python bench_verify_serving.py            # run + compare to BASELINE below
"""
from __future__ import annotations

import statistics as stats
from typing import Any

import bench_v5_1 as B

# Baseline = run results/benchmarks_v5_1_1/runs/20260613T211428 (pre-fix server).
BASELINE = {
    "ollama":        {"p75": {"warm_tps": 18.36, "itl_p50": 55.36, "itl_p95": 61.49}},
    "squish_daemon": {"p75": {"warm_tps": 14.58, "itl_p50": 48.63, "itl_p95": 166.14}},
}

PHASES = ["p75", "p500"]


def _phase_stats(phase_data: dict[str, Any]) -> dict[str, float | None]:
    es = phase_data["e2e_runs"]
    tps = [r["tokens_per_sec"] for r in es if r.get("tokens_per_sec")]
    i50 = [r["itl_p50_ms"] for r in es if r.get("itl_p50_ms")]
    i95 = [r["itl_p95_ms"] for r in es if r.get("itl_p95_ms")]
    return {
        "warm_tps": stats.median(tps) if tps else None,
        "itl_p50":  stats.median(i50) if i50 else None,
        "itl_p95":  stats.median(i95) if i95 else None,
    }


def main() -> None:
    from mlx_lm import load
    _, tok = load(B.SQUISH_MODEL_INT4)
    prompts = {
        "p75":  B._P75,
        "p500": B._build_prompt_to_tokens(B._P75, target_tokens=500),
    }
    B.log(f"prompt tokens: p75={len(tok.encode(prompts['p75']))}, "
          f"p500={len(tok.encode(prompts['p500']))}")

    after: dict[str, dict[str, Any]] = {}
    for cfg_id in ("squish_daemon",):
        cfg = B.CONFIGS[cfg_id]
        B.log(f"=== {cfg_id} : start (NEW server) ===")
        B.kill_all_serving()
        log_path = B.LOG_DIR / f"verify_{cfg_id}_{B.TS}.log"
        proc, sampler = cfg["start"](log_path)
        try:
            if not B.wait_ready(cfg["ready_url"], timeout=300):
                raise RuntimeError(f"{cfg_id} not ready")
            cfg["stream"]("Hello.", max_tokens=4)
            after[cfg_id] = {}
            for pname in PHASES:
                ph = B.run_one_phase(cfg_id, prompts[pname], pname)
                after[cfg_id][pname] = _phase_stats(ph)
                st = after[cfg_id][pname]
                B.log(f"  {pname}: warm_tps={st['warm_tps']:.1f}  "
                      f"itl_p50={st['itl_p50']:.1f}ms  itl_p95={st['itl_p95']:.1f}ms")
        finally:
            B.stop_server(proc, sampler)

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    print("# Category-1 serving fix — squish_daemon before/after (M3 16 GB)")
    print(f"{'Phase / metric':<22} | {'BEFORE':>10} | {'AFTER':>10} | {'Δ':>10} | "
          f"{'ollama':>10} | verdict")
    print("-" * 90)
    for pname in PHASES:
        a = after["squish_daemon"][pname]
        b = BASELINE["squish_daemon"].get(pname, {})
        oll = BASELINE["ollama"].get(pname, {})
        for metric, better_high, fmt in (
            ("warm_tps", True, "{:.1f}"),
            ("itl_p50", False, "{:.1f}ms"),
            ("itl_p95", False, "{:.1f}ms"),
        ):
            av = a.get(metric)
            bv = b.get(metric)
            ov = oll.get(metric)
            d = (av - bv) if (av is not None and bv is not None) else None
            # verdict only meaningful at p75 (where we have ollama baseline)
            verdict = ""
            if ov is not None and av is not None:
                win = (av > ov) if better_high else (av < ov)
                verdict = "BEAT ollama" if win else "still behind"
            print(f"{pname + ' ' + metric:<22} | "
                  f"{(fmt.format(bv) if bv is not None else '-'):>10} | "
                  f"{(fmt.format(av) if av is not None else '-'):>10} | "
                  f"{((f'{d:+.1f}') if d is not None else '-'):>10} | "
                  f"{(fmt.format(ov) if ov is not None else '-'):>10} | {verdict}")
    print()


if __name__ == "__main__":
    main()
