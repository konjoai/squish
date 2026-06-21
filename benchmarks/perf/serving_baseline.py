#!/usr/bin/env python3
"""Squish-only serving baseline — the reference numbers every perf PR compares to.

This is the *inference hot-path* baseline (PR-C scheduler / PR-D KV cache):
it loads each model via ``--mlx-model-dir`` (the ``mlx_lm.load`` path) and
measures, per model, with a fixed thermal protocol:

  * cold_start_s   — wall time from server launch to ``/health`` ready
  * ttft_s         — time to first token (max_tokens=1), at p75 and p2000
  * decode_tps     — steady-state decode tok/s (DECODE_TOKENS completion)
  * itl_p50/95/99  — inter-token latency, all decode gaps pooled across runs
  * peak_rss_bytes — peak resident set of the server process tree

Konjo benchmark rules (.claude/rules/benchmarking.md):
  * >= 5 warmup runs; report p50/p95/p99/stddev.
  * Identical thermal protocol (COOLDOWN_S idle, servers down) before every
    model so cross-model and cross-PR numbers are comparable.
  * Results written to benchmarks/results/<UTC>/ — never overwritten.

Run:
  PYTHONPATH=benchmarks/ollama_vs_squish \
    ~/squish/.venv/bin/python benchmarks/perf/serving_baseline.py [--quick] [tag]

The optional positional ``tag`` (default "baseline") names the output dir so a
post-change run can be captured as e.g. ``pr-c`` and diffed against baseline.
``--quick`` shrinks cooldown/runs for pipeline debugging only — never for a
number you intend to quote.
"""
from __future__ import annotations

import json
import os
import platform
import statistics as stats
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Reuse the v5.1 harness plumbing verbatim (same launcher, RSS sampler,
# streaming client, ready-probe) so this stays DRY with the shipped harness.
import bench_v5_1 as B  # noqa: E402  (path injected by caller via PYTHONPATH)

# ── Config (constant => reproducible; --quick overrides for debugging only) ──

MODELS: list[tuple[str, str, str]] = [
    ("qwen1.5b-int4", "~/models/Qwen2.5-1.5B-Instruct-int4", "INT4"),
    ("qwen1.5b-int3", "~/models/Qwen2.5-1.5B-Instruct-int3", "INT3"),
    ("qwen7b-int4", "~/models/Qwen2.5-7B-Instruct-int4", "INT4"),
    ("qwen7b-int3", "~/models/Qwen2.5-7B-Instruct-int3", "INT3"),
]

COOLDOWN_S = 120      # idle (servers down) before each model — thermal reset
                      # (matches the repo's THERMAL_H2H protocol)
WARMUP_RUNS = 5       # discarded; settles weights + JITs kernels
TTFT_RUNS = 8         # measured TTFT samples per prompt size
DECODE_RUNS = 8       # measured decode samples
DECODE_TOKENS = 128   # completion length for steady-state decode + ITL

REPO_ROOT = Path(__file__).resolve().parents[2]


def _apply_quick() -> None:
    """Shrink the protocol for pipeline debugging. Not for quotable numbers."""
    global COOLDOWN_S, WARMUP_RUNS, TTFT_RUNS, DECODE_RUNS
    COOLDOWN_S, WARMUP_RUNS, TTFT_RUNS, DECODE_RUNS = 5, 2, 3, 3


# ── Stats ────────────────────────────────────────────────────────────────────

def _pct(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0,1]) on a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def summarize(vals: list[float]) -> dict[str, Any]:
    """p50/p95/p99/min/max/stddev for a sample of measurements."""
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return {"n": 0, "p50": None, "p95": None, "p99": None,
                "min": None, "max": None, "stddev": None}
    s = sorted(clean)
    return {
        "n": len(s),
        "p50": _pct(s, 0.50),
        "p95": _pct(s, 0.95),
        "p99": _pct(s, 0.99),
        "min": s[0],
        "max": s[-1],
        "stddev": stats.pstdev(s) if len(s) > 1 else 0.0,
    }


# ── Measurement ──────────────────────────────────────────────────────────────

def _pooled_itl(decode_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool every decode inter-token gap across runs, then take percentiles.

    Pooling (vs per-run-then-median) gives a far larger sample for the tail,
    which is exactly where decode-loop / scheduler jitter shows up.
    """
    gaps: list[float] = []
    for d in decode_results:
        ts = d.get("chunk_timestamps") or []
        # Drop the first gap (that's TTFT, measured separately).
        gaps.extend((ts[i + 1] - ts[i]) * 1000.0 for i in range(1, len(ts) - 1))
    return summarize(gaps)


def measure_model(name: str, model_path: str, quant: str,
                  prompts: dict[str, str]) -> dict[str, Any]:
    """Cooldown, launch, warm, then measure one model end to end."""
    B.log(f"=== {name} : cooldown {COOLDOWN_S}s → launch ===")
    B.kill_all_serving()
    time.sleep(COOLDOWN_S)

    lp = B.LOG_DIR / f"baseline_{name}_{B.TS}.log"
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        B._squish_cmd(model_path, []),
        stdout=open(lp, "wb"), stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": B.SQUISH_API_KEY},
    )
    sampler = B.RSSSampler(proc.pid)
    sampler.start()
    result: dict[str, Any] = {"name": name, "quant": quant, "model_path": model_path}
    try:
        ready_url = f"http://{B.SQUISH_HOST}:{B.SQUISH_PORT}/health"
        if not B.wait_ready(ready_url, timeout=300):
            raise RuntimeError(f"{name} did not become ready within 300s")
        result["cold_start_s"] = time.perf_counter() - t0
        B.log(f"  cold_start={result['cold_start_s']:.2f}s; {WARMUP_RUNS} warmups")
        for _ in range(WARMUP_RUNS):
            B.stream_squish(prompts["p75"], max_tokens=DECODE_TOKENS)

        # TTFT at two prefill sizes.
        ttft: dict[str, Any] = {}
        for psize in ("p75", "p2000"):
            samples = [B.stream_squish(prompts[psize], max_tokens=1)["ttft_s"]
                       for _ in range(TTFT_RUNS)]
            ttft[psize] = summarize([s for s in samples if s])
        result["ttft_s"] = ttft

        # Steady-state decode + pooled inter-token latency (p75 prefill).
        decode_runs = [B.stream_squish(prompts["p75"], max_tokens=DECODE_TOKENS)
                       for _ in range(DECODE_RUNS)]
        result["decode_tps"] = summarize(
            [d["tokens_per_sec"] for d in decode_runs if d["tokens_per_sec"]])
        result["itl_ms"] = _pooled_itl(decode_runs)
        result["peak_rss_bytes"] = sampler.peak_bytes
        B.log(f"  ttft75={_ms(ttft['p75']['p50'])}  "
              f"decode={_t(result['decode_tps']['p50'])}  "
              f"itl_p95={_ms_raw(result['itl_ms']['p95'])}  "
              f"rss={result['peak_rss_bytes'] / 1e9:.2f}GB")
    finally:
        B.stop_server(proc, sampler)
    return result


# ── Formatting ───────────────────────────────────────────────────────────────

def _ms(v: float | None) -> str:
    return "-" if v is None else f"{v * 1000:.0f}ms" if v < 1 else f"{v:.2f}s"


def _ms_raw(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}ms"


def _t(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}t/s"


def print_table(results: list[dict[str, Any]]) -> None:
    print()
    print("# Squish serving baseline (M3 16 GB) — inference hot path")
    cols = [r["name"] for r in results]
    print(f"{'metric':<22}" + "".join(f"{c:>16}" for c in cols))
    print("-" * (22 + 16 * len(cols)))
    rows: list[tuple[str, Any]] = [
        ("cold_start", lambda r: _ms(r.get("cold_start_s"))),
        ("ttft p75 p50", lambda r: _ms_raw(r["ttft_s"]["p75"]["p50"] * 1000)),
        ("ttft p75 p95", lambda r: _ms_raw(r["ttft_s"]["p75"]["p95"] * 1000)),
        ("ttft p2000 p50", lambda r: _ms_raw(r["ttft_s"]["p2000"]["p50"] * 1000)),
        ("decode p50", lambda r: _t(r["decode_tps"]["p50"])),
        ("decode stddev", lambda r: _t(r["decode_tps"]["stddev"])),
        ("itl p50", lambda r: _ms_raw(r["itl_ms"]["p50"])),
        ("itl p95", lambda r: _ms_raw(r["itl_ms"]["p95"])),
        ("itl p99", lambda r: _ms_raw(r["itl_ms"]["p99"])),
        ("peak RSS", lambda r: f"{r['peak_rss_bytes'] / 1e9:.2f}GB"),
    ]
    for label, fn in rows:
        print(f"{label:<22}" + "".join(f"{fn(r):>16}" for r in results))
    print()


def main() -> int:
    argv = sys.argv[1:]
    if "--quick" in argv:
        _apply_quick()
        argv.remove("--quick")
    tag = argv[0] if argv else "baseline"

    out_dir = REPO_ROOT / "benchmarks" / "results" / f"{B.TS}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    B.log(f"Serving baseline — tag={tag}  cooldown={COOLDOWN_S}s  "
          f"warmup={WARMUP_RUNS}  ttft_runs={TTFT_RUNS}  decode_runs={DECODE_RUNS}")
    B.log("Building prompts (p75, p2000) via Qwen2.5 tokenizer …")
    prompts = {
        "p75": B._P75,
        "p2000": B._build_prompt_to_tokens(B._P75, target_tokens=2000),
    }

    present = [(n, os.path.expanduser(p), q) for n, p, q in MODELS
               if os.path.isdir(os.path.expanduser(p))]
    missing = [n for n, p, _ in MODELS if not os.path.isdir(os.path.expanduser(p))]
    if missing:
        B.log(f"SKIP (not on disk): {missing}")

    results = [measure_model(n, p, q, prompts) for n, p, q in present]

    meta = {
        "timestamp": B.TS,
        "tag": tag,
        "host": "Apple M3 MacBook Pro 16 GB",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx_version": _pkg_version("mlx"),
        "mlx_lm_version": _pkg_version("mlx_lm"),
        "squish_version": _squish_version(),
        "protocol": {"cooldown_s": COOLDOWN_S, "warmup_runs": WARMUP_RUNS,
                     "ttft_runs": TTFT_RUNS, "decode_runs": DECODE_RUNS,
                     "decode_tokens": DECODE_TOKENS},
        "results": results,
    }
    out = out_dir / "serving_baseline.json"
    out.write_text(json.dumps(meta, indent=2, default=str))
    B.log(f"Wrote {out}")
    print_table(results)
    return 0


def _pkg_version(mod: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(mod)
    except md.PackageNotFoundError:
        return "unknown"


def _squish_version() -> str:
    try:
        import squish
        return squish.__version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
