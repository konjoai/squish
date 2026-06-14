#!/usr/bin/env python3
"""Lever benchmark — quantify the post-serving-fix model-side levers.

Measures four levers on top of the already-fixed serving layer (queue-decouple
+ GC guard + P-core QoS pinning, all live in squish/server.py):

  1. P-core pinning — compare INT4 p75 p95 here vs the stored unpinned 51 ms.
  2. INT3           — lower-bit weights → less memory traffic → faster decode.
  3. Quantized KV   — --kv-cache-mode int8 → less KV bandwidth at long context.
  4. Speculative    — 1.5B draft + 7B target.  HARD-GATED on temperature>0 in
                      the server, so it is measured at temp 0.7 against a temp
                      0.7 baseline (a separate operating point from the temp 0
                      headline rows above).

warm tok/s is decode-only (excludes prefill); itl_p50/p95 exclude the first
token.  Output: results/benchmarks_v5_1_1/levers/<UTC>.json + stdout table.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics as stats
import subprocess
import time
from pathlib import Path
from typing import Any

import bench_v5_1 as B

DRAFT_MODEL = "/Users/wscholl/models/Qwen2.5-1.5B-Instruct-int4"
OUT_DIR = B.REPO_ROOT / "results" / "benchmarks_v5_1_1" / "levers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = 4              # measured e2e runs per phase (+1 discarded warmup)
E2E_TOKENS = 200

# Stored reference points (pre-lever, post-serving-fix, temp 0, UNPINNED):
#   bench_verify_serving.py run 20260613T232949
REF = {
    "ollama_p75_tps": 18.4, "ollama_p75_p95": 61.5,
    "int4_p75_tps": 20.0, "int4_p75_p95": 51.2,   # fixed server, no pin
}


def _stream(prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    """OpenAI streaming client with configurable temperature + usage."""
    import urllib.request
    body = json.dumps({
        "model": "squish",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "max_tokens": max_tokens, "temperature": temperature,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://{B.SQUISH_HOST}:{B.SQUISH_PORT}/v1/chat/completions",
        data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {B.SQUISH_API_KEY}",
        },
    )
    t_req = time.perf_counter()
    t_first = None
    parts: list[str] = []
    ts: list[float] = []
    comp_tokens = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for c in (d.get("choices") or []):
                chunk = (c.get("delta") or {}).get("content") or ""
                if chunk:
                    t = time.perf_counter()
                    ts.append(t)
                    if t_first is None:
                        t_first = t
                    parts.append(chunk)
            if d.get("usage"):
                comp_tokens = d["usage"].get("completion_tokens")
    t_done = time.perf_counter()
    n = comp_tokens or len(parts)
    win = (t_done - t_first) if t_first else None
    tps = (n / win) if (win and win > 0 and n) else None
    return {"ttft_s": (t_first - t_req) if t_first else None,
            "tokens_per_sec": tps, "chunk_timestamps": ts}


def _measure_phase(prompt: str, temperature: float) -> dict[str, float | None]:
    _stream(prompt, E2E_TOKENS, temperature)             # discarded warmup
    tps_v: list[float] = []
    i50_v: list[float] = []
    i95_v: list[float] = []
    for _ in range(RUNS):
        d = _stream(prompt, E2E_TOKENS, temperature)
        itl = B._inter_token_stats(d)
        if d["tokens_per_sec"]:
            tps_v.append(d["tokens_per_sec"])
        if itl["itl_p50_ms"]:
            i50_v.append(itl["itl_p50_ms"])
        if itl["itl_p95_ms"]:
            i95_v.append(itl["itl_p95_ms"])
    return {
        "warm_tps": stats.median(tps_v) if tps_v else None,
        "itl_p50":  stats.median(i50_v) if i50_v else None,
        "itl_p95":  stats.median(i95_v) if i95_v else None,
    }


def _launch(model: str, extra: list[str], log_path: Path):
    B.kill_all_serving()
    cmd = [
        B.SQUISH_PY, "-m", "squish.server",
        "--mlx-model-dir", model,
        "--port", str(B.SQUISH_PORT), "--host", B.SQUISH_HOST,
        "--log-level", "warning", *extra,
    ]
    proc = subprocess.Popen(
        cmd, stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": B.SQUISH_API_KEY},
    )
    sampler = B.RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def _run_config(name: str, model: str, extra: list[str],
                phases: dict[str, str], temperature: float) -> dict[str, Any]:
    B.log(f"=== {name} (temp={temperature}) extra={extra} ===")
    log_path = B.LOG_DIR / f"levers_{name}_{B.TS}.log"
    proc, sampler = _launch(model, extra, log_path)
    out: dict[str, Any] = {"temperature": temperature, "extra": extra, "phases": {}}
    try:
        if not B.wait_ready(f"http://{B.SQUISH_HOST}:{B.SQUISH_PORT}/health", timeout=300):
            raise RuntimeError(f"{name} not ready")
        _stream("Hello.", 4, temperature)  # trigger load
        for pname, ptext in phases.items():
            st = _measure_phase(ptext, temperature)
            out["phases"][pname] = st
            B.log(f"  {pname}: tps={_f(st['warm_tps'])}  "
                  f"p50={_f(st['itl_p50'])}ms  p95={_f(st['itl_p95'])}ms")
    finally:
        B.stop_server(proc, sampler)
    out["peak_rss_bytes"] = sampler.peak_bytes
    return out


def _f(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "-"


def main() -> None:
    from mlx_lm import load
    _, tok = load(B.SQUISH_MODEL_INT4)
    p75 = B._P75
    p2000 = B._build_prompt_to_tokens(B._P75, target_tokens=2000)
    B.log(f"prompts: p75={len(tok.encode(p75))}t  p2000={len(tok.encode(p2000))}t")

    short = {"p75": p75}
    both = {"p75": p75, "p2000": p2000}
    has_int3 = os.path.isdir(B.SQUISH_MODEL_INT3)
    has_draft = os.path.isdir(DRAFT_MODEL)

    results: dict[str, Any] = {"timestamp": B.TS, "host": "Apple M3 16 GB",
                               "runs": RUNS, "ref": REF, "configs": {}}

    # ── temp 0 headline operating point ───────────────────────────────────────
    results["configs"]["int4_pinned"] = _run_config(
        "int4_pinned", B.SQUISH_MODEL_INT4, [], both, 0.0)
    if has_int3:
        results["configs"]["int3_pinned"] = _run_config(
            "int3_pinned", B.SQUISH_MODEL_INT3, [], both, 0.0)
    results["configs"]["int4_qkv_int8"] = _run_config(
        "int4_qkv_int8", B.SQUISH_MODEL_INT4,
        ["--kv-cache-mode", "int8", "--kv-cache-window", "64"], both, 0.0)

    # ── temp 0.7 operating point (spec decode is gated on temp>0) ─────────────
    results["configs"]["int4_t07_base"] = _run_config(
        "int4_t07_base", B.SQUISH_MODEL_INT4, [], short, 0.7)
    if has_draft:
        results["configs"]["int4_t07_spec"] = _run_config(
            "int4_t07_spec", B.SQUISH_MODEL_INT4,
            ["--draft-model", DRAFT_MODEL, "--draft-depth", "4"], short, 0.7)
    else:
        B.log(f"SPEC SKIPPED — draft model not found at {DRAFT_MODEL}")

    out = OUT_DIR / f"{B.TS}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    B.log(f"Wrote {out}")
    _print_summary(results)


def _print_summary(r: dict[str, Any]) -> None:
    c = r["configs"]
    ref = r["ref"]
    print()
    print("# Lever benchmark — squish on the fixed+pinned serving layer (M3 16 GB)")
    print(f"Reference (temp 0): ollama p75 {ref['ollama_p75_tps']} tok/s / "
          f"p95 {ref['ollama_p75_p95']}ms · "
          f"INT4 pre-pin p75 {ref['int4_p75_tps']} tok/s / p95 {ref['int4_p75_p95']}ms")
    print()
    print(f"{'config':<16} {'temp':>5} {'phase':>6} {'tok/s':>8} {'p50ms':>8} {'p95ms':>8}")
    print("-" * 58)
    for name, cfg in c.items():
        for pname, st in cfg["phases"].items():
            print(f"{name:<16} {cfg['temperature']:>5} {pname:>6} "
                  f"{_f(st['warm_tps']):>8} {_f(st['itl_p50']):>8} {_f(st['itl_p95']):>8}")
    # Spec-decode speedup callout
    if "int4_t07_spec" in c and "int4_t07_base" in c:
        b = c["int4_t07_base"]["phases"]["p75"]["warm_tps"]
        s = c["int4_t07_spec"]["phases"]["p75"]["warm_tps"]
        if b and s:
            print(f"\nSpec decode @temp0.7 p75: base {b:.1f} → draft {s:.1f} tok/s "
                  f"({s / b:.2f}x)")


if __name__ == "__main__":
    main()
