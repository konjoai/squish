#!/usr/bin/env python3
"""p4000-only thermal isolation: ollama 0.18.2 vs 0.30.7, each from a COLD start.

The full 5-config head-to-head (bench_thermal_h2h.py) measures ollama p4000 first
(cool) and ollama_recheck p4000 last (hot, after three squish configs).  That
back-half heat made the p4000 cross-version numbers untrustworthy.  This harness
removes the contamination: each ollama binary gets its own 120s cooldown and is the
ONLY load on the machine, so both are measured from the same near-baseline temp.

Sequence: 0.18.2 -> 0.30.7 -> 0.18.2 (recheck).  The recheck is the drift probe: if
0.18.2-first ~= 0.18.2-recheck, the cooldowns reset and 0.30.7 (measured in the
middle) sat on equal thermal footing.  Reuses bench_v5_1's measurement code verbatim
(same prompt, same RUNS, same TTFT/itl/e2e math).

NOTE: the published 0.18.2-vs-0.30.7 numbers were produced with the repo at commit
37bac10 (the exact tree that produced the 0.30.7 baseline), swapping only the ollama
binary.  Run from benchmarks/ollama_vs_squish with the repo venv:

    BENCH_OLLAMA_018=/opt/homebrew/bin/ollama \\
    BENCH_OLLAMA_307=/usr/local/bin/ollama \\
    BENCH_OLLAMA_MODEL=qwen2.5:7b \\
    ../../.venv/bin/python bench_p4000_iso.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

import bench_v5_1 as B

BIN_018 = os.environ.get("BENCH_OLLAMA_018", "/opt/homebrew/bin/ollama")
BIN_307 = os.environ.get("BENCH_OLLAMA_307", "/usr/local/bin/ollama")
if os.environ.get("BENCH_OLLAMA_MODEL"):
    B.OLLAMA_MODEL = os.environ["BENCH_OLLAMA_MODEL"]

COOLDOWN_S = 120
SETTLE_S = 25
PHASE = "p4000"
SEQ = [("ollama_018", BIN_018), ("ollama_307", BIN_307), ("ollama_018_recheck", BIN_018)]

OUT_DIR = B.REPO_ROOT / "results" / "benchmarks_v5_1_1" / "thermal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ver(binpath: str) -> str:
    out = subprocess.run([binpath, "--version"], capture_output=True, text=True).stdout
    return out.strip().replace("\n", " | ")


def _cooldown(seconds: int) -> None:
    B.kill_all_serving()
    B.log(f"  cooldown {seconds}s (idle, all servers down) ...")
    time.sleep(seconds)


def run_p4000(run_id: str, binpath: str, prompt: str) -> dict[str, Any]:
    B.OLLAMA_BIN = binpath  # start_ollama reads this global live
    cfg = B.CONFIGS["ollama"]
    B.log(f"=== {run_id} ({binpath}) : cooldown -> start ===")
    _cooldown(COOLDOWN_S)
    log_path = B.LOG_DIR / f"p4000iso_{run_id}_{B.TS}.log"
    proc, sampler = cfg["start"](log_path)
    try:
        if not B.wait_ready(cfg["ready_url"], timeout=300):
            raise RuntimeError(f"{run_id} did not become ready")
        cfg["stream"]("Hello.", max_tokens=4)  # trigger load + warm
        B.log(f"  settle {SETTLE_S}s before {PHASE}")
        time.sleep(SETTLE_S)
        B.log(f"  -- {run_id} phase {PHASE} --")
        phase = B.run_one_phase("ollama", prompt, PHASE)
        import statistics as _st

        es = phase["e2e_runs"]
        tps = _st.median([x["tokens_per_sec"] for x in es if x["tokens_per_sec"]])
        B.log(f"    -> tps={tps:.2f}")
    finally:
        B.stop_server(proc, sampler)
    return {"label": f"Ollama {run_id}", "quant": "q4_K_M",
            "peak_rss_bytes": sampler.peak_bytes, "phases": {PHASE: phase}}


def main() -> None:
    from mlx_lm import load
    _tok_dir = os.environ.get("BENCH_TOKENIZER_DIR", B.SQUISH_MODEL_INT4)
    _, tok = load(_tok_dir)
    prompt = B._build_prompt_to_tokens(B._P75, target_tokens=4000)
    ntok = len(tok.encode(prompt))
    B.log(f"p4000 prompt tokens: {ntok}")
    B.log(f"isolation: cooldown={COOLDOWN_S}s, settle={SETTLE_S}s, phase={PHASE} only, "
          f"runs/metric={B.RUNS}")

    results: dict[str, Any] = {
        "timestamp": B.TS, "host": "Apple M3 16 GB", "mode": "p4000-isolation",
        "cooldown_s": COOLDOWN_S, "settle_s": SETTLE_S, "prompt_tokens": ntok,
        "runs_per_metric": B.RUNS,
        "versions": {"ollama_018": _ver(BIN_018), "ollama_307": _ver(BIN_307)},
        "configs": {}, "summary": {},
    }
    B.log(f"versions: 018={results['versions']['ollama_018']}")
    B.log(f"versions: 307={results['versions']['ollama_307']}")
    for run_id, binpath in SEQ:
        data = run_p4000(run_id, binpath, prompt)
        results["configs"][run_id] = data
        results["summary"][run_id] = B.summarize(data)

    out = OUT_DIR / f"{B.TS}_p4000iso.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    B.log(f"Wrote {out}")
    _print(results)


def _g(r: dict[str, Any], cid: str, key: str) -> float | None:
    try:
        m = r["summary"][cid]["phases"][PHASE][key]
        return m.get("median") if isinstance(m, dict) else None
    except Exception:
        return None


def _print(r: dict[str, Any]) -> None:
    print("\n# p4000 isolation - ollama 0.18.2 vs 0.30.7 (M3 16 GB), each from cold")
    print(f"cooldown {r['cooldown_s']}s . settle {r['settle_s']}s . "
          f"{r['runs_per_metric']} runs/metric . p4000={r['prompt_tokens']} tok\n")
    print(f"{'config':<22} {'warm tok/s':>10} {'itl_p50':>8} {'itl_p95':>9} "
          f"{'E2E-200':>8} {'TTFT ms':>8}  peakRSS")
    for cid in r["configs"]:
        tps = _g(r, cid, "warm_tps")
        i50 = _g(r, cid, "itl_p50_ms")
        i95 = _g(r, cid, "itl_p95_ms")
        e2e = _g(r, cid, "e2e_200tok_s")
        ttft = _g(r, cid, "ttft_s")
        rss = r["configs"][cid].get("peak_rss_bytes", 0) / 1e9
        print(f"{cid:<22} {tps:>10.2f} {i50:>8.1f} {i95:>9.1f} "
              f"{e2e:>8.2f} {ttft * 1000:>8.0f}  {rss:>5.2f}GB")
    a = _g(r, "ollama_018", "warm_tps")
    b = _g(r, "ollama_018_recheck", "warm_tps")
    if a and b:
        print(f"\nDRIFT (0.18.2 first->recheck p4000): {a:.2f} -> {b:.2f} tok/s "
              f"({(b - a) / a * 100:+.1f}%) - small = cooldowns reset, "
              f"0.30.7 measured on equal thermal footing.")


if __name__ == "__main__":
    main()
