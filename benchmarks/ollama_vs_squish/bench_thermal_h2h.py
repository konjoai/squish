#!/usr/bin/env python3
"""Thermally-controlled head-to-head: ollama 0.18.2 vs current squish.

The plain single-session run (run_headtohead_018) is biased by RUN ORDER: the
M3 throttles under sustained load, so whichever engine runs later starts hotter
and looks slower.  Evidence: the bare INT4 daemon measured 20.8 tok/s at p2000
when it ran first (cool) but 15.1 when it ran third (hot) — same code.

This harness removes that bias by design (no sudo thermal API exists here):

  * COOLDOWN before every config — kill all serving and idle so each config
    starts from the same near-baseline temperature.  The cumulative heat from
    prior configs no longer carries over.
  * SETTLE between phases — a short idle so p4000 isn't measured right on the
    heat spike from p2000's prefills.
  * DRIFT CHECK — ollama is measured first AND last.  If ollama-first ≈
    ollama-last, the cooldowns held and the cross-config numbers are fair.

Order: ollama, squish configs, ollama_recheck.  Each starts cool, so order no
longer confers an advantage.  Output: results/benchmarks_v5_1_1/thermal/<UTC>.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import bench_v5_1 as B

# The true 0.18.2 binary (the /usr/local/bin app symlink auto-updated to 0.30.7).
B.OLLAMA_BIN = "/opt/homebrew/bin/ollama"

COOLDOWN_S    = 120     # idle before each config (servers down)
SETTLE_S      = 25      # idle between phases (same server up)
PHASES        = ["p75", "p2000", "p4000"]

# ollama_recheck reuses the ollama launcher/stream — the end-of-run drift probe.
B.CONFIGS["ollama_recheck"] = {**B.CONFIGS["ollama"],
                               "label": "Ollama 0.18.2 (recheck)"}
ORDER = ["ollama", "squish_daemon", "squish_recommended_int4",
         "squish_recommended_int3", "ollama_recheck"]

OUT_DIR = B.REPO_ROOT / "results" / "benchmarks_v5_1_1" / "thermal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _cooldown(seconds: int) -> None:
    B.kill_all_serving()            # tears down servers (+ its own 3s settle)
    B.log(f"  ❄ cooldown {seconds}s (idle, all servers down) …")
    time.sleep(seconds)


def run_config_thermal(cfg_id: str, prompts: dict[str, str]) -> dict[str, Any]:
    cfg = B.CONFIGS[cfg_id]
    B.log(f"=== {cfg_id} : cooldown → start ===")
    _cooldown(COOLDOWN_S)
    log_path = B.LOG_DIR / f"thermal_{cfg_id}_{B.TS}.log"
    proc, sampler = cfg["start"](log_path)
    phases: dict[str, Any] = {}
    try:
        if not B.wait_ready(cfg["ready_url"], timeout=300):
            raise RuntimeError(f"{cfg_id} did not become ready")
        cfg["stream"]("Hello.", max_tokens=4)        # trigger load + warm
        for pname in PHASES:
            B.log(f"  ⏸ settle {SETTLE_S}s before {pname}")
            time.sleep(SETTLE_S)
            B.log(f"  ── phase {pname} ──")
            phases[pname] = B.run_one_phase(cfg_id, prompts[pname], pname)
            es = phases[pname]["e2e_runs"]
            import statistics as _st
            tps = _st.median([x["tokens_per_sec"] for x in es if x["tokens_per_sec"]])
            i50 = _st.median([x["itl_p50_ms"] for x in es if x.get("itl_p50_ms")])
            i95 = _st.median([x["itl_p95_ms"] for x in es if x.get("itl_p95_ms")])
            B.log(f"    -> tps={tps:.1f}  itl_p50={i50:.1f}ms  itl_p95={i95:.1f}ms")
    finally:
        B.stop_server(proc, sampler)
    return {"label": cfg["label"], "quant": cfg.get("quant", "?"),
            "peak_rss_bytes": sampler.peak_bytes, "phases": phases}


def main() -> None:
    from mlx_lm import load
    _, tok = load(B.SQUISH_MODEL_INT4)
    prompts = {
        "p75":   B._P75,
        "p2000": B._build_prompt_to_tokens(B._P75, target_tokens=2000),
        "p4000": B._build_prompt_to_tokens(B._P75, target_tokens=4000),
    }
    toks = {k: len(tok.encode(v)) for k, v in prompts.items()}
    B.log(f"prompt tokens: {toks}")
    B.log(f"thermal control: cooldown={COOLDOWN_S}s/config, settle={SETTLE_S}s/phase")

    results: dict[str, Any] = {
        "timestamp": B.TS, "host": "Apple M3 16 GB",
        "ollama_bin": B.OLLAMA_BIN, "cooldown_s": COOLDOWN_S, "settle_s": SETTLE_S,
        "prompt_token_counts": toks, "runs_per_metric": B.RUNS,
        "configs": {}, "summary": {},
    }
    for cfg_id in ORDER:
        data = run_config_thermal(cfg_id, prompts)
        results["configs"][cfg_id] = data
        results["summary"][cfg_id] = B.summarize(data)

    out = OUT_DIR / f"{B.TS}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    B.log(f"Wrote {out}")
    _summary(results)


def _g(run, cfg, ph, key):
    try:
        m = run["summary"][cfg]["phases"][ph][key]
        return m.get("median") if isinstance(m, dict) else None
    except Exception:
        return None


def _summary(r: dict[str, Any]) -> None:
    print("\n# Thermally-controlled head-to-head — ollama 0.18.2 vs squish (M3 16 GB)")
    print(f"cooldown {r['cooldown_s']}s/config · settle {r['settle_s']}s/phase · "
          f"{r['runs_per_metric']} runs/metric\n")
    labels = {c: r["summary"][c].get("quant", "") for c in r["configs"]}
    for ph in PHASES:
        print(f"── {ph} ({r['prompt_token_counts'][ph]} tok) — warm tok/s | itl_p50 | itl_p95 ──")
        for c in ORDER:
            tps = _g(r, c, ph, "warm_tps")
            i50 = _g(r, c, ph, "itl_p50_ms")
            i95 = _g(r, c, ph, "itl_p95_ms")
            lab = r["configs"][c]["label"]
            s = (f"{tps:>5.1f} tok/s | {i50:>5.1f} | {i95:>6.1f} ms"
                 if tps and i50 and i95 else "   -")
            print(f"   {lab:<32} {s}")
        print()
    # Drift check
    o1 = _g(r, "ollama", "p75", "warm_tps")
    o2 = _g(r, "ollama_recheck", "p75", "warm_tps")
    if o1 and o2:
        drift = (o2 - o1) / o1 * 100
        print(f"DRIFT CHECK (p75 ollama first vs last): {o1:.1f} → {o2:.1f} tok/s "
              f"({drift:+.1f}%) — small = cooldowns held, comparison is fair.")


if __name__ == "__main__":
    main()
