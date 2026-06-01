#!/usr/bin/env python3
"""Cold-load profiling harness for the squish server.

Spawns a fresh Python subprocess for each measurement so interpreter
startup is included in t0. Phase checkpoints are emitted by
``profile_child.py`` and aggregated here.

Modes:
  cold (5 runs) — ``sudo purge`` between each run when permitted
  warm (3 runs) — no purge; OS page cache stays hot

Outputs:
  benchmarks/load_profile/results/cold_<timestamp>.json   — raw + medians
  benchmarks/load_profile/results/cold_<timestamp>.prof   — cProfile
                                                            (1 representative
                                                            cold run)
  benchmarks/load_profile/results/PROFILE_REPORT.md       — markdown table

The harness does NOT require sudo. If ``sudo -n purge`` fails (no NOPASSWD
entry, or sudo not installed) we fall back to "process restart only"
cold — the same definition the Ollama-vs-Squish bench uses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median, stdev
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VENV_PY = "/Users/wscholl/squish/.venv/bin/python"
DEFAULT_MODEL_DIR = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int4"

PHASE_ORDER = [
    "squish_imports",
    "mlx_core_import",
    "mlx_lm_import",
    "mlx_utils_import",
    "weights_loaded",
    "tokenizer_loaded",
    "warmup_done",
    "server_bound",
]


def purge_page_cache() -> bool:
    """Best-effort macOS page-cache flush. Returns True if purge ran."""
    if not shutil.which("purge"):
        return False
    # Try non-interactive sudo first; fall back to bare purge (may need root)
    for cmd in (["sudo", "-n", "purge"], ["purge"]):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def kill_servers() -> None:
    for pat in ("squish.server", "ollama serve", "ollama runner"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(2)


def one_run(model_dir: str, prof_path: str | None) -> dict[str, Any]:
    """Spawn one child run, return parsed checkpoint dict."""
    cmd = [VENV_PY, str(HERE / "profile_child.py"), "--model-dir", model_dir]
    if prof_path:
        cmd += ["--prof", prof_path]

    t_spawn = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None  # for mypy

    checkpoints: dict[str, float] = {}
    result_payload: dict[str, Any] | None = None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("CKPT "):
            rec = json.loads(line[5:])
            checkpoints[rec["phase"]] = rec["delta_s"]
        elif line.startswith("RESULT "):
            result_payload = json.loads(line[7:])
        # Other lines are ignored (uvicorn noise, warnings)

    proc.wait(timeout=60)
    t_exit = time.perf_counter()

    return {
        "spawn_to_exit_s": round(t_exit - t_spawn, 4),
        "rc":              proc.returncode,
        "checkpoints":     checkpoints,
        "result":          result_payload,
    }


def compute_phase_deltas(checkpoints: dict[str, float]) -> dict[str, float]:
    """Convert absolute-from-t0 checkpoints to per-phase deltas."""
    ordered = [(p, checkpoints[p]) for p in PHASE_ORDER if p in checkpoints]
    out: dict[str, float] = {}
    prev = 0.0
    for name, t in ordered:
        out[name] = round(t - prev, 4)
        prev = t
    return out


def median_of(values: list[float]) -> float:
    return round(median(values), 4) if values else 0.0


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute medians per checkpoint + per phase across runs."""
    cps: dict[str, list[float]] = {p: [] for p in PHASE_ORDER}
    for r in runs:
        for p, t in r["checkpoints"].items():
            if p in cps:
                cps[p].append(t)
    medians_cum = {p: median_of(v) for p, v in cps.items() if v}
    medians_per_phase: dict[str, float] = {}
    prev = 0.0
    for p in PHASE_ORDER:
        if p in medians_cum:
            medians_per_phase[p] = round(medians_cum[p] - prev, 4)
            prev = medians_cum[p]
    spawn_to_exit = [r["spawn_to_exit_s"] for r in runs]
    return {
        "median_cumulative_s": medians_cum,
        "median_per_phase_s":  medians_per_phase,
        "spawn_to_exit_s":     {
            "median": median_of(spawn_to_exit),
            "min":    round(min(spawn_to_exit), 4) if spawn_to_exit else 0,
            "max":    round(max(spawn_to_exit), 4) if spawn_to_exit else 0,
        },
        "n_runs": len(runs),
    }


def render_report(cold_agg: dict[str, Any], warm_agg: dict[str, Any], ts: str) -> str:
    lines = []
    lines.append(f"# Cold-load profile report  ({ts})")
    lines.append("")
    lines.append("Host: Apple M3 MacBook Pro 16 GB · macOS 25.5.0 · Python 3.14.3 · MLX-LM 0.31.1")
    lines.append(f"Model: `{DEFAULT_MODEL_DIR.split('/')[-1]}` (INT4 MLX safetensors)")
    lines.append("")
    lines.append("Times are cumulative seconds from process start (t0).")
    lines.append("`Δ` is the per-phase delta (this phase only).")
    lines.append("")
    lines.append(
        "| Phase             | Cold cum (s) | Cold Δ (s) | Warm cum (s) | Warm Δ (s) |"
    )
    lines.append(
        "|-------------------|-------------:|-----------:|-------------:|-----------:|"
    )
    for p in PHASE_ORDER:
        c_cum = cold_agg["median_cumulative_s"].get(p, 0.0)
        c_d   = cold_agg["median_per_phase_s"].get(p, 0.0)
        w_cum = warm_agg["median_cumulative_s"].get(p, 0.0)
        w_d   = warm_agg["median_per_phase_s"].get(p, 0.0)
        lines.append(
            f"| {p:<17} | {c_cum:>12.3f} | {c_d:>10.3f} | {w_cum:>12.3f} | {w_d:>10.3f} |"
        )
    lines.append("")
    lines.append(
        f"Subprocess wall (spawn→exit) — cold median {cold_agg['spawn_to_exit_s']['median']:.2f} s, "
        f"warm median {warm_agg['spawn_to_exit_s']['median']:.2f} s."
    )
    lines.append("")
    lines.append("## Per-run cumulative times")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--cold-runs", type=int, default=5)
    ap.add_argument("--warm-runs", type=int, default=3)
    ap.add_argument("--no-purge", action="store_true",
                    help="Skip sudo purge between cold runs (use when not authorised)")
    ap.add_argument("--label", default="",
                    help="Optional tag baked into output filenames (e.g. 'pre-fix', 'fix3')")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.label}" if args.label else ""
    out_json = RESULTS_DIR / f"cold_{ts}{tag}.json"
    prof_path = str(RESULTS_DIR / f"cold_{ts}{tag}.prof")

    purge_ok_first = False
    if not args.no_purge:
        purge_ok_first = purge_page_cache()
        if not purge_ok_first:
            print("[warn] sudo purge unavailable — cold runs will only restart the process",
                  file=sys.stderr)

    print(f"==> Cold phase: {args.cold_runs} runs", flush=True)
    cold_runs: list[dict[str, Any]] = []
    for i in range(args.cold_runs):
        kill_servers()
        if not args.no_purge:
            purge_page_cache()
        prof_for_run = prof_path if i == 0 else None  # cProfile only on run 1
        r = one_run(args.model_dir, prof_for_run)
        per_phase = compute_phase_deltas(r["checkpoints"])
        print(
            f"  cold[{i + 1}] rc={r['rc']} cum={r['checkpoints']} "
            f"per_phase={per_phase}",
            flush=True,
        )
        cold_runs.append(r)

    print(f"==> Warm phase: {args.warm_runs} runs (no purge)", flush=True)
    warm_runs: list[dict[str, Any]] = []
    for i in range(args.warm_runs):
        kill_servers()
        r = one_run(args.model_dir, None)
        per_phase = compute_phase_deltas(r["checkpoints"])
        print(
            f"  warm[{i + 1}] rc={r['rc']} cum={r['checkpoints']} "
            f"per_phase={per_phase}",
            flush=True,
        )
        warm_runs.append(r)

    cold_agg = aggregate(cold_runs)
    warm_agg = aggregate(warm_runs)

    out = {
        "timestamp": ts,
        "label":     args.label,
        "host":      "Apple M3 MacBook Pro 16GB",
        "model":     args.model_dir,
        "purge_supported": purge_ok_first,
        "cold": {
            "agg":  cold_agg,
            "runs": cold_runs,
        },
        "warm": {
            "agg":  warm_agg,
            "runs": warm_runs,
        },
    }
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_json}", flush=True)

    report = render_report(cold_agg, warm_agg, ts)
    report_path = RESULTS_DIR / "PROFILE_REPORT.md"
    report_path.write_text(report)
    print(f"Wrote {report_path}", flush=True)
    print(f"cProfile sample: {prof_path}", flush=True)
    print()
    print(report)


if __name__ == "__main__":
    main()
