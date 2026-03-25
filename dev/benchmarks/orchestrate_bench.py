#!/usr/bin/env python3
"""
orchestrate_bench.py — start a squish server, run _run_bench_pass.py, stop.

Usage:
    python3 dev/benchmarks/orchestrate_bench.py \
        --pass-label "blazing (no agent)" \
        --port 11435 \
        --server-flags "--blazing --int2" \
        --model models/Qwen2.5-7B-Instruct-int2 \
        --out /tmp/pass_a.json
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).parent / "_run_bench_pass.py"
API = "squish"

ap = argparse.ArgumentParser()
ap.add_argument("--pass-label",    default="pass")
ap.add_argument("--port",          type=int, default=11435)
ap.add_argument("--server-flags",  default="--blazing --int2")
ap.add_argument("--model",         default="models/Qwen2.5-7B-Instruct-int2")
ap.add_argument("--out",           default="")
ap.add_argument("--wait",          type=int, default=120)
ap.add_argument("--verbose",       action="store_true")
args = ap.parse_args()

G  = "\033[32m"; R = "\033[31m"; C = "\033[36m"; W = "\033[1;37m"
D  = "\033[2m"; NC = "\033[0m"


def _wait_ready(port: int, max_wait: float) -> float:
    """Return elapsed seconds when ready, or -1 on timeout."""
    t0 = time.perf_counter()
    deadline = t0 + max_wait
    url = f"http://127.0.0.1:{port}/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API}"})
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        print(f"  {D}  waiting… ({time.perf_counter()-t0:.0f}s/{max_wait}s){NC}",
              end="\r")
        time.sleep(3)
    return -1.0


# ── Build server command ──────────────────────────────────────────────────────
model_path = str(REPO / args.model)
squish_bin = "squish"

server_cmd = [squish_bin, "serve", model_path,
              "--port", str(args.port), "--no-browser", "--log-level", "warning"]
server_cmd += shlex.split(args.server_flags)

print(f"\n{C}{'='*66}{NC}")
print(f"{W}  Starting server: {args.pass_label}{NC}")
print(f"  {D}{' '.join(server_cmd)}{NC}")

proc = subprocess.Popen(
    server_cmd,
    cwd=str(REPO),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)

t_load = _wait_ready(args.port, args.wait)
if t_load < 0:
    stderr_out = ""
    try:
        proc.terminate()
        _, stderr_bytes = proc.communicate(timeout=10)
        stderr_out = stderr_bytes.decode(errors="replace")[-1000:]
    except Exception:
        pass
    print(f"\n{R}  ERROR: server did not start within {args.wait}s{NC}")
    if stderr_out:
        print(f"  stderr: {stderr_out}")
    sys.exit(1)

print(f"\n  {G}Server ready in {t_load:.1f}s{NC}  (port {args.port})")

# ── Warm-up ping ──────────────────────────────────────────────────────────────
try:
    import json
    import urllib.request as ur  # noqa: F811
    body = json.dumps({"model":"squish","messages":[{"role":"user","content":"Hello"}],
                       "max_tokens":4,"stream":False}).encode()
    ur.urlopen(ur.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=body,
        headers={"Content-Type":"application/json","Authorization":f"Bearer {API}"},
    ), timeout=60)
    print(f"  {G}Warm-up done{NC}")
except Exception as e:
    print(f"  {D}Warm-up skipped: {e}{NC}")

# ── Run benchmark pass ────────────────────────────────────────────────────────
runner_cmd = [sys.executable, str(RUNNER),
              "--port", str(args.port),
              "--label", args.pass_label]
if args.out:
    runner_cmd += ["--out", args.out]
if args.verbose:
    runner_cmd.append("--verbose")

ret = subprocess.run(runner_cmd, cwd=str(REPO))

# ── Stop server ───────────────────────────────────────────────────────────────
proc.terminate()
try:
    proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    proc.kill()
print(f"\n  {D}Server stopped.{NC}")

sys.exit(ret.returncode)
