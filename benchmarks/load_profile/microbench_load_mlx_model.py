#!/usr/bin/env python3
"""Time ``squish.server.load_mlx_model`` end-to-end in a fresh process.

Cheaper than the full Ollama-vs-Squish bench, designed to validate the
parallel-tokenizer fix and any other change to the production load
path. Each run is a fresh subprocess.

Outputs medians to stdout. No file artifacts.

Usage:
    python benchmarks/load_profile/microbench_load_mlx_model.py [--runs N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from statistics import median

VENV_PY = "/Users/wscholl/squish/.venv/bin/python"
DEFAULT_MODEL = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int4"

CHILD = textwrap.dedent("""
    import json
    import sys
    import time
    import squish.server as _srv
    model_dir = sys.argv[1]
    t0 = time.perf_counter()
    _srv.load_mlx_model(model_dir, verbose=False)
    elapsed = time.perf_counter() - t0
    print("LOAD_TIME " + json.dumps({"load_mlx_model_s": elapsed}), flush=True)
""")


def one_run(model_dir: str) -> float:
    proc = subprocess.run(
        [VENV_PY, "-c", CHILD, model_dir],
        capture_output=True, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout.splitlines():
        if line.startswith("LOAD_TIME "):
            return float(json.loads(line[len("LOAD_TIME "):])["load_mlx_model_s"])
    raise RuntimeError(f"child did not emit LOAD_TIME line.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    samples: list[float] = []
    for i in range(args.runs):
        # kill any prior server process (best-effort)
        subprocess.run(["pkill", "-f", "squish.server"], capture_output=True)
        time.sleep(1)
        s = one_run(args.model_dir)
        samples.append(s)
        print(f"  run[{i + 1}]: load_mlx_model = {s:.3f} s", flush=True)

    samples_sorted = sorted(samples)
    print()
    print(f"runs:   {samples}")
    print(f"median: {median(samples):.3f} s")
    print(f"min:    {min(samples):.3f} s")
    print(f"max:    {max(samples):.3f} s")


if __name__ == "__main__":
    main()
