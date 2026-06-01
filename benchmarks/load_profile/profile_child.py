#!/usr/bin/env python3
"""Inner profiler — runs in a fresh subprocess and emits checkpoint deltas.

Invoked by profile_cold_load.py. Reports phase boundaries on stdout as JSON
lines (one per checkpoint), then a final JSON summary line prefixed with
``RESULT:``. cProfile output is written to the path given by ``--prof``.

Phase checkpoints
─────────────────
  t0 : process start (recorded immediately at entry)
  t1 : squish package imports complete
  t2 : mlx + mlx_lm imports complete
  t3 : tokenizer loaded
  t4 : model weights instantiated + Metal evaluated
  t5 : warmup forward pass complete
  t6 : HTTP server bound, /health returned 200

Times are reported as deltas from t0 (seconds, perf_counter wall time).
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import sys
import time
from pathlib import Path

# t0: process start. Set as early as possible.
T0 = time.perf_counter()


def emit(name: str, delta: float, extra: dict | None = None) -> None:
    rec = {"phase": name, "delta_s": round(delta, 4)}
    if extra:
        rec.update(extra)
    print("CKPT " + json.dumps(rec), flush=True)


def run_load(model_dir: str, prof: cProfile.Profile | None) -> dict[str, float]:
    """Drive the load sequence with checkpoints between phases.

    Returns a dict {phase_name: delta_s_from_t0}.
    """
    if prof is not None:
        prof.enable()

    # ── t1: squish imports complete ──────────────────────────────────────
    import squish  # noqa: F401 — top-level package
    import squish.server as _srv  # noqa: F401
    t1 = time.perf_counter() - T0
    emit("squish_imports", t1)

    # ── t2: split into mlx.core, mlx_lm, and load_model/tokenizer imports ─
    import mlx.core as mx  # noqa: F401
    t2a = time.perf_counter() - T0
    emit("mlx_core_import", t2a)

    import mlx_lm  # noqa: F401
    t2b = time.perf_counter() - T0
    emit("mlx_lm_import", t2b)

    from mlx_lm.utils import load_model, load_tokenizer
    t2 = time.perf_counter() - T0
    emit("mlx_utils_import", t2)

    # ── t3 (tokenizer) and t4 (weights) — sequential timing so each phase
    # has a clean delta. The production load path in
    # ``squish.server.load_mlx_model`` runs these on a worker thread for
    # ~0.5 s of overlap; that gain is reflected by the Ollama-vs-Squish
    # bench, not by this per-phase report.
    model_path = Path(model_dir)

    model, config = load_model(model_path, lazy=False)
    t4 = time.perf_counter() - T0
    emit("weights_loaded", t4)

    tokenizer = load_tokenizer(
        model_path, None, eos_token_ids=config.get("eos_token_id", None)
    )
    t3 = time.perf_counter() - T0
    emit("tokenizer_loaded", t3)

    # Register state and run warmup so subsequent checkpoints match the
    # real squish.server cold-start sequence.
    _srv._state.model = model
    _srv._state.tokenizer = tokenizer
    _srv._state.model_name = model_path.name
    _srv._state.loaded_at = time.time()
    _srv._state.load_time_s = t4
    _srv._state.loader_tag = "mlx_lm"
    _srv._cap_metal_cache(verbose=False)
    _srv._warmup_model(verbose=False)
    _srv._cap_metal_cache(verbose=False)
    t5 = time.perf_counter() - T0
    emit("warmup_done", t5)

    # ── t6: bind HTTP server, wait for /health ──────────────────────────
    _srv._LOAD_COMPLETE.set()
    import threading
    import urllib.request
    import uvicorn

    port = 21435  # avoid collision with normal server
    config_uv = uvicorn.Config(_srv.app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config_uv)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Poll /health until 200
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as r:
                r.read()
                break
        except Exception:
            time.sleep(0.02)

    t6 = time.perf_counter() - T0
    emit("server_bound", t6)

    # Shut down the server so the subprocess exits cleanly
    server.should_exit = True

    if prof is not None:
        prof.disable()

    return {
        "squish_imports":   t1,
        "mlx_imports":      t2,
        "weights_loaded":   t4,
        "tokenizer_loaded": t3,
        "warmup_done":      t5,
        "server_bound":     t6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prof", default="")
    args = ap.parse_args()

    profiler = cProfile.Profile() if args.prof else None
    summary = run_load(args.model_dir, profiler)
    if profiler is not None:
        profiler.dump_stats(args.prof)

    # Final RESULT line (parsed by the parent harness)
    print("RESULT " + json.dumps({
        "summary": summary,
        "process_pid": os.getpid(),
        "model_dir": args.model_dir,
    }), flush=True)


if __name__ == "__main__":
    main()
