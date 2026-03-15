#!/usr/bin/env python3
"""
bench_mxcompile_ffn.py — mx.compile FFN fusion speedup benchmark.

Measures the throughput improvement (ms / forward-pass) that ``mx.compile``
provides for a SwiGLU-based Feed-Forward Network (FFN), and exposes hooks for
patching a real loaded model with ``patch_model_compiled_ffn``.

Stage 1 — synthetic numpy FFN timing (always runs; no GPU/model required):
  baseline_ms    average elapsed time per FFN forward pass [ms]
  note           reminder that mx.compile gains require a loaded model

Stage 2 — compiled FFN patch (requires --model-dir and mlx + mlx_lm):
  compiled_ms    average elapsed time per patched forward pass [ms]
  speedup_x      baseline_ms / compiled_ms

Results are written to dev/results/mxcompile_ffn_bench.json and an ASCII
summary table is printed to stdout.

Usage
-----
  python3 dev/benchmarks/bench_mxcompile_ffn.py
  python3 dev/benchmarks/bench_mxcompile_ffn.py --model-dir models/Qwen2.5-1.5B
  python3 dev/benchmarks/bench_mxcompile_ffn.py --n-warmup 5 --n-iters 20
  python3 dev/benchmarks/bench_mxcompile_ffn.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / "dev" / "results" / "mxcompile_ffn_bench.json"

# ── colour helpers (same convention as bench_2bit.py) ─────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

# ── try to import patch_model_compiled_ffn (graceful skip) ───────────────────
try:
    from squish.hardware.fused_kernels import patch_model_compiled_ffn  # type: ignore[import]
    _FUSED_AVAILABLE = True
except Exception:
    patch_model_compiled_ffn = None  # type: ignore[assignment]
    _FUSED_AVAILABLE = False

# ── FFN dimensions for the synthetic benchmark ────────────────────────────────
_FFN_IN   = 4096
_FFN_GATE = 16384   # intermediate (gated) dimension — typical 4× expansion
_FFN_OUT  = 4096
_BATCH    = 1       # single-token decode scenario


# ── display helpers ───────────────────────────────────────────────────────────

def _hdr(title: str) -> None:
    print(f"\n{W}{'─' * 72}{NC}")
    print(f"{C}  {title}{NC}")
    print(f"{W}{'─' * 72}{NC}")


def _row(label: str, val: str, extra: str = "") -> None:
    print(f"  {label:<48} {G}{val:>16}{NC}  {D}{extra}{NC}")


def _skip(label: str, reason: str = "") -> None:
    print(f"  {Y}~ SKIP{NC}  {label:<48} {D}{reason}{NC}")


def _err(label: str, reason: str = "") -> None:
    print(f"  {R}✗ ERROR{NC}  {label:<46} {D}{reason}{NC}")


# ── numpy SwiGLU helpers ──────────────────────────────────────────────────────

def _swiglu(x: np.ndarray) -> np.ndarray:
    """SwiGLU activation: gate half * SiLU(gate half) applied element-wise.

    Splits x along the last axis into (gate, up), then returns gate * silu(up).
    """
    half = x.shape[-1] // 2
    gate = x[..., :half]
    up   = x[..., half:]
    # SiLU: x * sigmoid(x)
    silu_up = up * (1.0 / (1.0 + np.exp(-up)))
    return gate * silu_up


class _NumpyFFN:
    """Simple 2-layer FFN with SwiGLU gate (Linear → SwiGLU → Linear).

    Weights are random float32; this models the gated projection pattern used
    in Llama / Qwen style transformer FFNs:
        y = W_down @ SwiGLU( [W_gate @ x ; W_up @ x] )
    which is implemented here as a single up-projection to 2 * gate_dim
    followed by SwiGLU and a down-projection.
    """

    def __init__(
        self,
        in_dim: int = _FFN_IN,
        gate_dim: int = _FFN_GATE,
        out_dim: int = _FFN_OUT,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng(0)
        # W_up_gate: projects in_dim → 2 * gate_dim (gate + up interleaved)
        scale = float(in_dim) ** -0.5
        self.W_up_gate = (rng.standard_normal((2 * gate_dim, in_dim)) * scale).astype(np.float32)
        # W_down: projects gate_dim → out_dim
        scale_d = float(gate_dim) ** -0.5
        self.W_down = (rng.standard_normal((out_dim, gate_dim)) * scale_d).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x (batch, in_dim) → (batch, out_dim)."""
        h = x @ self.W_up_gate.T    # (batch, 2 * gate_dim)
        h = _swiglu(h)               # (batch, gate_dim)
        return h @ self.W_down.T     # (batch, out_dim)


# ── per-scenario benchmark ────────────────────────────────────────────────────

def bench_ffn_throughput(
    x: np.ndarray,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    """Benchmark the numpy FFN forward pass.

    Parameters
    ----------
    x : np.ndarray, shape (batch, in_dim)
        Input activation tensor.
    n_warmup : int
        Number of warm-up iterations (not timed).
    n_iters : int
        Number of measured iterations.

    Returns
    -------
    dict with keys:
        avg_ms    average elapsed time per forward pass [ms]
        min_ms    minimum elapsed time per forward pass [ms]
        max_ms    maximum elapsed time per forward pass [ms]
        n_iters   number of measured iterations
    """
    ffn = _NumpyFFN(in_dim=x.shape[-1])

    # Warm-up
    for _ in range(n_warmup):
        _ = ffn.forward(x)

    elapsed: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = ffn.forward(x)
        elapsed.append((time.perf_counter() - t0) * 1e3)

    return {
        "avg_ms": round(sum(elapsed) / len(elapsed), 6),
        "min_ms": round(min(elapsed), 6),
        "max_ms": round(max(elapsed), 6),
        "n_iters": n_iters,
    }


# ── synthetic benchmark ───────────────────────────────────────────────────────

def run_synthetic_bench(n_warmup: int = 5, n_iters: int = 20) -> dict:
    """Benchmark a 4096→16384→4096 SwiGLU FFN without/with mx.compile.

    The numpy path always runs.  The mx.compile timing is deferred to a real
    model load because mx.compile operates on the MLX graph; it is not
    measurable on a standalone numpy forward pass.

    Returns
    -------
    dict with keys:
        baseline_ms   average ms per numpy FFN forward pass
        note          explanation that mx.compile requires --model-dir
        ffn_dims      dict of in/gate/out dimensions used
        n_warmup      warm-up iterations used
        n_iters       timed iterations used
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((_BATCH, _FFN_IN)).astype(np.float32)

    timing = bench_ffn_throughput(x, n_warmup=n_warmup, n_iters=n_iters)

    return {
        "baseline_ms": timing["avg_ms"],
        "baseline_min_ms": timing["min_ms"],
        "baseline_max_ms": timing["max_ms"],
        "note": "mx.compile speedup requires a loaded model; see --model-dir",
        "ffn_dims": {"in": _FFN_IN, "gate": _FFN_GATE, "out": _FFN_OUT},
        "n_warmup": n_warmup,
        "n_iters": n_iters,
    }


# ── model-level compiled FFN benchmark (Stage 2) ─────────────────────────────

def _run_compiled_ffn_bench(
    model_dir: str,
    n_warmup: int,
    n_iters: int,
    baseline_ms: float,
) -> Optional[dict]:
    """Patch a real mlx_lm model with compiled FFN and measure throughput.

    Returns None if mlx / mlx_lm are unavailable or model load fails.
    """
    if not _FUSED_AVAILABLE:
        return None

    try:
        import mlx.core as mx                # type: ignore[import]
        from mlx_lm import load              # type: ignore[import]
    except ImportError:
        return None

    try:
        model, tokenizer = load(model_dir)
    except Exception as exc:
        print(f"{R}  [compiled-ffn] failed to load model: {exc}{NC}")
        return None

    try:
        n_patched = patch_model_compiled_ffn(model)
        print(f"  Patched {n_patched} FFN layer(s) with compiled kernel.")
    except Exception as exc:
        print(f"{R}  [compiled-ffn] patch failed: {exc}{NC}")
        return None

    # Warm-up + measure via tokenised forward passes
    prompt = "The quick brown fox jumps over the lazy dog."
    try:
        token_ids = tokenizer.encode(prompt)[:8]
        input_ids = mx.array(token_ids)[None]

        for _ in range(n_warmup):
            _ = model(input_ids)
            mx.eval(_)

        elapsed: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            out = model(input_ids)
            mx.eval(out)
            elapsed.append((time.perf_counter() - t0) * 1e3)

        compiled_ms = sum(elapsed) / len(elapsed)
        speedup = baseline_ms / compiled_ms if compiled_ms > 0 else float("nan")

        return {
            "compiled_ms": round(compiled_ms, 6),
            "speedup_x": round(speedup, 4),
            "n_patched_layers": n_patched,
        }
    except Exception as exc:
        print(f"{R}  [compiled-ffn] inference failed: {exc}{NC}")
        return None


# ── summary table ─────────────────────────────────────────────────────────────

def _print_summary(synth: dict, compiled: Optional[dict]) -> None:
    _hdr("mx.compile FFN fusion  —  benchmark summary")

    _row("Numpy baseline FFN (4096→16384→4096, SwiGLU)",
         f"{synth['baseline_ms']:.4f} ms",
         f"min={synth['baseline_min_ms']:.4f} ms  max={synth['baseline_max_ms']:.4f} ms")

    if compiled:
        _row("Compiled FFN (mx.compile, real model)",
             f"{compiled['compiled_ms']:.4f} ms",
             f"speedup {compiled['speedup_x']:.2f}x  "
             f"({compiled['n_patched_layers']} layers patched)")
    else:
        _skip("Compiled FFN speedup",
              "requires --model-dir with mlx + mlx_lm  (or patch_model_compiled_ffn unavailable)")

    note = synth.get("note", "")
    if note:
        print(f"\n  {D}Note: {note}{NC}")

    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark mx.compile FFN fusion speedup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Path to an mlx_lm model directory.  Required for the compiled-FFN "
             "Stage 2 benchmark; Stage 1 (numpy) always runs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="FILE",
        help="Path for the JSON results file.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        metavar="N",
        help="Number of warm-up iterations (not timed).",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=20,
        metavar="N",
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and report plan without running benchmarks.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fused_status = f"{G}available{NC}" if _FUSED_AVAILABLE else f"{Y}unavailable{NC}"

    print(f"{C}bench_mxcompile_ffn.py{NC}  —  mx.compile FFN fusion speedup")
    print(f"  platform        : {platform.platform()}")
    print(f"  python          : {sys.version.split()[0]}")
    print(f"  output          : {output_path}")
    print(f"  fused_kernels   : {fused_status}")
    print(f"  n-warmup        : {args.n_warmup}")
    print(f"  n-iters         : {args.n_iters}")
    print(f"  FFN dims        : {_FFN_IN} → {_FFN_GATE} → {_FFN_OUT}  (SwiGLU)")

    if args.dry_run:
        print(f"\n{Y}  [dry-run] no benchmarks will be run.{NC}")
        return

    # Stage 1 — synthetic numpy benchmark
    _hdr("Stage 1 — synthetic numpy FFN throughput")
    print(f"  Shape: ({_BATCH}, {_FFN_IN})  n_warmup={args.n_warmup}  n_iters={args.n_iters}")

    synth = run_synthetic_bench(n_warmup=args.n_warmup, n_iters=args.n_iters)

    # Stage 2 — compiled FFN with real model (optional)
    compiled: Optional[dict] = None
    if args.model_dir:
        _hdr("Stage 2 — compiled FFN (mx.compile, real model)")
        print(f"  Model directory : {args.model_dir}")
        compiled = _run_compiled_ffn_bench(
            args.model_dir,
            n_warmup=args.n_warmup,
            n_iters=args.n_iters,
            baseline_ms=synth["baseline_ms"],
        )
        if compiled is None:
            _skip("compiled FFN", "mlx/mlx_lm unavailable or model load failed")
    else:
        _skip("compiled FFN (Stage 2)", "no --model-dir provided")

    # Print summary
    _print_summary(synth, compiled)

    # Collect and save results
    output: dict[str, Any] = {
        "meta": {
            "platform": platform.platform(),
            "python": sys.version,
            "n_warmup": args.n_warmup,
            "n_iters": args.n_iters,
            "fused_kernels_available": _FUSED_AVAILABLE,
        },
        "synthetic": synth,
    }
    if compiled is not None:
        output["compiled_ffn"] = compiled

    output_path.write_text(json.dumps(output, indent=2))
    print(f"{G}  Results saved to {output_path}{NC}\n")


if __name__ == "__main__":
    main()
