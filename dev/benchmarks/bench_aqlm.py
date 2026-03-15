#!/usr/bin/env python3
"""
bench_aqlm.py — AQLM 2-bit quantisation accuracy benchmark vs INT4 baseline.

Evaluates AQLM (Additive Quantization of Language Models) against a simple
INT4 per-row scalar baseline on the same weight matrices:

  aqlm   Additive Quantization of Language Models   (ICML 2024, Phase 9A)
  int4   4-bit scalar quantization                   (squish baseline, bpw=4.0)

Stage 1 — weight-reconstruction (always runs; no GPU/model required):
  bpw            bits per weight after compression
  snr_db         signal-to-noise ratio of reconstructed vs. original [dB]
  compress_ms    wall-clock time for the compress step [ms]
  decompress_ms  wall-clock time for the decompress step [ms]

Stage 2 — model evaluation (requires --model-dir and mlx + mlx_lm):
  perplexity     wikitext-2 test perplexity (exp of mean token NLL)

Results are written to dev/results/aqlm_bench.json and an ASCII summary
table is printed to stdout.

Usage
-----
  python3 dev/benchmarks/bench_aqlm.py
  python3 dev/benchmarks/bench_aqlm.py --model-dir models/Qwen2.5-1.5B
  python3 dev/benchmarks/bench_aqlm.py --output dev/results/aqlm_bench.json
  python3 dev/benchmarks/bench_aqlm.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / "dev" / "results" / "aqlm_bench.json"

# ── colour helpers (same convention as bench_2bit.py) ─────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

RNG = np.random.default_rng(42)

# ── benchmark constants ───────────────────────────────────────────────────────
BENCH_ROWS = 64   # synthetic weight-matrix height
BENCH_COLS = 64   # synthetic weight-matrix width
N_MATRICES = 10   # number of random matrices in the synthetic sweep


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


# ── SNR helper ────────────────────────────────────────────────────────────────

def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in dB.

    SNR_dB = 10 * log10( E[W²] / E[(W − Ŵ)²] )

    Higher is better.
    """
    orig  = original.astype(np.float64).ravel()
    recon = reconstructed.astype(np.float64).ravel()
    n = min(orig.size, recon.size)
    orig, recon = orig[:n], recon[:n]

    sig_power   = float(np.mean(orig ** 2))
    noise_power = float(np.mean((orig - recon) ** 2))

    if noise_power == 0.0:
        return float("inf")
    if sig_power == 0.0:
        return float("-inf")
    return 10.0 * math.log10(sig_power / noise_power)


# ── per-method benchmarks ─────────────────────────────────────────────────────

def bench_weight_snr(W: np.ndarray, cfg=None) -> dict:
    """Compress W with AQLMQuantizer and measure reconstruction quality.

    Parameters
    ----------
    W : np.ndarray, shape (rows, cols), dtype float32
        Weight matrix to compress.
    cfg : AQLMConfig, optional
        AQLM configuration.  Defaults to AQLMConfig() (2 codebooks, k=16,
        group_size=8).

    Returns
    -------
    dict with keys:
        method        "aqlm"
        status        "ok" | "error"
        bpw           bits per weight (index storage + codebook overhead)
        snr_db        reconstruction SNR [dB]
        compress_ms   wall-clock compress time [ms]
        decompress_ms wall-clock decompress time [ms]
        reason        error message if status == "error"
    """
    from squish.quant.aqlm import AQLMConfig, AQLMQuantizer  # local import

    W = np.asarray(W, dtype=np.float32)
    out_features, in_features = W.shape

    if cfg is None:
        cfg = AQLMConfig()

    quantizer = AQLMQuantizer(cfg)

    try:
        t0 = time.perf_counter()
        layer = quantizer.compress(W)
        compress_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        W_hat = quantizer.decompress(layer)
        decompress_ms = (time.perf_counter() - t0) * 1e3

        total_weights = out_features * in_features
        bpw = layer.compressed_bits / total_weights if total_weights > 0 else float("nan")
        snr = _snr_db(W, W_hat)

        return {
            "method": "aqlm",
            "status": "ok",
            "bpw": round(bpw, 4),
            "snr_db": round(snr, 4),
            "compress_ms": round(compress_ms, 4),
            "decompress_ms": round(decompress_ms, 4),
        }
    except Exception as exc:
        return {"method": "aqlm", "status": "error", "reason": str(exc)}


def bench_int4_snr(W: np.ndarray) -> dict:
    """INT4 baseline — quantize per-row with symmetric scale, measure SNR.

    Bits per weight is fixed at 4.0 (scale overhead is negligible for large
    matrices; a small per-row overhead of 32/cols is added).

    Parameters
    ----------
    W : np.ndarray, shape (rows, cols), dtype float32

    Returns
    -------
    dict with keys:
        method        "int4"
        status        "ok" | "error"
        bpw           4.0 + scale overhead
        snr_db        reconstruction SNR [dB]
        compress_ms   wall-clock quantize time [ms]
        decompress_ms wall-clock dequantize time [ms]
    """
    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape

    try:
        t0 = time.perf_counter()

        # Per-row symmetric INT4: scale = max(|W|) / 7
        w_abs_max = np.abs(W).max(axis=1, keepdims=True)  # (rows, 1)
        scales = np.where(w_abs_max == 0.0, 1.0, w_abs_max / 7.0)

        W_scaled = W / scales                              # normalise to [-7, 7]
        q = np.round(W_scaled).clip(-7, 7).astype(np.int8)
        compress_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        W_hat = (q.astype(np.float32)) * scales
        decompress_ms = (time.perf_counter() - t0) * 1e3

        # bpw: 4 bits / weight + 32-bit float scale per row
        bpw = 4.0 + (32.0 / cols)
        snr = _snr_db(W, W_hat)

        return {
            "method": "int4",
            "status": "ok",
            "bpw": round(bpw, 4),
            "snr_db": round(snr, 4),
            "compress_ms": round(compress_ms, 4),
            "decompress_ms": round(decompress_ms, 4),
        }
    except Exception as exc:
        return {"method": "int4", "status": "error", "reason": str(exc)}


# ── synthetic benchmark sweep ─────────────────────────────────────────────────

def run_synthetic_bench() -> list[dict]:
    """Run AQLM and INT4 benchmarks on N_MATRICES random (BENCH_ROWS, BENCH_COLS) matrices.

    Returns
    -------
    list[dict]
        One dict per (matrix_index, method) pair, N_MATRICES * 2 entries total.
    """
    results: list[dict] = []

    for i in range(N_MATRICES):
        W = RNG.standard_normal((BENCH_ROWS, BENCH_COLS)).astype(np.float32)

        aqlm_r = bench_weight_snr(W)
        aqlm_r["matrix_index"] = i
        results.append(aqlm_r)

        int4_r = bench_int4_snr(W)
        int4_r["matrix_index"] = i
        results.append(int4_r)

    return results


# ── perplexity helper (requires mlx + mlx_lm) ─────────────────────────────────

_WIKITEXT_SAMPLE = (
    "= Valkyria Chronicles III = \n"
    "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , "
    "lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles "
    "III outside Japan , is a tactical role @-@ playing video game developed by Sega and "
    "Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is "
    "the third game in the Valkyria series . Employing the same fusion of tactical and "
    "real @-@ time gameplay as its predecessors , the story runs parallel to the first game "
    "and follows the Nameless , a penal military unit serving the nation of Gallia during "
    "the Second Europan War who perform missions too sensitive for the regular army to carry "
    "out directly ."
)


def _run_perplexity(model_dir: str, n_tokens: int) -> Optional[float]:
    """Compute wikitext-2 perplexity using a loaded mlx_lm model.

    Returns None if mlx / mlx_lm are not installed.
    """
    try:
        import mlx.core as mx          # type: ignore[import]
        import mlx_lm                  # type: ignore[import]
        from mlx_lm import load        # type: ignore[import]
    except ImportError:
        return None

    try:
        model, tokenizer = load(model_dir)
    except Exception as exc:
        print(f"{R}  [perplexity] failed to load model: {exc}{NC}")
        return None

    text = _WIKITEXT_SAMPLE
    try:
        tokens = tokenizer.encode(text)
    except Exception:
        tokens = list(range(n_tokens))  # fallback dummy tokens

    tokens = tokens[:n_tokens]
    if len(tokens) < 2:
        return None

    input_ids = mx.array(tokens[:-1])[None]  # (1, T-1)
    target_ids = tokens[1:]

    try:
        logits = model(input_ids)
        if hasattr(logits, "logits"):
            logits = logits.logits
        # logits: (1, T-1, vocab_size) or (T-1, vocab_size)
        logits = mx.array(logits).reshape(-1, logits.shape[-1])
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll_vals = [-float(log_probs[t, target_ids[t]]) for t in range(len(target_ids))]
        mean_nll = sum(nll_vals) / len(nll_vals)
        return math.exp(mean_nll)
    except Exception as exc:
        print(f"{R}  [perplexity] inference failed: {exc}{NC}")
        return None


# ── summary table ─────────────────────────────────────────────────────────────

def _print_summary(results: list[dict], ppl: Optional[float]) -> None:
    """Print an ASCII summary of the synthetic benchmark results."""
    _hdr("AQLM vs INT4  —  weight-reconstruction benchmark")

    by_method: dict[str, list[dict]] = {}
    for r in results:
        by_method.setdefault(r["method"], []).append(r)

    print(f"\n  {'Method':<10} {'Avg SNR (dB)':>14} {'Avg BPW':>10} "
          f"{'Avg Compress (ms)':>20} {'Avg Decompress (ms)':>22}")
    print(f"  {'-'*10} {'-'*14} {'-'*10} {'-'*20} {'-'*22}")

    for method, rows in sorted(by_method.items()):
        ok = [r for r in rows if r.get("status") == "ok"]
        if not ok:
            print(f"  {method:<10} {'ERROR':>14}")
            continue

        avg_snr   = sum(r["snr_db"]        for r in ok) / len(ok)
        avg_bpw   = sum(r["bpw"]           for r in ok) / len(ok)
        avg_cmp   = sum(r["compress_ms"]   for r in ok) / len(ok)
        avg_dcmp  = sum(r["decompress_ms"] for r in ok) / len(ok)

        color = G if method == "aqlm" else Y
        print(
            f"  {color}{method:<10}{NC} "
            f"{avg_snr:>14.2f} "
            f"{avg_bpw:>10.3f} "
            f"{avg_cmp:>20.3f} "
            f"{avg_dcmp:>22.3f}"
        )

    if ppl is not None:
        print(f"\n  {C}Perplexity (wikitext-2, AQLM-compressed model):{NC} {ppl:.3f}")

    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AQLM 2-bit quantisation vs INT4 baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Path to an mlx_lm model directory for perplexity evaluation "
             "(Stage 2).  Omit to run weight-reconstruction benchmarks only.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="FILE",
        help="Path for the JSON results file.",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Number of wikitext-2 tokens to use for perplexity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and report plan without running benchmarks.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{C}bench_aqlm.py{NC}  —  AQLM 2-bit vs INT4 baseline")
    print(f"  platform  : {platform.platform()}")
    print(f"  python    : {sys.version.split()[0]}")
    print(f"  output    : {output_path}")
    print(f"  matrices  : {N_MATRICES} × ({BENCH_ROWS}, {BENCH_COLS})")

    if args.dry_run:
        print(f"\n{Y}  [dry-run] no benchmarks will be run.{NC}")
        return

    # Stage 1 — weight reconstruction
    _hdr("Stage 1 — synthetic weight-reconstruction benchmark")
    print(f"  Running {N_MATRICES} matrices of shape ({BENCH_ROWS}, {BENCH_COLS}) …")

    synthetic_results = run_synthetic_bench()

    # Stage 2 — perplexity (optional)
    ppl: Optional[float] = None
    if args.model_dir:
        _hdr("Stage 2 — perplexity evaluation")
        print(f"  Model directory : {args.model_dir}")
        print(f"  Tokens          : {args.n_tokens}")
        ppl = _run_perplexity(args.model_dir, args.n_tokens)
        if ppl is None:
            _skip("perplexity", "mlx/mlx_lm unavailable or model load failed")
    else:
        _skip("perplexity", "no --model-dir provided")

    # Print summary
    _print_summary(synthetic_results, ppl)

    # Collect and save results
    output: dict[str, Any] = {
        "meta": {
            "platform": platform.platform(),
            "python": sys.version,
            "bench_rows": BENCH_ROWS,
            "bench_cols": BENCH_COLS,
            "n_matrices": N_MATRICES,
        },
        "synthetic": synthetic_results,
    }
    if ppl is not None:
        output["perplexity"] = ppl

    output_path.write_text(json.dumps(output, indent=2))
    print(f"{G}  Results saved to {output_path}{NC}\n")


if __name__ == "__main__":
    main()
