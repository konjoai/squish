"""Pure helpers for the squish KV-cache HF Space demo (W107).

Every function here is deterministic, returns plain Python / numpy values,
and contains no Gradio imports — so the unit tests in
`tests/test_spaces_demo.py` can pin its behaviour without a Gradio runtime.

The Space itself (`spaces/app.py`) is a thin presentation layer over these
helpers plus the stable KV-cache APIs from :mod:`squish.kv.kv_cache`:
``_quantize_int{8,4,2}_per_channel``, ``_dequantize_int{8,4,2}_per_channel``,
``estimate_kv_memory``, ``recommended_kv_mode_3tier``,
``recommend_mode_for_budget``, and ``HadamardKVCache._build_hadamard``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from squish.kv.kv_cache import (
    HadamardKVCache,
    KV_INT2_AUTO_THRESHOLD,
    KV_INT4_DEFAULT_THRESHOLD,
    _dequantize_int2_per_channel,
    _dequantize_int4_per_channel,
    _dequantize_int8_per_channel,
    _quantize_int2_per_channel,
    _quantize_int4_per_channel,
    _quantize_int8_per_channel,
    estimate_kv_memory,
    recommend_mode_for_budget,
    recommended_kv_mode_3tier,
)

# Tiers exposed in the demo, ordered best → smallest. "fp16" is the reference
# row in the memory-budget table; it is *not* a quantisation tier.
_QUANT_TIERS: tuple = ("int8", "int4", "int2")
_REFERENCE_TIER: str = "fp16"


def snr_db(signal: np.ndarray, recon: np.ndarray) -> float:
    """Reconstruction SNR in dB (10·log₁₀ E[s²] / E[(s−r)²]).

    Returns ``+inf`` for an exact reconstruction (zero error). Uses fp64
    accumulation to dodge fp16 underflow on near-zero residuals.
    """
    s = signal.astype(np.float64)
    r = recon.astype(np.float64)
    err = float(np.mean((s - r) ** 2))
    if err == 0.0:
        return float("inf")
    sig = float(np.mean(s * s))
    return 10.0 * np.log10(sig / err)


def make_synthetic_activations(
    n_tokens: int,
    head_dim: int,
    distribution: str,
    seed: int = 42,
) -> np.ndarray:
    """Build a (n_tokens, head_dim) fp16 synthetic activation tensor.

    Three shapes deliberately exercise the failure modes that motivated
    the W104/W105 codecs:

    - ``"gaussian"`` — IID N(0, 0.3²), the easy case.
    - ``"heavy_tailed"`` — Student-t (df=3) scaled to σ≈0.3; ~0.3% of
      values exceed 6σ. INT2 collapses without rotation; the demo lets the
      user toggle Hadamard preprocessing to watch SNR jump 5-10 dB.
    - ``"outlier"`` — N(0, 0.1²) with 1% of entries replaced by ±5.0
      spikes. Reproduces the "bin-collapse" failure that naive INT2 hits.
    """
    if n_tokens <= 0 or head_dim <= 0:
        raise ValueError("n_tokens and head_dim must be positive")
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        arr = rng.standard_normal((n_tokens, head_dim)) * 0.3
    elif distribution == "heavy_tailed":
        # Student-t df=3, scaled so the bulk has σ ≈ 0.3.
        arr = rng.standard_t(df=3.0, size=(n_tokens, head_dim)) * 0.17
    elif distribution == "outlier":
        arr = rng.standard_normal((n_tokens, head_dim)) * 0.1
        mask = rng.random(arr.shape) < 0.01
        arr[mask] = rng.choice([-5.0, 5.0], size=int(mask.sum()))
    else:
        raise ValueError(
            f"distribution must be one of gaussian/heavy_tailed/outlier, got {distribution!r}"
        )
    return arr.astype(np.float16)


def apply_hadamard(arr_f16: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply a randomised Walsh-Hadamard rotation along ``head_dim``.

    Re-uses ``HadamardKVCache._build_hadamard`` so the rotation matrix is
    bit-identical to the one the production INT2/INT4 path uses. The
    rotation is energy-preserving (orthogonal) — only the per-channel
    distribution changes, which is what the codec actually cares about.
    """
    if arr_f16.ndim != 2:
        raise ValueError(f"apply_hadamard: expected 2-D, got {arr_f16.ndim}-D")
    head_dim = arr_f16.shape[-1]
    rng = np.random.default_rng(seed)
    H = HadamardKVCache._build_hadamard(head_dim, rng).astype(np.float32)
    rotated = arr_f16.astype(np.float32) @ H
    return rotated.astype(np.float16)


@dataclass(frozen=True)
class TierResult:
    """Per-tier outcome of one quantize-dequantize round-trip.

    ``bytes_per_token`` includes the per-token fp32 scale (4 B) plus the
    code bytes (head_dim, head_dim/2, head_dim/4 for int8/int4/int2).
    ``compression_vs_fp16`` is the headline ratio; for head_dim=128 it is
    ~1.94× (int8), ~3.76× (int4), ~7.11× (int2).
    """
    mode: str
    snr_db: float
    bytes_per_token: int
    compression_vs_fp16: float


def run_all_tiers(arr_f16: np.ndarray) -> list[TierResult]:
    """Round-trip ``arr_f16`` through INT8 / INT4 / INT2 and report metrics."""
    if arr_f16.ndim != 2:
        raise ValueError(f"run_all_tiers: expected 2-D, got {arr_f16.ndim}-D")
    head_dim = arr_f16.shape[-1]
    results: list[TierResult] = []

    fp16_bpt = head_dim * 2  # the reference: head_dim fp16 values per token

    q8, s8 = _quantize_int8_per_channel(arr_f16)
    deq8 = _dequantize_int8_per_channel(q8, s8)
    int8_bpt = head_dim + 4
    results.append(TierResult(
        mode="int8",
        snr_db=snr_db(arr_f16, deq8),
        bytes_per_token=int8_bpt,
        compression_vs_fp16=fp16_bpt / int8_bpt,
    ))

    if head_dim % 2 == 0:
        q4, s4 = _quantize_int4_per_channel(arr_f16)
        deq4 = _dequantize_int4_per_channel(q4, s4, head_dim)
        int4_bpt = (head_dim // 2) + 4
        results.append(TierResult(
            mode="int4",
            snr_db=snr_db(arr_f16, deq4),
            bytes_per_token=int4_bpt,
            compression_vs_fp16=fp16_bpt / int4_bpt,
        ))

    if head_dim % 4 == 0:
        q2, s2 = _quantize_int2_per_channel(arr_f16)
        deq2 = _dequantize_int2_per_channel(q2, s2, head_dim)
        int2_bpt = (head_dim // 4) + 4
        results.append(TierResult(
            mode="int2",
            snr_db=snr_db(arr_f16, deq2),
            bytes_per_token=int2_bpt,
            compression_vs_fp16=fp16_bpt / int2_bpt,
        ))

    return results


def recommend_mode_for_context(context_tokens: int) -> str:
    """Wrapper exposing ``recommended_kv_mode_3tier`` to the UI layer.

    Defaults: ≤ 8 K → int8, 8 K-16 K → int4, > 16 K → int2.
    """
    return recommended_kv_mode_3tier(context_tokens)


def memory_table_rows(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    *,
    window: int = 128,
) -> list[dict]:
    """Closed-form per-tier memory rows for a given model + context length.

    Returns a list of dicts (mode, total_mb, recent_window_mb, fits_label,
    compression_ratio) — one for fp16 (the reference) and one for each of
    int8/int4/int2. ``fits_label`` is filled by the caller once it knows
    the user's RAM budget.
    """
    rows: list[dict] = []
    fp16_est = estimate_kv_memory(
        n_layers, n_kv_heads, head_dim, context_tokens, _REFERENCE_TIER, window=window,
    )
    rows.append({
        "mode": "fp16 (reference)",
        "total_mb": fp16_est.total_bytes / 1e6,
        "recent_window_mb": fp16_est.recent_window_bytes / 1e6,
        "compression_ratio": 1.0,
    })
    for tier in _QUANT_TIERS:
        try:
            est = estimate_kv_memory(
                n_layers, n_kv_heads, head_dim, context_tokens, tier, window=window,
            )
        except ValueError:
            continue
        rows.append({
            "mode": tier,
            "total_mb": est.total_bytes / 1e6,
            "recent_window_mb": est.recent_window_bytes / 1e6,
            "compression_ratio": est.compression_ratio,
        })
    return rows


# Pre-loaded "Try these examples" entries. Each tuple is the positional
# argument list for the Tensor Inspector tab. Ordered so the first entry
# is the easy case and the last is the dramatic INT2-bin-collapse demo.
EXAMPLES: tuple = (
    # (n_tokens, head_dim, distribution, rotate)
    (256, 128, "gaussian",     False),
    (256, 128, "heavy_tailed", False),
    (256, 128, "heavy_tailed", True),   # rotation lifts INT2/INT4 SNR
    (256, 128, "outlier",      False),  # naive INT2 collapses
    (256, 128, "outlier",      True),   # rotation rescues it
)


# Memory-budgeter presets — n_layers, n_kv_heads, head_dim sourced from
# the public model configs of the squishai catalogue. Kept small
# (5 entries) so the dropdown stays scannable.
MODEL_PRESETS: dict = {
    "Qwen2.5-0.5B   (24 layers, 2 KV heads, head_dim 64)":  (24, 2,  64),
    "Qwen2.5-1.5B   (28 layers, 2 KV heads, head_dim 128)": (28, 2,  128),
    "Qwen2.5-3B     (36 layers, 2 KV heads, head_dim 128)": (36, 2,  128),
    "Qwen2.5-7B     (28 layers, 4 KV heads, head_dim 128)": (28, 4,  128),
    "Llama-3.1-8B   (32 layers, 8 KV heads, head_dim 128)": (32, 8,  128),
}


def label_budget_fit(rows: Iterable[dict], budget_mb: float) -> list[dict]:
    """Annotate each memory row with a "fits / over by N MB" label.

    Pure-presentation, separated for testability — the Gradio side just
    consumes the resulting list. ``budget_mb ≤ 0`` disables the column
    (every row is labelled ``"-"``).
    """
    labelled: list[dict] = []
    for row in rows:
        new = dict(row)
        if budget_mb <= 0:
            new["fits"] = "-"
        else:
            total_mb = row["total_mb"] + row["recent_window_mb"]
            if total_mb <= budget_mb:
                new["fits"] = f"yes ({budget_mb - total_mb:.0f} MB headroom)"
            else:
                new["fits"] = f"no (over by {total_mb - budget_mb:.0f} MB)"
        labelled.append(new)
    return labelled


def recommend_for_budget_mb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    budget_mb: float,
    *,
    window: int = 128,
) -> str:
    """Convenience wrapper — RAM budget in MB → recommended mode label.

    Returns ``"int8"`` / ``"int4"`` / ``"int2"`` if some tier fits, or
    the literal ``"none — context too long for budget"`` if even INT2 is
    over budget (the caller must shrink context, layers, or heads).
    """
    if budget_mb <= 0:
        raise ValueError(f"budget_mb must be positive, got {budget_mb}")
    budget_bytes = int(budget_mb * 1e6)
    mode = recommend_mode_for_budget(
        n_layers, n_kv_heads, head_dim, context_tokens, budget_bytes, window=window,
    )
    if mode is None:
        return "none — context too long for budget"
    return mode


__all__ = (
    "EXAMPLES",
    "MODEL_PRESETS",
    "TierResult",
    "apply_hadamard",
    "label_budget_fit",
    "make_synthetic_activations",
    "memory_table_rows",
    "recommend_for_budget_mb",
    "recommend_mode_for_context",
    "run_all_tiers",
    "snr_db",
    "KV_INT2_AUTO_THRESHOLD",
    "KV_INT4_DEFAULT_THRESHOLD",
)
