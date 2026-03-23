"""squish/kernels/rs_magic_pig.py — Rust-backed MagicPIG LSH attention scoring.

Wraps ``squish_quant_rs.magic_pig_score_f32`` with a NumPy fallback.

MagicPIG accelerates long-context inference by using Locality-Sensitive
Hashing (LSH) to identify high-attention-score KV pairs ahead of the GEMV.
This wrapper exposes the parallel head-level attention GEMV + softmax kernel
and provides an attention-weight extraction helper for analysis.

Reference: He et al., "MagicPIG: LSH Sampling for Efficient LLM Generation,"
arXiv 2410.16179, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MagicPIGConfig",
    "RustMagicPIG",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "magic_pig_score_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_score(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Multi-head scaled dot-product attention.

    Args:
        q: ``(H, Tq, d)`` float32 queries.
        k: ``(H, S, d)`` float32 keys.
        v: ``(H, S, d)`` float32 values.

    Returns:
        ``(H, Tq, d)`` float32 attention output.
    """
    h, tq, d = q.shape
    scale = d ** -0.5
    out = np.zeros_like(q)
    for hi in range(h):
        # (Tq, S) logits
        logits = (q[hi] @ k[hi].T) * scale  # (Tq, S)
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True).clip(1e-8, None)
        out[hi] = weights @ v[hi]  # (Tq, d)
    return out.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class MagicPIGConfig:
    """Configuration for :class:`RustMagicPIG`.

    (No tunable parameters beyond what is passed per-call for this kernel.)
    """


class RustMagicPIG:
    """Rust-accelerated MagicPIG attention GEMV.

    Computes scaled dot-product attention over all ``H`` heads in parallel
    using Rayon.  Each head processes ``Tq`` query tokens against ``S`` key/
    value pairs with numerically-stable softmax.  Falls back to NumPy when
    ``squish_quant_rs`` is unavailable.

    Example::

        pig = RustMagicPIG()
        output = pig.score(Q, K, V)
        weights = pig.attention_weights(Q, K)
    """

    def __init__(self, config: Optional[MagicPIGConfig] = None) -> None:
        self._cfg = config or MagicPIGConfig()

    def score(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Multi-head scaled dot-product attention.

        Args:
            q: ``(H, Tq, d)`` float32 query tensor.
            k: ``(H, S, d)`` float32 key tensor.
            v: ``(H, S, d)`` float32 value tensor.

        Returns:
            ``(H, Tq, d)`` float32 attention output.

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        qa = np.ascontiguousarray(q, dtype=np.float32)
        ka = np.ascontiguousarray(k, dtype=np.float32)
        va = np.ascontiguousarray(v, dtype=np.float32)
        if qa.ndim != 3 or ka.ndim != 3 or va.ndim != 3:
            raise ValueError(
                f"q, k, v must be 3-D (H, T, d); got {qa.shape}, {ka.shape}, {va.shape}"
            )
        if qa.shape[0] != ka.shape[0] or ka.shape[0] != va.shape[0]:
            raise ValueError("Head dimension H must match across q, k, v")
        if ka.shape[1] != va.shape[1]:
            raise ValueError(f"K and V sequence lengths must match; got {ka.shape[1]}, {va.shape[1]}")
        if _HAS_RUST:
            return np.asarray(_sq.magic_pig_score_f32(qa, ka, va), dtype=np.float32)
        return _numpy_score(qa, ka, va)

    def attention_weights(
        self,
        q: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Compute softmax attention weights without applying them to values.

        Args:
            q: ``(H, Tq, d)`` float32 query tensor.
            k: ``(H, S, d)`` float32 key tensor.

        Returns:
            ``(H, Tq, S)`` float32 softmax attention weights.
        """
        qa = np.asarray(q, dtype=np.float32)
        ka = np.asarray(k, dtype=np.float32)
        scale = qa.shape[-1] ** -0.5
        # (H, Tq, S) = (H, Tq, d) @ (H, d, S)
        logits = np.einsum("htd,hsd->hts", qa, ka) * scale
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True).clip(1e-8, None)
        return weights.astype(np.float32)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
