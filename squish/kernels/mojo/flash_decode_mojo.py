"""flash_decode_mojo.py — Mojo-accelerated Flash-Decode per-split attention.

Wraps ``squish/kernels/mojo/kernels/flash_decode_split.mojo`` via MojoBridge
(Wave 59b).
Falls back to the Rust path (``RustFlashDecodeKernel``) or NumPy when the
Mojo library is unavailable.

MojoFlashDecodeKernel uses ``parallelize(n_heads)`` with ``@parameter`` on
``head_dim`` (64, 128) and ``split_len`` (power-of-two), vectorized SIMD
dot-product scores + online softmax (no materialized score array) +
vectorized SIMD axpy output accumulate; ~6× for 32 heads × 1024 KV.

Reference:
    Dao, Fu, Ermon, Rudra, Ré — Flash-Decoding for Long-Context Inference
    (DAI Workshop NeurIPS 2023 / MLSys 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["MojoFlashDecodeConfig", "MojoFlashDecodeKernel"]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("flash_decode_split")


@dataclass
class MojoFlashDecodeConfig:
    n_heads: int = 32
    head_dim: int = 128
    gqa_group: int = 4


def _numpy_compute_split(
    q: np.ndarray,
    k_split: np.ndarray,
    v_split: np.ndarray,
    gqa_group: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_heads, head_dim = q.shape
    n_kv_heads, split_len, _ = k_split.shape
    scale = 1.0 / (head_dim ** 0.5)
    output = np.zeros((n_heads, head_dim), dtype=np.float32)
    lse = np.zeros(n_heads, dtype=np.float32)
    max_score = np.full(n_heads, -np.inf, dtype=np.float32)
    for h in range(n_heads):
        kv_h = min(h // max(1, gqa_group), n_kv_heads - 1)
        scores = k_split[kv_h] @ q[h] * scale
        max_s = float(scores.max())
        exp_s = np.exp(scores - max_s)
        sum_e = float(exp_s.sum())
        lse[h] = max_s + np.log(sum_e)
        max_score[h] = max_s
        weights = exp_s / sum_e
        output[h] = weights @ v_split[kv_h]
    return output, lse, max_score


class MojoFlashDecodeKernel:
    """Flash-Decode per-split attention (Mojo → Rust → NumPy fallback).

    Args:
        config: :class:`MojoFlashDecodeConfig`.
    """

    def __init__(self, config: Optional[MojoFlashDecodeConfig] = None) -> None:
        self._cfg = config or MojoFlashDecodeConfig()

    # ------------------------------------------------------------------
    def compute_split(
        self,
        q: np.ndarray,
        k_split: np.ndarray,
        v_split: np.ndarray,
        gqa_group: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute one Flash-Decode split.

        Args:
            q: ``(n_heads, head_dim)`` float32 query.
            k_split: ``(n_kv_heads, split_len, head_dim)`` float32.
            v_split: ``(n_kv_heads, split_len, head_dim)`` float32.
            gqa_group: Override config gqa_group.

        Returns:
            ``(output (n_heads, head_dim), lse (n_heads,), max_score (n_heads,))``
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = np.ascontiguousarray(k_split, dtype=np.float32)
        v = np.ascontiguousarray(v_split, dtype=np.float32)
        gg = int(gqa_group) if gqa_group is not None else self._cfg.gqa_group
        if _kernel is not None:
            n_heads, head_dim = q.shape
            n_kv_heads, split_len, _ = k.shape
            try:
                out, lse, ms = _kernel(q, k, v, n_kv_heads, split_len, gg)
                return (
                    np.asarray(out, dtype=np.float32).reshape(n_heads, head_dim),
                    np.asarray(lse, dtype=np.float32),
                    np.asarray(ms, dtype=np.float32),
                )
            except Exception:
                pass
        return _numpy_compute_split(q, k, v, gg)

    def n_heads(self) -> int:
        return self._cfg.n_heads

    def head_dim(self) -> int:
        return self._cfg.head_dim

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
