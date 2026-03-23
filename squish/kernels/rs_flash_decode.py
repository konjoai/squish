"""rs_flash_decode.py — Rust-accelerated Flash-Decode per-split attention.

Wraps ``flash_decode_split_f32`` from ``squish_quant_rs`` (Wave 59a).
Falls back to NumPy when the Rust extension is unavailable.

RustFlashDecodeKernel parallelizes over n_heads via Rayon for each
KV-split: per-head GEMV ``scores = K_split @ q[h]`` + online softmax +
``output_h = V_split.T @ weights``; ~5× for 32 heads × 1024 KV tokens.

Hooks into ``flash_decode.py`` ``FlashDecodeAttention._compute_split()``.

Reference:
    Dao, Fu, Ermon, Rudra, Ré — Flash-Decoding for Long-Context Inference
    (DAI Workshop NeurIPS 2023 / MLSys 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "flash_decode_split_f32")
except ImportError:
    _HAS_RUST = False

__all__ = ["FlashDecodeConfig", "RustFlashDecodeKernel"]


@dataclass
class FlashDecodeConfig:
    n_heads: int = 32
    head_dim: int = 128
    gqa_group: int = 4         # n_q_heads / n_kv_heads


def _numpy_compute_split(
    q: np.ndarray,
    k_split: np.ndarray,
    v_split: np.ndarray,
    gqa_group: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy reference flash-decode split: per-head GEMV + online softmax."""
    n_heads, head_dim = q.shape
    n_kv_heads, split_len, _ = k_split.shape
    scale = 1.0 / (head_dim ** 0.5)
    output = np.zeros((n_heads, head_dim), dtype=np.float32)
    lse = np.zeros(n_heads, dtype=np.float32)
    max_score = np.full(n_heads, -np.inf, dtype=np.float32)
    for h in range(n_heads):
        kv_h = min(h // max(1, gqa_group), n_kv_heads - 1)
        scores = k_split[kv_h] @ q[h] * scale          # (split_len,)
        max_s = float(scores.max())
        exp_s = np.exp(scores - max_s)
        sum_e = float(exp_s.sum())
        lse[h] = max_s + np.log(sum_e)
        max_score[h] = max_s
        weights = exp_s / sum_e                         # (split_len,)
        output[h] = weights @ v_split[kv_h]             # (head_dim,)
    return output.ravel(), lse, max_score


class RustFlashDecodeKernel:
    """Flash-Decode per-split attention kernel (Rust-backed or NumPy fallback).

    Args:
        config: :class:`FlashDecodeConfig`.
    """

    def __init__(self, config: Optional[FlashDecodeConfig] = None) -> None:
        self._cfg = config or FlashDecodeConfig()

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
            q: ``(n_heads, head_dim)`` float32 query for this decode step.
            k_split: ``(n_kv_heads, split_len, head_dim)`` float32 key split.
            v_split: ``(n_kv_heads, split_len, head_dim)`` float32 value split.
            gqa_group: Override config gqa_group.

        Returns:
            ``(output (n_heads, head_dim), lse (n_heads,), max_score (n_heads,))``
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = np.ascontiguousarray(k_split, dtype=np.float32)
        v = np.ascontiguousarray(v_split, dtype=np.float32)
        n_heads, head_dim = q.shape
        n_kv_heads, split_len, _ = k.shape
        gg = int(gqa_group) if gqa_group is not None else self._cfg.gqa_group
        if _HAS_RUST:
            # Rust expects k/v as 2D: (n_kv_heads * split_len, head_dim)
            k2d = k.reshape(n_kv_heads * split_len, head_dim)
            v2d = v.reshape(n_kv_heads * split_len, head_dim)
            out_flat, lse, max_s = _sq.flash_decode_split_f32(
                q, k2d, v2d, n_kv_heads, split_len, gg
            )
            return out_flat.reshape(n_heads, head_dim), lse, max_s
        out_flat, lse, max_s = _numpy_compute_split(q, k, v, gg)
        return out_flat.reshape(n_heads, head_dim), lse, max_s

    def n_heads(self) -> int:
        return self._cfg.n_heads

    def head_dim(self) -> int:
        return self._cfg.head_dim

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
