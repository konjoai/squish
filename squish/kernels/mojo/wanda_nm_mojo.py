"""squish/kernels/mojo/wanda_nm_mojo.py — Mojo-backed Wanda N:M sparsity kernel.

Wraps ``wanda_nm`` Mojo kernel via MojoBridge with a NumPy fallback.
Computes Wanda element-wise importance (|W| × rms) and applies N:M
structured sparsity masking with SIMD-vectorised column-block sorting.

Reference: Sun et al., "A Simple and Effective Pruning Approach for
Large Language Models." ICLR 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "WandaNMMojoConfig",
    "MojoWandaNM",
]

_bridge = MojoBridge()
_importance_kernel = _bridge.load_kernel("wanda_nm_importance")
_mask_kernel = _bridge.load_kernel("wanda_nm_mask")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_importance(weight: np.ndarray, rms: np.ndarray) -> np.ndarray:
    return np.abs(weight) * rms[np.newaxis, :]


def _numpy_nm_mask(importance: np.ndarray, n: int, m: int) -> np.ndarray:
    rows, cols = importance.shape
    mask = np.zeros_like(importance, dtype=np.uint8)
    n_blocks = (cols + m - 1) // m
    for bi in range(n_blocks):
        cs = bi * m
        ce = min(cs + m, cols)
        block = importance[:, cs:ce]
        top_idx = np.argsort(-block, axis=1)[:, :n]
        for row in range(rows):
            for idx in top_idx[row]:
                mask[row, cs + idx] = 1
    return mask


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class WandaNMMojoConfig:
    """Configuration for :class:`MojoWandaNM`.

    Attributes:
        n:          Non-zero entries kept per M-element block.
        m:          Block size for N:M sparsity.
    """

    n: int = 2
    m: int = 4


class MojoWandaNM:
    """Mojo-backed Wanda N:M importance scoring and mask generation.

    Uses ``parallelize`` over rows (importance) and column blocks (mask),
    with ``vectorize`` for the per-block sort.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[WandaNMMojoConfig] = None) -> None:
        self._cfg = config or WandaNMMojoConfig()

    def importance(
        self,
        weight: np.ndarray,
        activation_rms: np.ndarray,
    ) -> np.ndarray:
        """Compute |W| × rms element-wise importance.

        Args:
            weight:          ``(out_f, in_f)`` float32.
            activation_rms:  ``(in_f,)`` float32.

        Returns:
            ``(out_f, in_f)`` float32 importance matrix.
        """
        w = np.ascontiguousarray(weight, dtype=np.float32)
        rms = np.ascontiguousarray(activation_rms, dtype=np.float32).ravel()
        if rms.shape[0] != w.shape[1]:
            raise ValueError(
                f"rms length {rms.shape[0]} != weight in_features {w.shape[1]}"
            )
        if _importance_kernel is not None:
            out = np.empty_like(w)
            _importance_kernel(
                w.ctypes.data, rms.ctypes.data, out.ctypes.data,
                w.shape[0], w.shape[1],
            )
            return out
        return _numpy_importance(w, rms)

    def nm_mask(
        self,
        importance: np.ndarray,
        n: Optional[int] = None,
        m: Optional[int] = None,
    ) -> np.ndarray:
        """Generate N:M structured sparsity mask.

        Args:
            importance: ``(out_f, in_f)`` float32 importance matrix.
            n:          Non-zero per block (overrides config).
            m:          Block size (overrides config).

        Returns:
            ``(out_f, in_f)`` uint8 mask — 1 = keep.
        """
        n_ = int(n) if n is not None else self._cfg.n
        m_ = int(m) if m is not None else self._cfg.m
        imp = np.ascontiguousarray(importance, dtype=np.float32)
        if _mask_kernel is not None:
            out = np.zeros(imp.shape, dtype=np.uint8)
            _mask_kernel(
                imp.ctypes.data, out.ctypes.data,
                imp.shape[0], imp.shape[1], n_, m_,
            )
            return out
        return _numpy_nm_mask(imp, n_, m_)

    def prune(
        self,
        weight: np.ndarray,
        activation_rms: np.ndarray,
        n: Optional[int] = None,
        m: Optional[int] = None,
    ) -> np.ndarray:
        """Compute importance, generate mask, and zero pruned weights.

        Args:
            weight:          ``(out_f, in_f)`` float32.
            activation_rms:  ``(in_f,)`` float32.
            n:               Non-zero per block (overrides config).
            m:               Block size (overrides config).

        Returns:
            ``(out_f, in_f)`` float32 pruned weight tensor.
        """
        imp = self.importance(weight, activation_rms)
        mask = self.nm_mask(imp, n=n, m=m)
        return (np.ascontiguousarray(weight, dtype=np.float32) * mask).astype(np.float32)

    def backend(self) -> str:
        if _importance_kernel is not None or _mask_kernel is not None:
            return "mojo"
        return "numpy"
