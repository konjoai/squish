"""squish/kernels/mojo/green_kv_score_mojo.py — Mojo-backed GreenKV scoring.

Wraps the ``green_kv_score`` Mojo kernel via MojoBridge with a NumPy
fallback.  Computes per-token KV-cache importance scores as mean softmax
attention weights over a sliding observation window of query vectors,
with SIMD-vectorised dot-product accumulation per head.

Reference: Zhang et al., "GreenKV: Achieving KV Cache Compression with
Nearly-Lossless Accuracy for Large Language Models," 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "GreenKVMojoConfig",
    "MojoGreenKVScore",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("green_kv_score")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_score(q_obs: np.ndarray, k: np.ndarray) -> np.ndarray:
    n_heads, obs_window, head_dim = q_obs.shape
    seq_len = k.shape[1]
    scale = head_dim ** -0.5
    scores = np.zeros((n_heads, seq_len), dtype=np.float32)
    for h in range(n_heads):
        for qp in range(obs_window):
            logits = q_obs[h, qp] @ k[h].T * scale
            logits -= logits.max()
            w = np.exp(logits)
            w /= w.sum() + 1e-9
            scores[h] += w
        scores[h] /= obs_window
    return scores


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class GreenKVMojoConfig:
    """Configuration for :class:`MojoGreenKVScore`.

    Attributes:
        obs_window: Default observation window size (overrideable per call).
    """

    obs_window: int = 32


class MojoGreenKVScore:
    """Mojo-backed GreenKV per-head KV-cache importance scorer.

    SIMD-vectorised dot-product + softmax per head, ``parallelize`` over
    heads.  Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[GreenKVMojoConfig] = None) -> None:
        self._cfg = config or GreenKVMojoConfig()

    def score(
        self,
        q_obs: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Compute per-token importance scores.

        Args:
            q_obs: Recent query slice ``(n_heads, obs_window, head_dim)``
                   float32.
            k:     Full key cache ``(n_heads, seq_len, head_dim)`` float32.

        Returns:
            ``(n_heads, seq_len)`` float32 importance scores.
        """
        q = np.ascontiguousarray(q_obs, dtype=np.float32)
        kk = np.ascontiguousarray(k, dtype=np.float32)
        if q.ndim != 3 or kk.ndim != 3:
            raise ValueError("q_obs and k must be 3-D (H, T, D)")
        if q.shape[0] != kk.shape[0]:
            raise ValueError("n_heads mismatch")
        if q.shape[2] != kk.shape[2]:
            raise ValueError("head_dim mismatch")
        if _kernel is not None:
            n_heads = q.shape[0]
            seq_len = kk.shape[1]
            out = np.zeros((n_heads, seq_len), dtype=np.float32)
            _kernel(
                q.ctypes.data, kk.ctypes.data, out.ctypes.data,
                n_heads, q.shape[1], seq_len, q.shape[2],
            )
            return out
        return _numpy_score(q, kk)

    def top_k_mask(
        self,
        scores: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        """Return a boolean mask keeping the top-``budget`` tokens per head.

        Args:
            scores: ``(n_heads, seq_len)`` float32 importance scores.
            budget: Number of tokens to retain per head.

        Returns:
            ``(n_heads, seq_len)`` uint8 mask — 1 = keep.
        """
        s = np.asarray(scores, dtype=np.float32)
        n_heads, seq_len = s.shape
        budget = min(budget, seq_len)
        mask = np.zeros((n_heads, seq_len), dtype=np.uint8)
        for h in range(n_heads):
            top_idx = np.argpartition(-s[h], budget - 1)[:budget]
            mask[h, top_idx] = 1
        return mask

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
