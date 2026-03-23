"""squish/kernels/mojo/magic_pig_mojo.py — Mojo-backed MagicPIG attention scoring.

Wraps the ``magic_pig_score_kernel`` Mojo stub via MojoBridge with a NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "MagicPIGMojoConfig",
    "MojoMagicPIG",
]

_bridge = MojoBridge()
_score_kernel = _bridge.load_kernel("magic_pig_score_kernel")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_score(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, tq, d = q.shape
    scale = d ** -0.5
    out = np.zeros_like(q)
    for hi in range(h):
        logits = (q[hi] @ k[hi].T) * scale
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True).clip(1e-8, None)
        out[hi] = weights @ v[hi]
    return out.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class MagicPIGMojoConfig:
    """Configuration for :class:`MojoMagicPIG`.

    (No tunable parameters beyond what is passed per-call.)
    """


class MojoMagicPIG:
    """Mojo-backed MagicPIG attention GEMV.

    Falls back to NumPy when the Mojo runtime is absent.

    Example::

        pig = MojoMagicPIG()
        output = pig.score(Q, K, V)
        weights = pig.attention_weights(Q, K)
    """

    def __init__(self, config: Optional[MagicPIGMojoConfig] = None) -> None:
        self._cfg = config or MagicPIGMojoConfig()

    def score(
        self,
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
        h, tq, d = qa.shape
        if ka.shape[0] != h or va.shape[0] != h:
            raise ValueError(
                f"Head dim mismatch: q has {h} heads but k has {ka.shape[0]}, v has {va.shape[0]}"
            )
        if ka.shape[1] != va.shape[1]:
            raise ValueError(
                f"K and V sequence lengths must match; got {ka.shape[1]} vs {va.shape[1]}"
            )
        seq_len = ka.shape[1]
        if _score_kernel is not None:
            out = np.zeros((h, tq, d), dtype=np.float32)
            _score_kernel(
                qa.ctypes.data, ka.ctypes.data, va.ctypes.data, out.ctypes.data,
                h, tq, seq_len, d,
            )
            return out
        return _numpy_score(qa, ka, va)

    def attention_weights(self, q: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Compute softmax attention weights without applying values.

        Args:
            q: ``(H, Tq, d)`` float32 queries.
            k: ``(H, S, d)`` float32 keys.

        Returns:
            ``(H, Tq, S)`` float32 softmax weights.
        """
        qa = np.asarray(q, dtype=np.float32)
        ka = np.asarray(k, dtype=np.float32)
        scale = qa.shape[-1] ** -0.5
        logits = np.einsum("htd,hsd->hts", qa, ka) * scale
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True).clip(1e-8, None)
        return weights.astype(np.float32)

    def backend(self) -> str:
        return "mojo" if _score_kernel is not None else "numpy"
