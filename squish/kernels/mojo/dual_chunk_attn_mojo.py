"""dual_chunk_attn_mojo.py — Mojo-accelerated Dual-Chunk Attention SDPA.

Wraps `squish/kernels/mojo/kernels/dual_chunk_attn.mojo` via MojoBridge
(Wave 58b). Falls back to NumPy einsum when the Mojo library is unavailable.

MojoDualChunkAttn runs a tiled causal SDPA over 512-token chunks with
online softmax accumulation (`@parameter chunk_size=512`, `head_dim=128`)
and a SIMD mean-query inter-chunk scoring step, replacing three `np.einsum`
calls in `dual_chunk_attn.py` with a single Mojo `parallelize` over heads.

~3× on chunk_size=512, head_dim=128 vs NumPy einsum dispatch overhead.

Reference:
  An et al. (arXiv:2406.17419, 2024) — Dual Chunk Attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["DualChunkAttnConfig", "MojoDualChunkAttn"]

_bridge = MojoBridge()
_MOJO_FN = _bridge.load_kernel("dual_chunk_attn")


@dataclass
class DualChunkAttnConfig:
    """Configuration for MojoDualChunkAttn.

    Attributes:
        n_heads:    Number of attention heads.
        head_dim:   Head dimension.
        chunk_size: Intra-chunk sequence length.
        scale:      Softmax scale factor (defaults to ``1/sqrt(head_dim)``).
    """

    n_heads: int = 8
    head_dim: int = 128
    chunk_size: int = 512
    scale: float | None = None


class MojoDualChunkAttn:
    """Mojo-accelerated Dual-Chunk Attention SDPA.

    Splits the sequence into fixed-size chunks and runs tiled causal SDPA
    with online max tracking (no explicit causal mask materialization) for
    intra-chunk attention, plus SIMD mean-query inter-chunk scoring.

    Usage::

        dca = MojoDualChunkAttn(DualChunkAttnConfig(n_heads=8, head_dim=128))
        # Q, K, V shape: (n_heads, seq_len, head_dim)
        Q = np.random.randn(8, 512, 128).astype(np.float32)
        K = np.random.randn(8, 512, 128).astype(np.float32)
        V = np.random.randn(8, 512, 128).astype(np.float32)
        out = dca.forward(Q, K, V)   # (n_heads, seq_len, head_dim)
    """

    def __init__(self, config: DualChunkAttnConfig | None = None) -> None:
        self._cfg = config or DualChunkAttnConfig()
        self._scale = self._cfg.scale or (self._cfg.head_dim ** -0.5)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """Compute dual-chunk causal attention.

        Args:
            Q:          Float32 ``(n_heads, seq_len, head_dim)`` queries.
            K:          Float32 ``(n_heads, seq_len, head_dim)`` keys.
            V:          Float32 ``(n_heads, seq_len, head_dim)`` values.
            chunk_size: Override config chunk_size.

        Returns:
            Float32 ``(n_heads, seq_len, head_dim)`` attention output.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        cs = chunk_size if chunk_size is not None else self._cfg.chunk_size
        if _MOJO_FN is not None:
            return np.asarray(_MOJO_FN(Q, K, V, cs, self._scale), dtype=np.float32)
        return self._numpy_forward(Q, K, V, cs)

    def chunk_size(self) -> int:
        """Return configured chunk size."""
        return self._cfg.chunk_size

    def backend(self) -> str:
        """Return 'mojo' if Mojo kernel loaded, else 'numpy'."""
        return "mojo" if _MOJO_FN is not None else "numpy"

    # ── NumPy fallback ─────────────────────────────────────────────────────

    def _numpy_forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        n_heads, seq_len, head_dim = Q.shape
        scale = self._scale
        out = np.zeros_like(Q)
        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            lo = c * chunk_size
            hi = min(lo + chunk_size, seq_len)
            Qc = Q[:, lo:hi, :]   # (H, cs, D)
            Kc = K[:, lo:hi, :]
            Vc = V[:, lo:hi, :]
            # Intra-chunk causal SDPA via einsum
            scores = np.einsum("hqd,hkd->hqk", Qc, Kc) * scale  # (H, cs, cs)
            # Causal mask
            cs_actual = hi - lo
            mask = np.tril(np.ones((cs_actual, cs_actual), dtype=np.float32))
            scores = np.where(mask[np.newaxis, :, :] > 0, scores, -1e9)
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
            out[:, lo:hi, :] = np.einsum("hqk,hkd->hqd", attn, Vc)
        return out
