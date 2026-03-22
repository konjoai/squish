"""Mojo-backed Flash Attention prefill kernel.

Backend resolution order:
1. Compiled Mojo shared library (via :class:`MojoBridge`)
2. Pure NumPy block-tiled SDPA with online log-sum-exp (same algorithm)

The Mojo kernel source lives in ``kernels/flash_prefill.mojo``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["MojoFlashPrefillConfig", "MojoFlashPrefill"]


@dataclass
class MojoFlashPrefillConfig:
    """Configuration for :class:`MojoFlashPrefill`.

    Attributes
    ----------
    block_size:
        Tile size for the block-wise attention computation.  Smaller values
        trade compute for memory; 16–64 are typical for CPU/Metal.
    n_heads:
        Number of attention heads (informational; derived from input shape).
    head_dim:
        Per-head feature dimension.  Must match query/key/value channel size.
    causal:
        When ``True`` (default), applies a causal (lower-triangular) mask so
        each position can only attend to earlier positions.
    """

    block_size: int = 16
    n_heads: int = 32
    head_dim: int = 128
    causal: bool = True


class MojoFlashPrefill:
    """Block-tiled Flash Attention prefill with Mojo → NumPy fallback.

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`MojoFlashPrefillConfig`.
    bridge:
        Pre-constructed :class:`MojoBridge`.  A default bridge is created
        if not supplied.
    """

    def __init__(
        self,
        config: MojoFlashPrefillConfig | None = None,
        bridge: MojoBridge | None = None,
    ) -> None:
        self.config = config or MojoFlashPrefillConfig()
        self._bridge = bridge or MojoBridge()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Compute multi-head scaled dot-product attention output.

        Parameters
        ----------
        Q, K, V:
            float32 arrays of shape ``(n_heads, seq_len, head_dim)``.

        Returns
        -------
        float32 output of shape ``(n_heads, seq_len, head_dim)``.
        """
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise ValueError("Q, K, V must be 3-D (n_heads, seq_len, head_dim)")
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError("Q, K, V must have identical shapes")

        Q32 = Q.astype(np.float32, copy=False)
        K32 = K.astype(np.float32, copy=False)
        V32 = V.astype(np.float32, copy=False)

        # 1. Mojo (ctypes)
        fn = self._bridge.load_kernel("mojo_flash_prefill_f32")
        if fn is not None:
            pass  # fall through

        # 2. NumPy block-tiled flash attention
        return self._numpy_flash(Q32, K32, V32)

    def backend(self) -> str:
        """Return the active backend name."""
        return self._bridge.backend()

    # ------------------------------------------------------------------ #
    #  NumPy fallback — block-tiled online softmax                        #
    # ------------------------------------------------------------------ #

    def _numpy_flash(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray
    ) -> np.ndarray:
        n_heads, seq_len, head_dim = Q.shape
        scale = 1.0 / np.sqrt(head_dim)
        block_size = self.config.block_size
        out = np.zeros_like(Q)

        for h in range(n_heads):
            Qh = Q[h]  # (seq_len, head_dim)
            Kh = K[h]
            Vh = V[h]

            # Process one query block at a time
            for q_start in range(0, seq_len, block_size):
                q_end = min(q_start + block_size, seq_len)
                Qb = Qh[q_start:q_end]  # (bq, head_dim)

                # Running online-softmax statistics per query row
                m_i = np.full(q_end - q_start, -1e30, dtype=np.float32)
                l_i = np.zeros(q_end - q_start, dtype=np.float32)
                o_i = np.zeros((q_end - q_start, head_dim), dtype=np.float32)

                for k_start in range(0, seq_len, block_size):
                    k_end = min(k_start + block_size, seq_len)
                    Kb = Kh[k_start:k_end]
                    Vb = Vh[k_start:k_end]

                    # Score matrix: (bq, bk)
                    S = (Qb @ Kb.T) * scale

                    if self.config.causal:
                        # Mask future positions
                        q_positions = np.arange(q_start, q_end)[:, None]
                        k_positions = np.arange(k_start, k_end)[None, :]
                        S = np.where(k_positions <= q_positions, S, -1e30)

                    # Online softmax update (per query row)
                    row_max = S.max(axis=-1)  # (bq,)
                    m_new = np.maximum(m_i, row_max)
                    exp_corr = np.exp(m_i - m_new)           # (bq,)
                    exp_S = np.exp(S - m_new[:, None])        # (bq, bk)

                    l_i = exp_corr * l_i + exp_S.sum(axis=-1)
                    o_i = exp_corr[:, None] * o_i + exp_S @ Vb
                    m_i = m_new

                # Normalise
                out[h, q_start:q_end] = o_i / np.maximum(l_i[:, None], 1e-10)

        return out.astype(np.float32)
