"""squish/kernels/flash_attn3.py

FlashAttn3Kernel — Pingpong warp-scheduled tiled attention (NumPy reference).

Implements the tile-level pingpong overlap pattern from FlashAttention-3:
two producer warps alternate filling shared memory while a consumer warp
computes GEMM tiles, hiding memory-latency behind compute.  This NumPy
reference captures the algorithmic structure (tiled online-softmax with
accumulator rescaling) rather than hardware specifics.

Reference
---------
Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony
and Low-precision." arXiv:2407.08608, 2024.
"""

from __future__ import annotations

__all__ = ["FlashAttn3Config", "FlashAttn3Kernel"]

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FlashAttn3Config:
    """Configuration for FlashAttn3Kernel.

    Parameters
    ----------
    block_size:
        Tile size for keys/values (B_c in the FlashAttention notation).
    pingpong_stages:
        Number of simulated pipeline stages (≥ 1).
    causal:
        Whether to apply a causal (lower-triangular) attention mask.
    scale:
        Softmax scale factor.  Defaults to ``head_dim ** -0.5`` at runtime.
    seed:
        RNG seed (unused in forward; kept for API consistency).
    """

    block_size: int = 64
    pingpong_stages: int = 2
    causal: bool = True
    scale: Optional[float] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.pingpong_stages < 1:
            raise ValueError("pingpong_stages must be >= 1")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class FlashAttn3Kernel:
    """Tiled online-softmax attention with pingpong accumulation.

    Processes Q in blocks of ``block_size`` rows, streaming K/V tiles
    through two alternating accumulation buffers to simulate the pingpong
    warp-scheduling pattern.

    Parameters
    ----------
    config:
        ``FlashAttn3Config`` instance.
    """

    def __init__(self, config: FlashAttn3Config) -> None:
        self.config = config

    def forward(
        self,
        Q: ndarray,
        K: ndarray,
        V: ndarray,
        mask: Optional[ndarray] = None,
    ) -> Tuple[ndarray, ndarray]:
        """Compute scaled dot-product attention with tiled online softmax.

        Parameters
        ----------
        Q:
            Query matrix, shape ``(T_q, head_dim)``.
        K:
            Key matrix, shape ``(T_k, head_dim)``.
        V:
            Value matrix, shape ``(T_k, head_dim)``.
        mask:
            Optional additive attention mask, shape broadcastable to
            ``(T_q, T_k)``; large negative values indicate positions to
            mask out.

        Returns
        -------
        out:
            Attention output, shape ``(T_q, head_dim)``.
        lse:
            Log-sum-exp per query, shape ``(T_q,)``; useful for numeric
            re-composition across heads.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)

        if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
            raise ValueError("Q, K, V must be 2-D matrices")
        T_q, head_dim = Q.shape
        T_k = K.shape[0]
        if K.shape[1] != head_dim or V.shape[0] != T_k or V.shape[1] != head_dim:
            raise ValueError(
                f"Dimension mismatch: Q({T_q},{head_dim}), "
                f"K({T_k},{K.shape[1]}), V({V.shape[0]},{V.shape[1]})"
            )

        scale = self.config.scale if self.config.scale is not None else head_dim ** -0.5

        out = np.zeros((T_q, head_dim), dtype=np.float32)
        # Running log-sum-exp and max per query
        m_prev = np.full(T_q, -np.inf, dtype=np.float32)
        l_prev = np.zeros(T_q, dtype=np.float32)

        Bc = self.config.block_size

        # Iterate over K/V tiles — pingpong: two buffers alternated
        tiles = list(range(0, T_k, Bc))
        buffers: list = [None, None]  # simulate stage-0 / stage-1

        for tile_idx, j_start in enumerate(tiles):
            j_end = min(j_start + Bc, T_k)
            stage = tile_idx % self.config.pingpong_stages

            K_tile = K[j_start:j_end]  # (Bc, d)
            V_tile = V[j_start:j_end]  # (Bc, d)

            # Scores: (T_q, Bc)
            S = scale * (Q @ K_tile.T)

            # Causal mask
            if self.config.causal:
                q_indices = np.arange(T_q)[:, np.newaxis]
                k_indices = np.arange(j_start, j_end)[np.newaxis, :]
                causal_mask = q_indices < k_indices
                S = np.where(causal_mask, -1e9, S)

            # Additive mask
            if mask is not None:
                S = S + np.asarray(mask, dtype=np.float32)[..., j_start:j_end]

            # Online softmax update
            m_new = np.maximum(m_prev, S.max(axis=-1))  # (T_q,)
            exp_S = np.exp(S - m_new[:, np.newaxis])    # (T_q, Bc)
            l_new = np.exp(m_prev - m_new) * l_prev + exp_S.sum(axis=-1)

            # Rescale accumulated output and add new contribution
            rescale = np.exp(m_prev - m_new)             # (T_q,)
            out = rescale[:, np.newaxis] * out + exp_S @ V_tile

            m_prev = m_new
            l_prev = l_new

            # Store tile in pingpong buffer (simulates pipeline staging)
            buffers[stage] = (K_tile, V_tile)

        # Normalise
        safe_l = np.where(l_prev > 0, l_prev, 1.0)
        out = out / safe_l[:, np.newaxis]

        # log-sum-exp: log(l) + m
        lse = np.log(safe_l) + m_prev

        return out, lse

    # ------------------------------------------------------------------
    # Convenience method matching flash_attention.py API
    # ------------------------------------------------------------------

    def __call__(
        self,
        Q: ndarray,
        K: ndarray,
        V: ndarray,
        mask: Optional[ndarray] = None,
    ) -> Tuple[ndarray, ndarray]:
        return self.forward(Q, K, V, mask=mask)
