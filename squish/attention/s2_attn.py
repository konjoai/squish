"""squish/attention/s2_attn.py

S2Attention — Sorted-Structured Sparse Attention.

Reference
---------
Chen et al. "S²-Attention: Sorted-Structured Sparse Attention for
Long-Context Language Modelling." ICLR 2025 (arXiv:2409.09735).

Algorithm
---------
For long sequences, the full attention matrix is O(n²) and dominates
prefill cost.  S²-Attention reduces this by:

1. **Sort** — rank all key tokens by their dot-product similarity to the
   query vector.  This can be done with a cheap approximate sort using
   query-magnitude gating.
2. **Select** — keep only the top-K tokens per head.
3. **Attend** — compute exact attention over the selected top-K tokens.

The hardware-friendly gather pattern (sorted contiguous access) avoids
sparse scatter-gather overhead.

Key properties
--------------
* NumPy-only.
* ``top_k`` — number of key positions to attend to per head.
* ``exact_threshold`` — if seq_len <= this, run full exact attention.
* ``n_heads`` — number of attention heads.
* ``head_dim`` — dimension per head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "S2AttnConfig",
    "S2Attention",
]


@dataclass
class S2AttnConfig:
    """Configuration for :class:`S2Attention`.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        top_k: Number of key positions to attend to per head.
        exact_threshold: Fall back to exact attention for seq_len <= this.
    """

    n_heads: int = 32
    head_dim: int = 128
    top_k: int = 256
    exact_threshold: int = 512

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


class S2Attention:
    """Sorted-structured sparse attention module.

    Parameters
    ----------
    config:
        S2Attention configuration.
    """

    def __init__(self, config: Optional[S2AttnConfig] = None) -> None:
        self._cfg = config or S2AttnConfig()
        self._sparse_calls: int = 0
        self._exact_calls: int = 0

    @property
    def config(self) -> S2AttnConfig:
        return self._cfg

    @property
    def sparse_call_count(self) -> int:
        return self._sparse_calls

    @property
    def exact_call_count(self) -> int:
        return self._exact_calls

    @property
    def total_calls(self) -> int:
        return self._sparse_calls + self._exact_calls

    def _exact_attention(
        self, query: np.ndarray, keys: np.ndarray, values: np.ndarray, scale: float
    ) -> np.ndarray:
        """Standard scaled dot-product attention.

        Parameters
        ----------
        query: ``(n_heads, head_dim)``
        keys: ``(n_heads, seq_len, head_dim)``
        values: ``(n_heads, seq_len, head_dim)``
        """
        scores = np.einsum("hd,hsd->hs", query, keys) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        return np.einsum("hs,hsd->hd", w, values)

    def forward(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute S²-Attention (sparse or exact based on sequence length).

        Parameters
        ----------
        query:
            Shape ``(n_heads, head_dim)`` — single query position.
        keys:
            Shape ``(n_heads, seq_len, head_dim)``.
        values:
            Shape ``(n_heads, seq_len, head_dim)``.

        Returns
        -------
        Tuple of (output, selected_indices):
            * output: ``(n_heads, head_dim)``
            * selected_indices: ``(n_heads, k_actual)`` — selected positions.
        """
        q_in = np.asarray(query, dtype=np.float32)
        K = np.asarray(keys, dtype=np.float32)
        V = np.asarray(values, dtype=np.float32)
        # Squeeze q_len dim if present (n_heads, 1, head_dim) → (n_heads, head_dim)
        has_q_len = q_in.ndim == 3
        q = q_in.squeeze(1) if has_q_len else q_in  # (n_heads, head_dim)
        seq_len = K.shape[1]
        scale = float(self._cfg.head_dim ** -0.5)

        k_actual = min(self._cfg.top_k, seq_len)

        if seq_len <= self._cfg.exact_threshold or k_actual >= seq_len:
            self._exact_calls += 1
            output = self._exact_attention(q, K, V, scale)
            if has_q_len:
                output = output[:, None, :]
            # Return top-k indices (by approximate score) even in exact path
            q_mean = q.mean(axis=0)
            k_mean = K.mean(axis=0)
            approx_scores = k_mean @ q_mean
            k_sel = min(k_actual, seq_len)
            selected = np.sort(np.argpartition(approx_scores, -k_sel)[-k_sel:]).astype(np.int32)
            return output.astype(np.float32), selected

        self._sparse_calls += 1
        # Use mean query across heads to select a shared set of positions
        q_mean = q.mean(axis=0)  # (head_dim,)
        k_mean = K.mean(axis=0)  # (seq_len, head_dim)
        approx_scores = k_mean @ q_mean  # (seq_len,)

        top_k_indices = np.argpartition(approx_scores, -k_actual)[-k_actual:]
        top_k_indices = np.sort(top_k_indices).astype(np.int32)  # (k_actual,) 1D

        # Gather sparse K/V (shared indices across heads)
        K_sparse = K[:, top_k_indices, :]  # (n_heads, k_actual, head_dim)
        V_sparse = V[:, top_k_indices, :]

        output = self._exact_attention(q, K_sparse, V_sparse, scale)
        if has_q_len:
            output = output[:, None, :]
        return output.astype(np.float32), top_k_indices

    def reset_stats(self) -> None:
        self._sparse_calls = 0
        self._exact_calls = 0
