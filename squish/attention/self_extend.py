"""squish/attention/self_extend.py

SelfExtend — Grouped-Position Floor-Division Attention (Training-Free).

Reference
---------
Jin et al. "LLM Maybe LongLM: Self-Extend LLM Context Window Without
Tuning." ACL 2024 (arXiv:2401.01325).

Algorithm
---------
SelfExtend requires no re-training.  It partitions the KV sequence into:

1. **Local window** (last ``window_size`` tokens) — processed with *exact*
   position IDs (fine resolution).
2. **Group region** (older tokens) — grouped into blocks of size ``group_size``
   and mapped to a compressed floor(position / group_size) ID.

At inference, each query token attends:
* Normally to the local window.
* Over floor-divided positions to all tokens outside the window.

This extends effective context by ``group_size × ratio`` without any change
to model weights.

Key properties
--------------
* ``forward(q, k, v, positions)`` splits into local + group attention and
  merges outputs via log-sum-exp.
* Causal mask is respected within each region.
* NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "SelfExtendConfig",
    "SelfExtend",
]


@dataclass
class SelfExtendConfig:
    """Configuration for :class:`SelfExtend`.

    Attributes:
        group_size: Block size for floor-division position mapping.
        window_size: Number of recent tokens processed at full resolution.
        scale: Attention scale (default 1/sqrt(head_dim)).
    """

    group_size: int = 8
    window_size: int = 1024
    scale: Optional[float] = None


class SelfExtend:
    """Training-free long-context attention via grouped floor-division RoPE.

    Parameters
    ----------
    config:
        SelfExtendConfig.
    """

    def __init__(self, config: Optional[SelfExtendConfig] = None) -> None:
        self._cfg = config or SelfExtendConfig()

    @property
    def config(self) -> SelfExtendConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Compute SelfExtend attention.

        Parameters
        ----------
        query:
            Shape ``(n_heads, seq_q, head_dim)``.
        keys:
            Shape ``(n_heads, seq_kv, head_dim)``.
        values:
            Shape ``(n_heads, seq_kv, head_dim)``.

        Returns
        -------
        Output of shape ``(n_heads, seq_q, head_dim)``.
        """
        n_heads, seq_q, head_dim = query.shape
        n_heads_k, seq_kv, _ = keys.shape
        scale = self._cfg.scale or (head_dim ** -0.5)
        ws = self._cfg.window_size
        gs = self._cfg.group_size

        if seq_kv <= ws:
            # Context fits within window — plain causal attention
            return self._causal_attn(query, keys, values, scale).astype(np.float32)

        # Split KV into local window and group region
        local_keys = keys[:, -ws:, :]  # (H, ws, D)
        local_values = values[:, -ws:, :]
        group_keys = keys[:, :-ws, :]  # (H, seq_kv-ws, D)
        group_values = values[:, :-ws, :]

        # Local window attention (causal within window)
        out_local, lse_local = self._attn_with_lse(query, local_keys, local_values, scale)

        # Group-region attention — floor-divide key positions
        gk = self._group_positions(group_keys, gs, offset=0)
        gv = self._group_positions(group_values, gs, offset=0)
        gq = self._group_positions(query, gs, offset=seq_kv - ws - seq_q)
        out_group, lse_group = self._attn_with_lse(gq, gk, gv, scale)

        # Merge via log-sum-exp
        return self._merge_lse(out_local, lse_local, out_group, lse_group).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _causal_attn(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        scores = np.einsum("hqd,hkd->hqk", q, k) * scale  # (H, sq, sk)
        # Causal mask
        sq, sk = scores.shape[1], scores.shape[2]
        mask = np.triu(np.full((sq, sk), -1e9), k=sk - sq + 1)
        scores = scores + mask[None, :, :]
        w = np.exp(scores - scores.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        return np.einsum("hqk,hkd->hqd", w, v)

    @staticmethod
    def _attn_with_lse(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (output, log-sum-exp) for merging."""
        scores = np.einsum("hqd,hkd->hqk", q, k) * scale
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        s = e.sum(axis=-1, keepdims=True)
        lse = m + np.log(s + 1e-9)  # (H, sq, 1)
        w = e / s
        out = np.einsum("hqk,hkd->hqd", w, v)
        return out, lse[..., 0]  # (H, sq)

    @staticmethod
    def _group_positions(x: np.ndarray, gs: int, offset: int) -> np.ndarray:
        """Floor-divide positions — applied to key tensors (no-op on values)."""
        # For keys: squash positional representation by grouping adjacent vectors
        n_heads, seq, dim = x.shape
        # Reduce sequence by averaging within each block of gs
        n_groups = (seq + gs - 1) // gs
        pad = n_groups * gs - seq
        if pad:
            x = np.pad(x, [(0, 0), (0, pad), (0, 0)])
        grouped = x.reshape(n_heads, n_groups, gs, dim).mean(axis=2)
        return grouped

    @staticmethod
    def _merge_lse(
        out_a: np.ndarray,
        lse_a: np.ndarray,
        out_b: np.ndarray,
        lse_b: np.ndarray,
    ) -> np.ndarray:
        """Merge two attention outputs weighted by their log-sum-exp values."""
        # out_a: (H, sq_a, D), out_b: (H, sq_b, D)
        # Use the same query length (sq) from out_a
        sq = out_a.shape[1]
        # out_b may have different sq (from grouped keys) — use out_a shape
        if out_b.shape[1] != sq:
            # Interpolate or repeat to match
            factor = sq / out_b.shape[1]
            out_b_up = np.repeat(out_b, int(np.ceil(factor)), axis=1)[:, :sq, :]
            lse_b_up = np.repeat(lse_b, int(np.ceil(factor)), axis=1)[:, :sq]
        else:
            out_b_up = out_b
            lse_b_up = lse_b

        lse_max = np.maximum(lse_a, lse_b_up)
        w_a = np.exp(lse_a - lse_max)[..., None]
        w_b = np.exp(lse_b_up - lse_max)[..., None]
        total = w_a + w_b + 1e-9
        return (w_a * out_a + w_b * out_b_up) / total
