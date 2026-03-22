"""squish/attention/relay_attn.py

RelayAttention — Skip Redundant Attention via a Relay Bank.

Reference
---------
Chen et al. "Relay Attention: Reducing the Computation of Consecutive
Identical Attentions for Long Context LLM Inference."
EMNLP 2024 (arXiv:2402.08268).

Algorithm
---------
In deep transformers, consecutive layers often produce nearly identical
softmax attention outputs (high cosine similarity).  RelayAttention
exploits this by caching the attention output from layer l in a *relay
bank* and re-using it at layer l+1 when the cosine similarity between
the two layers' query vectors exceeds a per-head threshold.

Specifically, for layer l+1:
  - For each head h, compute ``sim = cos(q_h^{l+1}, q_h^l)``.
  - If ``sim >= threshold_h``, skip the full attention computation and
    return the relayed output from layer l.
  - Otherwise, compute attention normally and update the relay bank.

Key properties
--------------
* NumPy-only.
* ``n_heads`` — number of attention heads.
* ``head_dim`` — dimension per head.
* ``threshold`` — per-head cosine similarity threshold for relay reuse.
* ``skip_fraction_target`` — soft cap: if too many heads are being relayed,
  threshold is tightened automatically during calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "RelayAttnConfig",
    "RelayAttention",
]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RelayAttnConfig:
    """Configuration for :class:`RelayAttention`.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        threshold: Cosine similarity threshold for relay reuse (per head).
        skip_fraction_target: Maximum fraction of heads allowed to relay
            simultaneously (used in adaptive threshold logic).
    """

    n_heads: int = 32
    head_dim: int = 128
    bypass_threshold: float = 0.95
    skip_fraction_target: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.bypass_threshold <= 1.0:
            raise ValueError("bypass_threshold must be in (0, 1]")


# ── Module ────────────────────────────────────────────────────────────────────


class RelayAttention:
    """Relay attention module that caches and reuses attention outputs.

    Parameters
    ----------
    config:
        Relay attention configuration.
    """

    def __init__(self, config: Optional[RelayAttnConfig] = None) -> None:
        self._cfg = config or RelayAttnConfig()
        # Per-head thresholds (may be adapted during calibration)
        self._thresholds: np.ndarray = np.full(
            self._cfg.n_heads, self._cfg.bypass_threshold, dtype=np.float32
        )
        # Relay bank: one cached attention output per head
        # Shape: (n_heads, head_dim)
        self._relay_bank: Optional[np.ndarray] = None
        # Previous query for similarity computation
        self._prev_query: Optional[np.ndarray] = None
        # Statistics
        self._total_steps: int = 0
        self._relayed_steps: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def config(self) -> RelayAttnConfig:
        return self._cfg

    @property
    def relay_hit_rate(self) -> float:
        if self._total_steps == 0:
            return 0.0
        return self._relayed_steps / self._total_steps

    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds.copy()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-head cosine similarity.  Both inputs: (n_heads, head_dim)."""
        norm_a = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
        return ((a / norm_a) * (b / norm_b)).sum(axis=1)  # (n_heads,)

    def attend(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention, using relay cache when similarity is high.

        Parameters
        ----------
        query:
            Shape ``(n_heads, q_len, head_dim)`` or ``(n_heads, head_dim)``.
        keys:
            Shape ``(n_heads, seq_len, head_dim)``.
        values:
            Shape ``(n_heads, seq_len, head_dim)``.

        Returns
        -------
        Tuple of (output, skip_mask):
            * output: same shape as ``query``
            * skip_mask: boolean ``(n_heads,)`` — True where relay was used.
        """
        q = np.asarray(query, dtype=np.float32)
        K = np.asarray(keys, dtype=np.float32)
        V = np.asarray(values, dtype=np.float32)

        q_3d = q.ndim == 3
        if q.ndim == 2:
            q = q[:, None, :]  # (n_heads, 1, head_dim)

        scale = float(self._cfg.head_dim ** -0.5)
        n_heads = self._cfg.n_heads
        q_len = q.shape[1]

        # Use compressed q for similarity: take mean over q_len
        q_rep = q.mean(axis=1)  # (n_heads, head_dim)

        # Determine which heads can use relay
        skip_mask = np.zeros(n_heads, dtype=bool)
        if self._relay_bank is not None and self._prev_query is not None:
            sim = self._cosine_sim(q_rep, self._prev_query)
            skip_mask = sim >= self._thresholds

        # Full attention for non-relayed heads
        output = np.zeros((n_heads, q_len, self._cfg.head_dim), dtype=np.float32)
        active_heads = np.where(~skip_mask)[0]

        for h in active_heads:
            # q[h]: (q_len, head_dim), K[h]: (seq_len, head_dim)
            scores = (q[h] @ K[h].T) * scale  # (q_len, seq_len)
            scores -= scores.max(axis=-1, keepdims=True)
            w = np.exp(scores)
            w /= w.sum(axis=-1, keepdims=True)
            output[h] = w @ V[h]  # (q_len, head_dim)

        # Fill relayed heads from relay bank
        relayed_heads = np.where(skip_mask)[0]
        if self._relay_bank is not None and len(relayed_heads) > 0:
            for h in relayed_heads:
                output[h] = self._relay_bank[h]

        # Update relay bank
        self._relay_bank = output.copy()
        self._prev_query = q_rep.copy()

        self._total_steps += n_heads
        self._relayed_steps += int(skip_mask.sum())

        if not q_3d:
            output = output[:, 0, :]  # back to (n_heads, head_dim)
        return output.astype(np.float32), skip_mask

    def adapt_thresholds(self) -> None:
        """Tighten thresholds if relay rate exceeds the target fraction."""
        if self._total_steps == 0:
            return
        current_rate = self._relayed_steps / self._total_steps
        if current_rate > self._cfg.skip_fraction_target:
            # Tighten by a small amount
            self._thresholds = np.minimum(1.0, self._thresholds + 0.01)
        elif current_rate < self._cfg.skip_fraction_target * 0.5:
            # Loosen slightly
            self._thresholds = np.maximum(0.0, self._thresholds - 0.005)
        self._thresholds = self._thresholds.clip(0.0, 1.0)

    def reset(self) -> None:
        """Clear relay bank (call at sequence boundaries)."""
        self._relay_bank = None
        self._prev_query = None

    def reset_stats(self) -> None:
        self._total_steps = 0
        self._relayed_steps = 0
