"""
squish/attention/per_layer_sparse_attn.py

PerLayerAdaptiveSparsity — Per-head entropy-based attention sparsity toggle.

Key insight
-----------
``--minference`` applies a global sparse attention pattern to ALL heads on
sequences above a length threshold.  In practice attention heads are
heterogeneous:

  * Some heads exhibit **local patterns** (strong diagonal bands, low entropy
    from position 4 onward) — these converge early and benefit from sparse
    computation.
  * Other heads remain **globally attentive** (high entropy across the full
    sequence) — forcing sparsity on these heads causes visible quality loss.

``PerLayerSparseAttn`` computes a lightweight entropy estimate for each head
during the *prefill* forward pass (using an attention-weight proxy derived
from the query-key dot-products), builds a binary ``use_sparse`` mask per
head per layer, and applies that mask selectively during *decode*.

Only heads flagged ``use_sparse=True`` are routed through ``SparseAttnIndex``
or ``MInference``; dense heads follow the standard code path.

Reference
---------
- Jiang et al., "MInference 1.0: Accelerating Pre-filling for Long-Context
  LLMs via Dynamic Sparse Attention", arXiv 2024.
- Zandieh et al., "SubGen: Token Generation in Sublinear Time and Memory",
  arXiv 2024.

Usage::

    from squish.attention.per_layer_sparse_attn import (
        PerLayerSparseConfig,
        PerLayerSparseAttn,
    )

    cfg    = PerLayerSparseConfig(n_layers=32, n_heads=32, entropy_threshold=0.5)
    sparse = PerLayerSparseAttn(cfg)

    # During prefill: profile each head
    # attn_weights shape: (n_layers, n_heads, seq_len, seq_len) — numpy float32
    sparse.profile_prefill(attn_weights)

    # During decode: get the mask
    mask = sparse.sparse_mask(layer=0)   # shape (n_heads,) bool
    # Apply sparse attention only where mask[h] == True
"""

from __future__ import annotations

__all__ = [
    "PerLayerSparseConfig",
    "HeadProfile",
    "PerLayerSparseAttn",
]

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PerLayerSparseConfig:
    """Configuration for per-head adaptive sparsity.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    n_heads : int
        Number of attention heads per layer (assumed uniform).
    entropy_threshold : float
        Heads with normalised attention-weight entropy *below* this value
        are flagged as local/sparse (``use_sparse=True``).  Range [0, 1].
    min_seq_len : int
        Minimum sequence length before sparse profiling is applied.
        Short sequences are always dense.
    warmup_steps : int
        Number of decode steps to wait before applying the sparse mask
        (allows the KV cache to stabilise first).
    ema_alpha : float
        EMA smoothing for per-head entropy across profiling calls.
        1.0 = always use the latest prefill entropy.
    """

    n_layers:          int   = 32
    n_heads:           int   = 32
    entropy_threshold: float = 0.5
    min_seq_len:       int   = 512
    warmup_steps:      int   = 4
    ema_alpha:         float = 1.0

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1; got {self.n_layers}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if not (0.0 <= self.entropy_threshold <= 1.0):
            raise ValueError(
                f"entropy_threshold must be in [0,1]; got {self.entropy_threshold}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be ≥ 0; got {self.warmup_steps}"
            )
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0,1]; got {self.ema_alpha}"
            )


# ---------------------------------------------------------------------------
# Head profile
# ---------------------------------------------------------------------------

@dataclass
class HeadProfile:
    """Per-head entropy profile for one layer.

    Attributes
    ----------
    layer : int
        Layer index.
    entropies : np.ndarray shape (n_heads,)
        Normalised per-head attention entropy estimate.
    sparse_mask : np.ndarray shape (n_heads,) bool
        True where the head is classified as local/sparse.
    n_sparse : int
        Number of sparse heads.
    """

    layer:       int
    entropies:   np.ndarray
    sparse_mask: np.ndarray

    @property
    def n_sparse(self) -> int:
        return int(self.sparse_mask.sum())

    @property
    def sparse_fraction(self) -> float:
        return self.n_sparse / max(len(self.sparse_mask), 1)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PerLayerSparseAttn:
    """Manages per-head adaptive sparse attention routing.

    Parameters
    ----------
    config : PerLayerSparseConfig
    """

    def __init__(self, config: PerLayerSparseConfig) -> None:
        self._cfg     = config
        self._profiles: List[Optional[HeadProfile]] = [
            None for _ in range(config.n_layers)
        ]
        # EMA entropy accumulators, shape (n_layers, n_heads)
        self._ema: Optional[np.ndarray] = None
        self._profiled = False
        self._decode_step = 0

    # ------------------------------------------------------------------
    # Prefill profiling
    # ------------------------------------------------------------------

    def profile_prefill(
        self,
        attn_weights: np.ndarray,
        seq_len: Optional[int] = None,
    ) -> None:
        """Profile attention heads from prefill attention weights.

        Parameters
        ----------
        attn_weights : np.ndarray
            Shape ``(n_layers, n_heads, seq_q, seq_k)`` — the raw softmax
            attention weight matrix from the prefill forward pass.
            If your model returns something different (e.g. only one layer
            at a time), call :meth:`profile_single_layer` instead.
        seq_len : int | None
            Effective query sequence length.  If ``None``, inferred from
            ``attn_weights.shape[-2]``.
        """
        w = np.asarray(attn_weights, dtype=np.float32)
        if w.ndim != 4:
            raise ValueError(
                f"attn_weights must be 4-D (n_layers, n_heads, seq_q, seq_k); "
                f"got shape {w.shape}"
            )
        n_l, n_h, sq, sk = w.shape
        if seq_len is None:
            seq_len = sq
        if seq_len < self._cfg.min_seq_len:
            self._profiled = False
            return

        entropies = self._compute_entropies(w)  # (n_layers, n_heads)
        self._update_ema(entropies)
        self._build_profiles()
        self._decode_step = 0
        self._profiled    = True

    def profile_single_layer(
        self,
        layer: int,
        attn_weights: np.ndarray,
        seq_len: Optional[int] = None,
    ) -> None:
        """Profile a single layer's attention weights.

        Parameters
        ----------
        layer : int
            Layer index (0-based).
        attn_weights : np.ndarray
            Shape ``(n_heads, seq_q, seq_k)``.
        seq_len : int | None
            If ``None``, inferred from the array shape.
        """
        w = np.asarray(attn_weights, dtype=np.float32)
        if w.ndim != 3:
            raise ValueError(
                f"attn_weights must be 3-D (n_heads, seq_q, seq_k); "
                f"got shape {w.shape}"
            )
        n_h, sq, _ = w.shape
        if seq_len is None:
            seq_len = sq
        if seq_len < self._cfg.min_seq_len:
            return

        # Single-layer slice: shape (1, n_heads, sq, sk)
        layer_w = w[np.newaxis]
        entropies_layer = self._compute_entropies(layer_w)[0]  # (n_heads,)

        if self._ema is None:
            self._ema = np.zeros((self._cfg.n_layers, self._cfg.n_heads), dtype=np.float32)
        α = self._cfg.ema_alpha
        self._ema[layer] = (1 - α) * self._ema[layer] + α * entropies_layer
        self._build_profile_for_layer(layer, self._ema[layer])
        self._profiled = True

    def reset(self) -> None:
        """Clear all profiles (call at the start of a new request)."""
        self._profiles    = [None for _ in range(self._cfg.n_layers)]
        self._profiled    = False
        self._decode_step = 0

    # ------------------------------------------------------------------
    # Decode-time access
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the decode step counter (call once per decode step)."""
        self._decode_step += 1

    def sparse_mask(self, layer: int) -> np.ndarray:
        """Return a boolean mask of shape ``(n_heads,)`` for *layer*.

        ``True`` means the head should use sparse attention.
        Returns all-``False`` when profiling has not run yet or the warmup
        period has not elapsed.
        """
        if (not self._profiled
                or self._decode_step < self._cfg.warmup_steps
                or layer >= len(self._profiles)
                or self._profiles[layer] is None):
            return np.zeros(self._cfg.n_heads, dtype=bool)
        return self._profiles[layer].sparse_mask  # type: ignore[union-attr]

    def is_active(self) -> bool:
        """``True`` if profiles are ready and warmup is complete."""
        return self._profiled and self._decode_step >= self._cfg.warmup_steps

    def head_profiles(self) -> List[Optional[HeadProfile]]:
        """Return per-layer head profiles (``None`` for unprobed layers)."""
        return list(self._profiles)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropies(attn: np.ndarray) -> np.ndarray:
        """Compute per-head normalised entropy from attention weights.

        Parameters
        ----------
        attn : (n_layers, n_heads, sq, sk) float32

        Returns
        -------
        entropies : (n_layers, n_heads) float32 in [0, 1]
        """
        n_l, n_h, sq, sk = attn.shape
        # Average over query positions to get a (n_layers, n_heads, sk) distribution
        avg = attn.mean(axis=2)  # (n_l, n_h, sk)
        avg = np.clip(avg, 1e-9, None)
        avg /= avg.sum(axis=-1, keepdims=True)

        # Shannon entropy
        h = -(avg * np.log(avg)).sum(axis=-1)  # (n_l, n_h)
        h_max = np.log(max(sk, 2))
        return (h / h_max).astype(np.float32)

    def _update_ema(self, entropies: np.ndarray) -> None:
        """Update EMA entropy accumulators."""
        α = self._cfg.ema_alpha
        n_l = min(entropies.shape[0], self._cfg.n_layers)
        n_h = min(entropies.shape[1], self._cfg.n_heads)
        if self._ema is None:
            self._ema = np.zeros(
                (self._cfg.n_layers, self._cfg.n_heads), dtype=np.float32
            )
        self._ema[:n_l, :n_h] = (
            (1 - α) * self._ema[:n_l, :n_h]
            + α * entropies[:n_l, :n_h]
        )

    def _build_profiles(self) -> None:
        """Rebuild per-layer HeadProfile objects from current EMA."""
        if self._ema is None:
            return
        for layer in range(self._cfg.n_layers):
            self._build_profile_for_layer(layer, self._ema[layer])

    def _build_profile_for_layer(
        self, layer: int, entropies: np.ndarray
    ) -> None:
        n_h = min(len(entropies), self._cfg.n_heads)
        ent = entropies[:n_h]
        mask = ent < self._cfg.entropy_threshold
        self._profiles[layer] = HeadProfile(
            layer       = layer,
            entropies   = ent.copy(),
            sparse_mask = mask.copy(),
        )
