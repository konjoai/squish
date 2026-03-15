"""
squish/kv/fast_weights.py

Phase 5 — TTT Fast Weights: Online KV Memory Compression

Motivation
──────────
Phase 3 Q-Filters evict geometrically low-relevance tokens from the KV cache,
but that information is permanently discarded.  Fast weights restore the lost
recall by *absorbing* evicted KV pairs into a compact outer-product memory
before they are dropped.

Architecture
────────────
For each transformer layer and each attention head, maintain a fast-weight
matrix  W_f ∈ ℝ^{head_dim × head_dim}  that continuously accumulates
key-value information via the linear-attention update rule:

    W_f ← α · W_f  +  lr · ΣΣ  v_t ⊗ k_t

where:
  α   = decay factor (default 0.999)  — prevents unbounded growth of old info
  lr  = learning rate (default 0.01)  — controls how much each KV pair contributes
  v_t = value vector for token t
  k_t = key   vector for token t

At query time, the compressed history response for query q is:

    out_compressed = W_f @ q   (added to the standard local-attention output)

The fast-weight memory is mathematically equivalent to linear attention computed
over all absorbed tokens, subject to the per-absorption decay.  This gives
approximate but never-zero recall of long context — even after Q-Filters evict
old tokens, their contribution survives in W_f.

Integration
───────────
• Called from Q-Filter eviction (``q_filters.py``): evicted tokens are absorbed
  into the fast-weight layer *before* being removed from the KV cache.
• The ``FastWeightManager`` is held by ``QuantizedKVCache`` alongside the
  ``QFilterManager``; both are activated by ``--fast-weight-lr > 0``.
• Queries against W_f are exposed via ``query_layer(idx, q)`` and can be used
  by the ReDrafter head or any downstream consumer.

Design choices
──────────────
• Pure numpy — no MLX dependency at import time; fully unit-testable.
• Lazy shape initialisation: W_f is not allocated until the first ``absorb``
  call, at which point n_heads and head_dim are inferred automatically.
• Thread-unsafe: external locking is the caller's responsibility (same as
  KVLayerCache).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FastWeightConfig:
    """
    Hyperparameters for the TTT Fast Weight memory.

    Parameters
    ----------
    lr    : Per-absorption learning rate.  Controls how strongly each absorbed
            KV pair contributes to W_f.  Larger values remember recent tokens
            more strongly but may cause instability.
    decay : Per-absorption multiplicative decay applied to W_f before the new
            outer-product update.  1.0 = no forgetting; 0.0 = full overwrite.
    """
    lr:    float = 0.01
    decay: float = 0.999

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not (0.0 <= self.decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {self.decay}")


# ---------------------------------------------------------------------------
# Per-layer fast weight
# ---------------------------------------------------------------------------

class FastWeightLayer:
    """
    Single-layer fast-weight memory.

    Maintains one W_f matrix per attention head.  Shape is inferred lazily
    on the first call to :meth:`absorb`.

    Parameters
    ----------
    config : :class:`FastWeightConfig`
    """

    __slots__ = (
        "config",
        "_W_f",           # np.ndarray | None  (n_heads, head_dim, head_dim)
        "_n_heads",       # int — set on first absorb
        "_head_dim",      # int — set on first absorb
        "_n_absorptions", # int — cumulative token-level absorption count
    )

    def __init__(self, config: FastWeightConfig) -> None:
        self.config         = config
        self._W_f: np.ndarray | None = None
        self._n_heads        = 0
        self._head_dim       = 0
        self._n_absorptions  = 0

    # ── Initialization ────────────────────────────────────────────────────────

    def _ensure_initialized(self, n_heads: int, head_dim: int) -> None:
        """Lazily allocate W_f on the first absorb call."""
        if self._W_f is None:
            self._n_heads  = n_heads
            self._head_dim = head_dim
            self._W_f      = np.zeros((n_heads, head_dim, head_dim), dtype=np.float32)

    # ── Core operations ───────────────────────────────────────────────────────

    def absorb(self, keys: np.ndarray, values: np.ndarray) -> None:
        """
        Absorb one or more KV pairs into the fast-weight matrix.

        Parameters
        ----------
        keys   : (n_heads, n_tokens, head_dim)  *or*  (n_heads, head_dim)
        values : (n_heads, n_tokens, head_dim)  *or*  (n_heads, head_dim)

        The outer-product update is:
            W_f ← decay · W_f  +  lr · Σ_t( v_t ⊗ k_t )  (per head)
        """
        keys   = np.asarray(keys,   dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        if keys.ndim == 2:
            keys   = keys[:, np.newaxis, :]    # (n_heads, 1, head_dim)
            values = values[:, np.newaxis, :]

        n_heads, n_tokens, head_dim = keys.shape
        self._ensure_initialized(n_heads, head_dim)

        # Apply decay before new updates
        # Scalar decay on a full (n_heads, hd, hd) matrix
        if self.config.decay < 1.0:
            self._W_f *= np.float32(self.config.decay)

        # Outer-product accumulation:
        #   delta[h, e, d] = sum_t( values[h, t, e] * keys[h, t, d] )
        # Using einsum: "hte,htd->hed"
        delta = np.einsum("hte,htd->hed", values, keys, dtype=np.float32)  # (n_heads, hd, hd)
        self._W_f += np.float32(self.config.lr) * delta

        self._n_absorptions += n_tokens

    def query(self, queries: np.ndarray) -> np.ndarray:
        """
        Query the compressed history.

        Parameters
        ----------
        queries : (n_heads, head_dim)  — current query vectors

        Returns
        -------
        out : (n_heads, head_dim)  — linear-attention over absorbed history.
              Zero if no absorptions have occurred yet.
        """
        queries = np.asarray(queries, dtype=np.float32)
        n_heads, head_dim = queries.shape

        if self._W_f is None:
            return np.zeros((n_heads, head_dim), dtype=np.float32)

        # W_f @ q per head:  einsum("hed,hd->he", W_f, q)
        return np.einsum("hed,hd->he", self._W_f, queries, dtype=np.float32)

    def decay_step(self) -> None:
        """Apply one decay step without absorbing new tokens (e.g., on idle steps)."""
        if self._W_f is not None and self.config.decay < 1.0:
            self._W_f *= np.float32(self.config.decay)

    def reset(self) -> None:
        """Zero all fast-weight matrices (call at start of each request)."""
        if self._W_f is not None:
            self._W_f[:] = np.float32(0.0)
        self._n_absorptions = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._W_f is not None

    @property
    def n_absorptions(self) -> int:
        return self._n_absorptions

    @property
    def weight_norm(self) -> float:
        """Frobenius norm of W_f (all heads summed) — useful for diagnostics."""
        if self._W_f is None:
            return 0.0
        return float(np.linalg.norm(self._W_f))


# ---------------------------------------------------------------------------
# Multi-layer manager
# ---------------------------------------------------------------------------

class FastWeightManager:
    """
    Manages per-layer :class:`FastWeightLayer` instances for a full transformer.

    Held by :class:`~squish.kv.kv_cache.QuantizedKVCache` when
    ``fast_weight_lr > 0`` and exposed to :class:`~squish.kv.q_filters.QFilterManager`
    so that evicted KV pairs are absorbed before being discarded.

    Parameters
    ----------
    config   : :class:`FastWeightConfig`
    n_layers : Number of transformer layers (== number of KV cache layers).
    """

    def __init__(self, config: FastWeightConfig, n_layers: int) -> None:
        self.config   = config
        self.n_layers = n_layers
        self._layers  = [FastWeightLayer(config) for _ in range(n_layers)]

    # ── Per-layer delegation ──────────────────────────────────────────────────

    def absorb_layer(
        self,
        layer_idx: int,
        keys:      np.ndarray,
        values:    np.ndarray,
    ) -> None:
        """Absorb KV pairs for the given layer index."""
        self._layers[layer_idx].absorb(keys, values)

    def query_layer(
        self,
        layer_idx: int,
        queries:   np.ndarray,
    ) -> np.ndarray:
        """Query compressed history for the given layer index."""
        return self._layers[layer_idx].query(queries)

    def decay_step_layer(self, layer_idx: int) -> None:
        """Advance the decay for one layer without new absorptions."""
        self._layers[layer_idx].decay_step()

    # ── Bulk operations ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all layers (call at start of each request)."""
        for layer in self._layers:
            layer.reset()

    def decay_all(self) -> None:
        """Apply one decay step to all layers."""
        for layer in self._layers:
            layer.decay_step()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary statistics for monitoring / logging."""
        absorptions = [layer.n_absorptions for layer in self._layers]
        norms       = [layer.weight_norm   for layer in self._layers]
        return {
            "fast_weight_lr":           self.config.lr,
            "fast_weight_decay":        self.config.decay,
            "fast_weight_absorptions":  absorptions,
            "fast_weight_total_absorbed": sum(absorptions),
            "fast_weight_norms":        [round(n, 4) for n in norms],
        }
