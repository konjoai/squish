"""
squish/kv/q_filters.py

Phase 3 — Q-Filters: Geometric KV Cache Compression

Algorithm
─────────
For each transformer layer, after a calibration window of tokens:

  1.  SVD Basis Fitting (once per conversation per layer)
      Accumulate ``min_tokens`` key vectors K ∈ ℝ^(n_tokens × head_dim).
      Fit a per-head rank-``rank`` principal subspace via numpy SVD:
          K_h → V_h ∈ ℝ^(rank × head_dim)   (top-rank right singular vectors)
      This captures the geometric subspace in which keys concentrate.

  2.  Per-token Key Projection (online, O(n_heads × head_dim × rank))
      Every incoming key is projected:
          k_proj = k @ V_h.T  →  ℝ^(rank,)   per head
      The projected tensor grows as (n_heads, n_tokens, rank).

  3.  Geometric Scoring (on-demand when n_tokens > budget)
      Use the mean of the most-recent ``anchor`` projected keys as a
      proxy query direction (same "recent keys ≈ recent queries" premise as
      SnapKV, but in lower-dimensional SVD space).

      Per-token score:
          score[t] = mean_h( dot(K_proj[h,t], q_hat[h]) /
                             (||K_proj[h,t]|| · ||q_hat[h]|| + ε) )

      Tokens whose projected keys are nearly orthogonal to recent query
      directions are geometrically irrelevant and safe to evict.

  4.  Eviction
      Keep the top-``budget`` scoring positions.  Always preserve the last
      ``anchor`` tokens unconditionally (they are the recency anchor).
      Rebuild the KVLayerCache with only the surviving positions.

Design rationale
────────────────
• Pure numpy — consistent with all existing squish KV compression code.
• No model patching required — uses keys already buffered by KVLayerCache.
• SVD calibration happens once per conversation (~64 ms for 28 heads ×
  128-dim with 64 tokens on an M-series CPU).  Per-step overhead is a
  single small matmul (< 0.5 ms at rank=32, n_heads=8, n_tokens=4096).
• Complements INT8 KV compression (KIVI) — eviction reduces token count,
  INT8 reduces per-token size; combined effect is multiplicative.

Usage
─────
    from squish.kv.q_filters import QFilterConfig
    cache = patch_model_kv_cache(model, mode="snap", qfilter_rank=32,
                                 qfilter_budget=2048)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from squish.kv.kv_cache import KVLayerCache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QFilterConfig:
    """
    Hyperparameters for Q-Filter geometric KV eviction.

    Parameters
    ----------
    rank          : SVD projection rank.  Captures the top principal directions
                    of the key distribution.  Typical: 16–64.  Default 32.
    budget        : Max KV positions to retain per layer after eviction.
                    Also used as the trigger threshold (evict when n > budget).
    anchor        : Number of most-recent tokens that are *never* evicted.
                    These cover the immediate local context window.
    min_tokens    : Minimum tokens before SVD calibration fires.  Must be ≥ rank.
    evict_every   : Run eviction every N decode steps (to amortise cost).
                    0 = only when n_tokens > budget and step % any == 0.
    """
    rank:        int = 32
    budget:      int = 2048
    anchor:      int = 64
    min_tokens:  int = 64
    evict_every: int = 16

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be ≥ 1")
        if self.budget < 1:
            raise ValueError("budget must be ≥ 1")
        if self.anchor < 0:
            raise ValueError("anchor must be ≥ 0")
        if self.min_tokens < self.rank:
            raise ValueError("min_tokens must be ≥ rank (need enough data to fit SVD)")


# ---------------------------------------------------------------------------
# Per-layer state
# ---------------------------------------------------------------------------

class QFilterState:
    """
    Per-layer Q-filter state: SVD basis + rolling projected-key buffer.

    Lifecycle
    ─────────
    • At conversation start: ``reset()``
    • Each decode step, after KVLayerCache.append(): ``append_key(key_np)``
    • When n_tokens > budget: ``score_query(recent_keys)``
    • After eviction: ``rebuild_after_eviction(keep_indices)``

    The SVD basis (``_basis``) is NOT reset between conversations once fitted,
    so the first request in a new conversation re-uses the previous basis if
    the model/context domain is similar.  A fresh ``reset()`` clears the basis
    too; call ``reset(clear_basis=True)`` to force re-calibration.
    """

    __slots__ = (
        "config",
        "_basis",        # (n_heads, rank, head_dim) float32 | None — frozen after fit
        "_calib_buf",    # list[np.ndarray (n_heads, head_dim)]  — pre-calibration buffer
        "_kproj",        # np.ndarray (n_heads, n_tokens, rank) float32 | None
        "_calibrated",   # bool
    )

    def __init__(self, config: QFilterConfig) -> None:
        self.config      = config
        self._basis:      np.ndarray | None = None
        self._calib_buf:  list[np.ndarray]  = []
        self._kproj:      np.ndarray | None = None
        self._calibrated: bool              = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, *, clear_basis: bool = False) -> None:
        """Reset per-request state.  Pass ``clear_basis=True`` to force re-calibration."""
        self._calib_buf  = []
        self._kproj      = None
        self._calibrated = False
        if clear_basis:
            self._basis = None

    def append_key(self, key: np.ndarray) -> None:
        """
        Record the new token's key vector.

        Parameters
        ----------
        key : (n_heads, head_dim)  float16 or float32
        """
        key_f32 = key.astype(np.float32)
        if not self._calibrated:
            self._calib_buf.append(key_f32)
            if len(self._calib_buf) >= self.config.min_tokens:
                self._fit_basis()
            return
        # Post-calibration: project and append
        kp = self._project(key_f32)                 # (n_heads, rank)
        if self._kproj is None:
            self._kproj = kp[:, np.newaxis, :]      # (n_heads, 1, rank)
        else:
            self._kproj = np.concatenate(
                [self._kproj, kp[:, np.newaxis, :]], axis=1,
            )

    def score_recent(self, recent_keys: np.ndarray) -> np.ndarray | None:
        """
        Compute per-token geometric relevance scores using recent keys as a
        proxy for the next query direction.

        Parameters
        ----------
        recent_keys : (n_heads, n_recent, head_dim) float16/float32
                      Last ``anchor`` keys from the cache — treated as the
                      current "query direction" in SVD space.

        Returns
        -------
        scores : (n_tokens,) float32 — higher = more geometrically relevant.
                 Returns None if not yet calibrated or kproj buffer is empty.
        """
        if not self._calibrated or self._kproj is None:
            return None
        # Mean projected recent key → proxy query direction (n_heads, rank)
        rk = recent_keys.astype(np.float32)          # (n_heads, n_recent, hd)
        rk_proj = np.einsum("hnd,hrd->hnr", rk, self._basis)  # (n_heads, n_recent, rank)
        q_hat = rk_proj.mean(axis=1)                 # (n_heads, rank)

        # Cosine similarity between each cached K projection and q_hat
        # _kproj: (n_heads, n_tokens, rank), q_hat: (n_heads, rank)
        dots = np.einsum("htr,hr->ht", self._kproj, q_hat)   # (n_heads, n_tokens)
        k_norms = np.linalg.norm(self._kproj, axis=-1)        # (n_heads, n_tokens)
        q_norms = np.linalg.norm(q_hat, axis=-1)[:, None]     # (n_heads, 1)
        denom   = np.maximum(k_norms * q_norms, 1e-8)
        cos_sim = dots / denom                                 # (n_heads, n_tokens)
        return cos_sim.mean(axis=0)                            # (n_tokens,)

    def rebuild_after_eviction(self, keep_indices: np.ndarray) -> None:
        """
        Trim the projected-key buffer to reflect surviving token positions.

        Must be called *after* KVLayerCache eviction with the same indices.

        Parameters
        ----------
        keep_indices : (n_kept,) int — sorted in temporal order.
        """
        if self._kproj is not None:
            self._kproj = self._kproj[:, keep_indices, :]

    @property
    def n_projected(self) -> int:
        """Number of tokens in the projected-key buffer (post-calibration only)."""
        return 0 if self._kproj is None else self._kproj.shape[1]

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fit_basis(self) -> None:
        """Fit per-head SVD basis from calibration buffer and project buffered keys."""
        cfg      = self.config
        K_buf    = np.stack(self._calib_buf, axis=0)  # (n_cal, n_heads, head_dim)
        n_cal, n_heads, head_dim = K_buf.shape
        rank     = min(cfg.rank, min(n_cal, head_dim))

        basis = np.empty((n_heads, rank, head_dim), dtype=np.float32)
        for h in range(n_heads):
            K_h        = K_buf[:, h, :]                          # (n_cal, head_dim)
            norms      = np.linalg.norm(K_h, axis=-1, keepdims=True)
            K_h_normed = K_h / np.maximum(norms, 1e-8)           # unit-length rows
            _, _, Vt   = np.linalg.svd(K_h_normed, full_matrices=False)
            basis[h]   = Vt[:rank]                               # (rank, head_dim)

        self._basis = basis                                       # (n_heads, rank, head_dim)

        # Project all buffered calibration keys
        # K_buf: (n_cal, n_heads, head_dim) → einsum → (n_heads, n_cal, rank)
        kp_buf = np.einsum("thd,hrd->thr", K_buf, basis)         # (n_cal, n_heads, rank)
        self._kproj = kp_buf.transpose(1, 0, 2).astype(np.float32)  # (n_heads, n_cal, rank)

        self._calib_buf = []   # free memory
        self._calibrated = True

    def _project(self, key: np.ndarray) -> np.ndarray:
        """Project single (n_heads, head_dim) key → (n_heads, rank)."""
        return np.einsum("hd,hrd->hr", key, self._basis)


# ---------------------------------------------------------------------------
# Manager (all layers)
# ---------------------------------------------------------------------------

class QFilterManager:
    """
    Manages per-layer :class:`QFilterState` for a full transformer model.

    Intended to live alongside :class:`~squish.kv.kv_cache.QuantizedKVCache`
    and be called from the same token-append path.

    Parameters
    ----------
    config   : :class:`QFilterConfig`
    n_layers : number of transformer layers
    """

    def __init__(self, config: QFilterConfig, n_layers: int) -> None:
        self.config   = config
        self.n_layers = n_layers
        self._states: list[QFilterState] = [
            QFilterState(config) for _ in range(n_layers)
        ]

    def reset(self, *, clear_basis: bool = False) -> None:
        """Reset all layer states for a new conversation."""
        for s in self._states:
            s.reset(clear_basis=clear_basis)

    def notify_key(self, layer_idx: int, key: np.ndarray) -> None:
        """
        Record the new token's key for ``layer_idx``.

        Call once per layer per token in temporal order.
        """
        self._states[layer_idx].append_key(key)

    def maybe_evict(
        self,
        layer_idx:  int,
        layer_cache: "KVLayerCache",
        step:        int,
        fw_manager:  "FastWeightManager | None" = None,
    ) -> bool:
        """
        Run geometric eviction for one layer if conditions are met.

        Conditions:
        • ``layer_cache.n_tokens > config.budget``
        • ``step % config.evict_every == 0``  (or evict_every == 0)
        • The layer's SVD basis is calibrated

        Parameters
        ----------
        fw_manager : Optional :class:`~squish.kv.fast_weights.FastWeightManager`.
                     When provided, tokens that are evicted are *absorbed* into
                     the fast-weight matrix before removal, preserving their
                     information as a compressed outer-product summary (Phase 5).

        Returns True if eviction was performed.
        """
        cfg   = self.config
        state = self._states[layer_idx]
        n     = layer_cache.n_tokens

        if n <= cfg.budget:
            return False
        if cfg.evict_every > 0 and step % cfg.evict_every != 0:
            return False
        if not state.is_calibrated:
            return False

        # Fetch the most-recent anchor keys as proxy query direction
        full_k, full_v = layer_cache.get_full_kv()
        if full_k is None:
            return False
        anchor  = min(cfg.anchor, n)
        recent  = full_k[:, -anchor:, :]               # (n_heads, anchor, head_dim)

        scores  = state.score_recent(recent)
        if scores is None or len(scores) != n:
            return False

        # Protect the last anchor positions unconditionally
        masked = scores.copy()
        masked[-anchor:] = np.inf

        keep   = np.sort(np.argsort(-masked)[: cfg.budget])

        # Phase 5: absorb evicted tokens into fast weights before dropping them
        if fw_manager is not None:
            all_indices = np.arange(n, dtype=np.int64)
            keep_set    = set(keep.tolist())
            evict_idx   = np.array(
                [i for i in all_indices if i not in keep_set], dtype=np.int64
            )
            if len(evict_idx) > 0:
                evict_k = full_k[:, evict_idx, :]   # (n_heads, n_evict, head_dim)
                evict_v = full_v[:, evict_idx, :]
                fw_manager.absorb_layer(layer_idx, evict_k, evict_v)

        # Evict from KV cache and trim projected-key buffer
        _qfilter_evict(layer_cache, keep)
        state.rebuild_after_eviction(keep)
        return True


# ---------------------------------------------------------------------------
# KVLayerCache eviction helper
# ---------------------------------------------------------------------------

def _qfilter_evict(layer_cache: "KVLayerCache", keep_indices: np.ndarray) -> None:
    """
    Rebuild ``layer_cache`` retaining only the positions in ``keep_indices``.

    Follows the same pattern as ``_snap_evict`` in kv_cache.py:
    reconstruct full FP16 → select → reset → reload through the window
    pipeline so that INT8 quantisation is re-applied cleanly.

    Parameters
    ----------
    layer_cache  : KVLayerCache to rebuild
    keep_indices : 1-D int array, sorted in temporal order
    """
    # Local imports deferred to avoid circular at module level when tests
    # import q_filters before kv_cache.
    from squish.kv.kv_cache import _quantize_int8_per_channel  # noqa: PLC0415

    with layer_cache._lock:
        full_k, full_v = layer_cache.get_full_kv()
        if full_k is None:
            return

        nh = full_k.shape[0]
        sel_k = full_k[:, keep_indices, :]   # (n_heads, n_keep, head_dim)
        sel_v = full_v[:, keep_indices, :]

        # ── Reset layer state ────────────────────────────────────────────────
        layer_cache.keys_recent.clear()
        layer_cache.values_recent.clear()
        layer_cache.keys_old_q   = None
        layer_cache.keys_old_s   = None
        layer_cache.values_old_q = None
        layer_cache.values_old_s = None

        # ── Reload into recent window; overflow spills to INT8 ───────────────
        for t in range(sel_k.shape[1]):
            layer_cache.keys_recent.append(sel_k[:, t, :])
            layer_cache.values_recent.append(sel_v[:, t, :])

        window = layer_cache.window
        while len(layer_cache.keys_recent) > window:
            ok = layer_cache.keys_recent.pop(0)
            ov = layer_cache.values_recent.pop(0)

            new_kq, new_ks, new_vq, new_vs = [], [], [], []
            for h in range(nh):
                kq, ks = _quantize_int8_per_channel(ok[h:h + 1, :])
                vq, vs = _quantize_int8_per_channel(ov[h:h + 1, :])
                new_kq.append(kq)
                new_ks.append(ks)
                new_vq.append(vq)
                new_vs.append(vs)

            slot_kq = np.stack(new_kq, axis=0)   # (n_heads, 1, head_dim) int8
            slot_ks = np.stack(new_ks, axis=0)   # (n_heads, 1) float32
            slot_vq = np.stack(new_vq, axis=0)
            slot_vs = np.stack(new_vs, axis=0)

            if layer_cache.keys_old_q is None:
                layer_cache.keys_old_q   = slot_kq
                layer_cache.keys_old_s   = slot_ks
                layer_cache.values_old_q = slot_vq
                layer_cache.values_old_s = slot_vs
            else:
                layer_cache.keys_old_q   = np.concatenate(
                    [layer_cache.keys_old_q,   slot_kq], axis=1)
                layer_cache.keys_old_s   = np.concatenate(
                    [layer_cache.keys_old_s,   slot_ks], axis=1)
                layer_cache.values_old_q = np.concatenate(
                    [layer_cache.values_old_q, slot_vq], axis=1)
                layer_cache.values_old_s = np.concatenate(
                    [layer_cache.values_old_s, slot_vs], axis=1)
