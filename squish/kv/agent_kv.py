#!/usr/bin/env python3
"""
squish/agent_kv.py

AgentKV — Asymmetric INT2 KV-cache for long-running agent loops.

Implements the three-zone KV layout from the SnapKV / StreamingLLM / GEAR
family of techniques:

┌─────────────┬─────────────────────┬──────────────────┐
│  Sink zone  │   History (INT2)    │  Local window    │
│  FP32 · K   │  2-bit quantised ·  │  FP32 · W tokens │
│  tokens     │  (len – K – W) tok  │                  │
└─────────────┴─────────────────────┴──────────────────┘

* **Sink zone** (first *sink_tokens* positions): preserved at full FP32
  precision because attention consistently concentrates on initial tokens
  (StreamingLLM / BigBird).

* **Local window** (last *window_tokens* positions): kept at FP32 because
  the most recent context is still actively attended.

* **History** (everything between sink and window): quantised to 2-bit
  INT2 per-head symmetric quantisation, halving the per-token memory vs
  INT4 and providing 16× compression vs FP32.

Quantisation scheme
───────────────────
Per-head symmetric INT2 with bit-packing (4 values per byte):

  scale  = max(abs(x)) / 1.5
  q      = round(x / scale * 1.5 + 1.5)  →  clipped to [0, 3]
  x_hat  = (q − 1.5) / 1.5 × scale

This maps the floating-point range [−scale, +scale] uniformly to the
integer grid {0, 1, 2, 3} (unsigned 2-bit integers).

The *head_dim* of keys and values must be divisible by 4 (true for all
standard transformer shapes: 64, 128, 256 …).

Usage::

    from squish.kv.agent_kv import AgentKVConfig, AgentKVCache
    import numpy as np

    cfg   = AgentKVConfig(sink_tokens=4, window_tokens=64,
                          n_heads=8, head_dim=128, n_layers=32)
    cache = AgentKVCache(cfg)

    # Per-layer append — k/v shape: (n_heads, 1, head_dim)
    for layer in range(32):
        cache.append(layer, k_layer, v_layer)

    # Retrieve full KV for attention — returns FP32
    k, v = cache.get(layer)           # (n_heads, seq_len, head_dim)
    print(cache.stats)                # AgentKVStats(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "AgentKVConfig",
    "AgentKVCache",
    "AgentKVStats",
    "Int2Block",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentKVConfig:
    """Configuration for the asymmetric INT2 KV cache.

    Parameters
    ----------
    sink_tokens : int
        Number of initial (sink) tokens to always keep at FP32.
        Must be ≥ 0.
    window_tokens : int
        Number of most-recent tokens to always keep at FP32.
        Must be ≥ 1.
    n_heads : int
        Number of KV attention heads.
    head_dim : int
        Per-head key/value dimensionality.  Must be divisible by 4 for
        2-bit packing.
    n_layers : int
        Number of transformer layers to manage.
    """

    sink_tokens:   int = 4
    window_tokens: int = 64
    n_heads:       int = 8
    head_dim:      int = 128
    n_layers:      int = 32

    def __post_init__(self) -> None:
        if self.sink_tokens < 0:
            raise ValueError(f"sink_tokens must be ≥ 0; got {self.sink_tokens}")
        if self.window_tokens < 1:
            raise ValueError(f"window_tokens must be ≥ 1; got {self.window_tokens}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for INT2 packing; "
                f"got {self.head_dim}"
            )
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1; got {self.n_layers}")


# ---------------------------------------------------------------------------
# INT2 packing helpers
# ---------------------------------------------------------------------------


def _quantise_int2(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantise *x* to 2-bit unsigned integers with per-head symmetric scaling.

    Parameters
    ----------
    x : np.ndarray of shape ``(n_heads, n_tokens, head_dim)``

    Returns
    -------
    packed : np.ndarray, uint8, shape ``(n_heads, n_tokens, head_dim // 4)``
        Bit-packed INT2 values (4 per byte, little-endian nibble order).
    scales : np.ndarray, float32, shape ``(n_heads, n_tokens, 1)``
        Per-head-per-token quantisation scale.
    """
    # Per-(head, token) scale: max absolute value / 1.5 maps to mid-point of
    # the unsigned INT2 range [0, 3] which spans (−1.5 × scale, +1.5 × scale).
    abs_max = np.abs(x).max(axis=-1, keepdims=True)  # (H, T, 1)
    scales = np.where(abs_max == 0, np.ones_like(abs_max), abs_max / 1.5)

    # Forward quantise: x → q ∈ {0, 1, 2, 3}
    q = np.round(x / scales * 1.5 + 1.5).clip(0, 3).astype(np.uint8)

    # Bit-pack: 4 successive values along head_dim → 1 byte
    #   packed[..., k] = q[..., 4k] | q[..., 4k+1]<<2 | q[..., 4k+2]<<4 | q[..., 4k+3]<<6
    n_heads, n_tokens, head_dim = q.shape
    q_r = q.reshape(n_heads, n_tokens, head_dim // 4, 4)
    packed = (
        q_r[..., 0]
        | (q_r[..., 1] << 2)
        | (q_r[..., 2] << 4)
        | (q_r[..., 3] << 6)
    ).astype(np.uint8)

    return packed, scales.astype(np.float32)


def _dequantise_int2(
    packed: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Reconstruct float32 values from INT2 bit-packed arrays.

    Parameters
    ----------
    packed : np.ndarray, uint8, shape ``(n_heads, n_tokens, head_dim // 4)``
    scales : np.ndarray, float32, shape ``(n_heads, n_tokens, 1)``

    Returns
    -------
    np.ndarray, float32, shape ``(n_heads, n_tokens, head_dim)``
    """
    n_heads, n_tokens, packed_dim = packed.shape
    head_dim = packed_dim * 4

    # Unpack bytes back to 4 uint8 values per packed element
    q0 =  packed        & 0x3
    q1 = (packed >> 2)  & 0x3
    q2 = (packed >> 4)  & 0x3
    q3 = (packed >> 6)  & 0x3
    # Interleave to restore head_dim order: [q0, q1, q2, q3, q0, q1, ...]
    q = np.stack([q0, q1, q2, q3], axis=-1).reshape(n_heads, n_tokens, head_dim)

    # Inverse: q ∈ {0,1,2,3} → float = (q − 1.5) / 1.5 × scale
    return ((q.astype(np.float32) - 1.5) / 1.5) * scales


# ---------------------------------------------------------------------------
# Int2Block — a compressed chunk of K or V history
# ---------------------------------------------------------------------------


@dataclass
class Int2Block:
    """One zone of INT2-compressed KV data.

    Attributes
    ----------
    packed : np.ndarray, uint8
        Bit-packed quantised values, shape
        ``(n_heads, n_tokens, head_dim // 4)``.
    scales : np.ndarray, float32
        Per-head-per-token dequantisation scales, shape
        ``(n_heads, n_tokens, 1)``.
    """

    packed: np.ndarray
    scales: np.ndarray

    @property
    def n_tokens(self) -> int:
        """Number of tokens stored in this block."""
        return self.packed.shape[1]

    def to_float32(self) -> np.ndarray:
        """Decompress to float32, shape ``(n_heads, n_tokens, head_dim)``."""
        return _dequantise_int2(self.packed, self.scales)

    @staticmethod
    def from_float32(x: np.ndarray) -> Int2Block:
        """Compress *x* (shape ``(n_heads, n_tokens, head_dim)``) to INT2."""
        packed, scales = _quantise_int2(x)
        return Int2Block(packed=packed, scales=scales)


# ---------------------------------------------------------------------------
# Per-layer KV store
# ---------------------------------------------------------------------------


@dataclass
class _LayerKV:
    """Internal per-layer KV storage in three zones."""

    sink_k:    np.ndarray | None = None   # (H, sink, D)
    sink_v:    np.ndarray | None = None
    hist_k:    Int2Block | None  = None   # compressed history
    hist_v:    Int2Block | None  = None
    window_k:  np.ndarray | None = None   # (H, W, D) — most recent
    window_v:  np.ndarray | None = None

    @property
    def total_tokens(self) -> int:
        s = self.sink_k.shape[1]   if self.sink_k   is not None else 0
        h = self.hist_k.n_tokens   if self.hist_k   is not None else 0
        w = self.window_k.shape[1] if self.window_k is not None else 0
        return s + h + w


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class AgentKVStats:
    """Summary statistics for an :class:`AgentKVCache` instance.

    Attributes
    ----------
    total_tokens : int
        Total tokens stored across all layers (average per layer × n_layers).
    sink_tokens : int
        Tokens in the FP32 sink zone per layer.
    history_tokens : int
        Tokens in the INT2 history zone per layer.
    window_tokens : int
        Tokens in the FP32 local window per layer.
    compactions : int
        Total number of compact() operations performed.
    estimated_bytes : int
        Approximate memory footprint in bytes.
    """

    total_tokens:    int = 0
    sink_tokens:     int = 0
    history_tokens:  int = 0
    window_tokens:   int = 0
    compactions:     int = 0
    estimated_bytes: int = 0


# ---------------------------------------------------------------------------
# AgentKVCache
# ---------------------------------------------------------------------------


class AgentKVCache:
    """Asymmetric INT2 KV cache: sink (FP32) + history (INT2) + window (FP32).

    Parameters
    ----------
    config : AgentKVConfig
    """

    def __init__(self, config: AgentKVConfig) -> None:
        self._cfg = config
        self._layers: list[_LayerKV] = [_LayerKV() for _ in range(config.n_layers)]
        self._compactions = 0

    # ── Write path ────────────────────────────────────────────────────────

    def append(
        self,
        layer_idx: int,
        k: np.ndarray,
        v: np.ndarray,
    ) -> None:
        """Append new key/value tensors for one layer.

        After appending, :meth:`compact` is called automatically if the local
        window has grown beyond :attr:`AgentKVConfig.window_tokens`.

        Parameters
        ----------
        layer_idx : int
            Layer index in ``[0, n_layers)``.
        k : np.ndarray, float32
            Keys of shape ``(n_heads, n_new, head_dim)``.
        v : np.ndarray, float32
            Values of shape ``(n_heads, n_new, head_dim)``.
        """
        self._validate_layer(layer_idx)
        lkv = self._layers[layer_idx]

        # Concatenate onto local window
        if lkv.window_k is None:
            lkv.window_k = k.astype(np.float32)
            lkv.window_v = v.astype(np.float32)
        else:
            lkv.window_k = np.concatenate([lkv.window_k, k.astype(np.float32)], axis=1)
            lkv.window_v = np.concatenate([lkv.window_v, v.astype(np.float32)], axis=1)

        # Auto-compact if window overflows
        if lkv.window_k.shape[1] > self._cfg.window_tokens:
            self.compact(layer_idx)

    def compact(self, layer_idx: int) -> None:
        """Compress window overflow into the INT2 history zone.

        Tokens that overflow the local window (oldest window tokens) are
        checked against the sink budget:

        * If the total token count is still within *sink_tokens*, overflow
          goes into the sink zone (FP32).
        * Otherwise overflow is quantised and appended to the INT2 history.

        This is called automatically by :meth:`append` but can also be
        triggered manually to force compaction ahead of a memory-pressure
        event.
        """
        self._validate_layer(layer_idx)
        lkv = self._layers[layer_idx]
        if lkv.window_k is None:
            return
        assert lkv.window_v is not None  # always set together with window_k

        window_len = lkv.window_k.shape[1]
        overflow   = window_len - self._cfg.window_tokens
        if overflow <= 0:
            return  # nothing to compact

        # Tokens to evict from window front
        evict_k = lkv.window_k[:, :overflow, :]
        evict_v = lkv.window_v[:, :overflow, :]
        lkv.window_k = lkv.window_k[:, overflow:, :]
        lkv.window_v = lkv.window_v[:, overflow:, :]

        # Determine how many tokens remain sink-eligible
        sink_used = lkv.sink_k.shape[1] if lkv.sink_k is not None else 0
        sink_cap  = self._cfg.sink_tokens
        sink_available = max(0, sink_cap - sink_used)

        if sink_available > 0:
            # Some or all evicted tokens go into FP32 sink
            to_sink = min(sink_available, overflow)
            sink_k_new = evict_k[:, :to_sink, :]
            sink_v_new = evict_v[:, :to_sink, :]
            evict_k    = evict_k[:, to_sink:, :]
            evict_v    = evict_v[:, to_sink:, :]
            if lkv.sink_k is None:
                lkv.sink_k = sink_k_new
                lkv.sink_v = sink_v_new
            else:
                lkv.sink_k = np.concatenate([lkv.sink_k, sink_k_new], axis=1)
                lkv.sink_v = np.concatenate([lkv.sink_v, sink_v_new], axis=1)

        if evict_k.shape[1] == 0:
            self._compactions += 1
            return

        # Remaining tokens → INT2 history
        new_hist_k = Int2Block.from_float32(evict_k)
        new_hist_v = Int2Block.from_float32(evict_v)
        if lkv.hist_k is None:
            lkv.hist_k = new_hist_k
            lkv.hist_v = new_hist_v
        else:
            assert lkv.hist_v is not None  # always set together with hist_k
            # Merge by decompressing, concatenating, recompressing
            # (avoids unbounded block fragmentation)
            merged_k = np.concatenate(
                [lkv.hist_k.to_float32(), evict_k], axis=1
            )
            merged_v = np.concatenate(
                [lkv.hist_v.to_float32(), evict_v], axis=1
            )
            lkv.hist_k = Int2Block.from_float32(merged_k)
            lkv.hist_v = Int2Block.from_float32(merged_v)

        self._compactions += 1

    # ── Read path ─────────────────────────────────────────────────────────

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the full KV sequence for *layer_idx*.

        Returns concatenated ``(keys, values)`` in causal order:
        sink → history (decompressed) → local window.

        Returns
        -------
        k : np.ndarray, float32, shape ``(n_heads, total_tokens, head_dim)``
        v : np.ndarray, float32, shape ``(n_heads, total_tokens, head_dim)``
        """
        self._validate_layer(layer_idx)
        lkv     = self._layers[layer_idx]
        k_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []

        if lkv.sink_k is not None:
            k_parts.append(lkv.sink_k)
            v_parts.append(lkv.sink_v)  # type: ignore[arg-type]
        if lkv.hist_k is not None:
            k_parts.append(lkv.hist_k.to_float32())
            v_parts.append(lkv.hist_v.to_float32())  # type: ignore[union-attr]
        if lkv.window_k is not None:
            k_parts.append(lkv.window_k)
            v_parts.append(lkv.window_v)  # type: ignore[arg-type]

        if not k_parts:
            # Empty cache — return zero-length arrays
            cfg = self._cfg
            empty = np.zeros((cfg.n_heads, 0, cfg.head_dim), dtype=np.float32)
            return empty, empty.copy()

        return np.concatenate(k_parts, axis=1), np.concatenate(v_parts, axis=1)

    # ── Cache management ──────────────────────────────────────────────────

    def evict_history(self, layer_idx: int, n_tokens: int) -> int:
        """Remove the oldest *n_tokens* from the INT2 history zone.

        Used by a :class:`~squish.serving.memory_governor.MemoryGovernor` callback
        to shed cache under memory pressure.

        Parameters
        ----------
        layer_idx : int
        n_tokens : int
            Number of tokens to drop from history (≥ 0).

        Returns
        -------
        int
            Actual number of tokens removed (≤ *n_tokens*).
        """
        self._validate_layer(layer_idx)
        if n_tokens <= 0:
            return 0
        lkv = self._layers[layer_idx]
        if lkv.hist_k is None:
            return 0
        current = lkv.hist_k.n_tokens
        to_drop = min(n_tokens, current)
        if to_drop == current:
            lkv.hist_k = None
            lkv.hist_v = None
        else:
            keep_k = lkv.hist_k.to_float32()[:, to_drop:, :]
            keep_v = lkv.hist_v.to_float32()[:, to_drop:, :]  # type: ignore[union-attr]
            lkv.hist_k = Int2Block.from_float32(keep_k)
            lkv.hist_v = Int2Block.from_float32(keep_v)
        return to_drop

    def reset(self, layer_idx: int | None = None) -> None:
        """Clear cached data.

        Parameters
        ----------
        layer_idx : int or None
            If given, clears only that layer; otherwise clears all layers.
        """
        if layer_idx is None:
            self._layers = [_LayerKV() for _ in range(self._cfg.n_layers)]
            self._compactions = 0
        else:
            self._validate_layer(layer_idx)
            self._layers[layer_idx] = _LayerKV()

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def stats(self) -> AgentKVStats:
        """Return an :class:`AgentKVStats` snapshot of current state."""
        cfg  = self._cfg
        sink_t = hist_t = win_t = 0
        for lkv in self._layers:
            if lkv.sink_k   is not None: sink_t += lkv.sink_k.shape[1]
            if lkv.hist_k   is not None: hist_t += lkv.hist_k.n_tokens
            if lkv.window_k is not None: win_t  += lkv.window_k.shape[1]

        # FP32 bytes (4 bytes × n_heads × tokens × head_dim × 2 (k+v))
        fp32_per_token = 4 * cfg.n_heads * cfg.head_dim * 2
        # INT2 bytes (0.25 bytes per value × n_heads × head_dim × 2, plus scale)
        int2_per_token = (
            (cfg.n_heads * cfg.head_dim * 2) // 4   # packed uint8
            + cfg.n_heads * 4 * 2                    # float32 scales (key+val)
        )
        n = cfg.n_layers
        estimated = (
            (sink_t + win_t) * fp32_per_token
            + hist_t          * int2_per_token
        )
        return AgentKVStats(
            total_tokens=sink_t + hist_t + win_t,
            sink_tokens=sink_t // n if n else 0,
            history_tokens=hist_t // n if n else 0,
            window_tokens=win_t // n if n else 0,
            compactions=self._compactions,
            estimated_bytes=estimated,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self._cfg.n_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range "
                f"[0, {self._cfg.n_layers})"
            )

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"AgentKVCache("
            f"layers={self._cfg.n_layers}, "
            f"sink={s.sink_tokens}, "
            f"hist={s.history_tokens}[INT2], "
            f"window={s.window_tokens}, "
            f"~{s.estimated_bytes//(1<<20)}MB)"
        )
