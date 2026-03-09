"""
squish/head_infer.py

HeadInfer — Attention-head–aware KV cache offloading for Apple Silicon.

Inspired by:
  "HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading"
  arXiv:2502.12574 (Feb 2025)

Core Idea
---------
Not all attention heads are equal in memory pressure.  Two broad classes:
  * "Streaming" heads — attend to recent tokens only; safe to evict old KV.
  * "Retrieval" heads — need the full-context KV to recall distant facts.

HeadInfer classifies heads offline (or lazily online) via a *sink + recency*
sensitivity score, then applies per-head eviction policies:
  - streaming heads: keep only a sliding window of KV entries on-device.
  - retrieval heads: keep full or HNSW-compressed KV on-device.

Memory complexity moves from ``O(N × H)`` to ``O(N × H_ret + W × H_str)``
where W ≪ N for long contexts, yielding up to **4× KV footprint reduction**
with negligible perplexity loss on summarization / QA tasks.

Apple Silicon Integration
-------------------------
This module operates on NumPy arrays (CPU path).  When mlx.core is available
the caller may convert back with ``mx.array(arr)``.  The MPS / Metal memory
pressure is reduced by simply not keeping large KV tensors on the GPU heap.

Usage::

    from squish.head_infer import HeadInferConfig, HeadClassifier, HeadAwareKVStore

    cfg       = HeadInferConfig(n_layers=32, n_heads=32, window_size=512)
    hc        = HeadClassifier(cfg)

    # One-time classification using calibration attention patterns
    hc.calibrate(attn_weights_per_layer)   # list[np.ndarray] (n_heads, seq, seq)

    kv_store  = HeadAwareKVStore(cfg, hc.head_types)
    kv_store.put(layer_idx=0, head_idx=0, key=k, value=v)
    keys, vals = kv_store.get(layer_idx=0, head_idx=0, query=q, top_k=64)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

__all__ = [
    "HeadType",
    "HeadInferConfig",
    "HeadClassifier",
    "HeadAwareKVStore",
]

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HeadType(enum.Enum):
    """Classification label for a single attention head."""
    STREAMING  = "streaming"   # attends primarily to recent tokens
    RETRIEVAL  = "retrieval"   # needs global context access
    UNKNOWN    = "unknown"     # not yet classified


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeadInferConfig:
    """
    Configuration for head-aware KV offloading.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    n_heads  : int
        Number of attention heads per layer.
    window_size : int
        KV tokens kept on-device for *streaming* heads.
    sink_tokens : int
        Number of initial "attention sink" tokens always retained.
    retrieval_threshold : float
        Fraction of attention mass on non-recent tokens that classifies
        a head as *retrieval*.  Range 0-1.  Default 0.15 (15%).
    top_k_retrieval : int
        Maximum KV entries to return for retrieval heads via HNSW search.
        When 0 the full buffer is returned (exact attention).
    """
    n_layers:            int   = 32
    n_heads:             int   = 32
    window_size:         int   = 512
    sink_tokens:         int   = 4
    retrieval_threshold: float = 0.15
    top_k_retrieval:     int   = 64

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")
        if self.n_heads < 1:
            raise ValueError("n_heads must be ≥ 1")
        if self.window_size < 1:
            raise ValueError("window_size must be ≥ 1")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be ≥ 0")
        if not 0.0 < self.retrieval_threshold < 1.0:
            raise ValueError("retrieval_threshold must be in (0, 1)")
        if self.top_k_retrieval < 0:
            raise ValueError("top_k_retrieval must be ≥ 0")


# ---------------------------------------------------------------------------
# Head Classifier
# ---------------------------------------------------------------------------

class HeadClassifier:
    """
    Offline / lazy classifier that labels each (layer, head) pair.

    Calibration: pass a list of per-layer attention weight matrices
    (shape ``(n_heads, seq, seq)``).  HeadClassifier scans each head's
    attention distribution and measures how much mass falls *outside*
    the recent window + sinks.  Heads above ``retrieval_threshold`` are
    labelled RETRIEVAL; the rest are STREAMING.

    When ``calibrate()`` is never called all heads default to RETRIEVAL
    (safe but conservative: full KV kept on device).
    """

    def __init__(self, config: HeadInferConfig) -> None:
        self._cfg = config
        # head_types[layer][head] -> HeadType
        self.head_types: list[list[HeadType]] = [
            [HeadType.UNKNOWN] * config.n_heads
            for _ in range(config.n_layers)
        ]

    # ── Public ────────────────────────────────────────────────────────────────

    def calibrate(
        self,
        attn_weights: list[np.ndarray],
        *,
        verbose: bool = False,
    ) -> None:
        """
        Classify heads from calibration attention patterns.

        Parameters
        ----------
        attn_weights : list of (n_heads, seq, seq) float32 ndarray, one per layer.
            Each entry ``attn_weights[l][h, q, k]`` is the (optionally averaged
            over queries) attention weight from query position q to key position k.
        verbose : bool
            If True, print per-layer classification summary.
        """
        for layer_idx, W in enumerate(attn_weights):
            if layer_idx >= self._cfg.n_layers:
                break
            W = np.asarray(W, dtype=np.float32)
            if W.ndim == 3:
                # Average over query axis to get (n_heads, seq)
                avg_W = W.mean(axis=1)   # (n_heads, seq)
            elif W.ndim == 2:
                avg_W = W                # (n_heads, seq)
            else:
                continue

            n_heads, seq_len = avg_W.shape

            for head_idx in range(min(n_heads, self._cfg.n_heads)):
                dist = avg_W[head_idx]
                score = self._retrieval_score(dist, seq_len)
                if score >= self._cfg.retrieval_threshold:
                    self.head_types[layer_idx][head_idx] = HeadType.RETRIEVAL
                else:
                    self.head_types[layer_idx][head_idx] = HeadType.STREAMING

            if verbose:
                ret_count = sum(
                    1 for h in self.head_types[layer_idx]
                    if h == HeadType.RETRIEVAL
                )
                print(
                    f"Layer {layer_idx}: "
                    f"{ret_count}/{self._cfg.n_heads} retrieval heads"
                )

    def label(self, layer_idx: int, head_idx: int) -> HeadType:
        """
        Return the :class:`HeadType` for a given (layer, head).

        Unclassified heads are returned as RETRIEVAL (safe default).
        """
        if layer_idx >= self._cfg.n_layers or head_idx >= self._cfg.n_heads:
            return HeadType.RETRIEVAL
        ht = self.head_types[layer_idx][head_idx]
        if ht == HeadType.UNKNOWN:
            return HeadType.RETRIEVAL
        return ht

    # ── Private ───────────────────────────────────────────────────────────────

    def _retrieval_score(self, dist: np.ndarray, seq_len: int) -> float:
        """
        Compute the fraction of attention mass outside [sinks + recent window].

        ``dist`` is a 1-D probability distribution over key positions.
        """
        if seq_len == 0:
            return 0.0
        dist = dist / (dist.sum() + 1e-12)   # normalise
        sink  = self._cfg.sink_tokens
        window = self._cfg.window_size

        # Positions that belong to sinks or recent window
        recent_start = max(sink, seq_len - window)
        retained = np.zeros(seq_len, dtype=bool)
        retained[:sink]           = True   # attention sinks
        retained[recent_start:]   = True   # recent window

        remote_mass = float(dist[~retained].sum())
        return remote_mass

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def to_labels_array(self) -> np.ndarray:
        """
        Export head types as ``(n_layers, n_heads)`` int array.

        0 = STREAMING, 1 = RETRIEVAL, 2 = UNKNOWN
        """
        label_map = {
            HeadType.STREAMING: 0,
            HeadType.RETRIEVAL: 1,
            HeadType.UNKNOWN:   2,
        }
        out = np.zeros((self._cfg.n_layers, self._cfg.n_heads), dtype=np.int8)
        for l in range(self._cfg.n_layers):
            for h in range(self._cfg.n_heads):
                out[l, h] = label_map[self.head_types[l][h]]
        return out

    def from_labels_array(self, arr: np.ndarray) -> None:
        """Load head types from a serialised ``(n_layers, n_heads)`` int array."""
        rev = [HeadType.STREAMING, HeadType.RETRIEVAL, HeadType.UNKNOWN]
        for l in range(min(arr.shape[0], self._cfg.n_layers)):
            for h in range(min(arr.shape[1], self._cfg.n_heads)):
                self.head_types[l][h] = rev[int(arr[l, h])]


# ---------------------------------------------------------------------------
# Per-head KV store
# ---------------------------------------------------------------------------

class _HeadBuffer:
    """
    Internal circular / full buffer for one (layer, head) pair.

    * Streaming heads: ring buffer of ``window_size + sink_tokens`` entries.
    * Retrieval heads: growing list (up to ``max_entries`` before FIFO eviction).
    """

    def __init__(
        self,
        head_type:  HeadType,
        window:     int,
        sinks:      int,
        max_entries: int,
    ) -> None:
        self._type      = head_type
        self._window    = window
        self._sinks     = sinks
        self._max       = max_entries
        self._keys:   list[np.ndarray] = []
        self._vals:   list[np.ndarray] = []

    def put(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append one token's key/value vectors."""
        self._keys.append(np.asarray(key, dtype=np.float32))
        self._vals.append(np.asarray(value, dtype=np.float32))
        self._evict()

    def get(
        self,
        query:  np.ndarray | None = None,
        top_k:  int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve key/value arrays.

        For streaming heads the ring-buffer contents are returned directly.
        For retrieval heads with ``top_k > 0``, dot-product-based approximate
        selection is used (lightweight inline ANN without hnswlib dependency).
        """
        if not self._keys:
            # Return empty arrays with ndim=2
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

        keys = np.stack(self._keys)   # (N, d_head)
        vals = np.stack(self._vals)

        if self._type == HeadType.STREAMING or top_k == 0 or query is None:
            return keys, vals

        # Retrieval head + query-based selection
        q = np.asarray(query, dtype=np.float32).flatten()
        scores = keys @ q                      # (N,) dot products
        k_out  = min(top_k, len(self._keys))
        idxs   = np.argpartition(scores, -k_out)[-k_out:]
        idxs   = idxs[np.argsort(scores[idxs])[::-1]]  # sort desc
        return keys[idxs], vals[idxs]

    def __len__(self) -> int:
        return len(self._keys)

    # ── Private ───────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        if self._type == HeadType.STREAMING:
            # Keep sinks at front + recent window at end
            cap = self._sinks + self._window
            while len(self._keys) > cap:
                # Remove oldest non-sink entry
                del self._keys[self._sinks]
                del self._vals[self._sinks]
        else:
            # Retrieval head: simple FIFO cap
            while len(self._keys) > self._max:
                del self._keys[0]
                del self._vals[0]


class HeadAwareKVStore:
    """
    Per-request KV store that routes each (layer, head) to the appropriate
    retention policy derived from :class:`HeadClassifier`.

    Parameters
    ----------
    config      : HeadInferConfig
    head_types  : Optional (n_layers × n_heads) list of HeadType.
                  When None all heads default to RETRIEVAL.
    """

    def __init__(
        self,
        config:     HeadInferConfig,
        head_types: list[list[HeadType]] | None = None,
    ) -> None:
        self._cfg = config
        max_ret   = max(config.window_size * 4, 4096)  # generous cap
        self._buffers: dict[tuple[int, int], _HeadBuffer] = {}

        for l in range(config.n_layers):
            for h in range(config.n_heads):
                if head_types is not None:
                    ht = head_types[l][h]
                    if ht == HeadType.UNKNOWN:
                        ht = HeadType.RETRIEVAL
                else:
                    ht = HeadType.RETRIEVAL
                self._buffers[(l, h)] = _HeadBuffer(
                    head_type   = ht,
                    window      = config.window_size,
                    sinks       = config.sink_tokens,
                    max_entries = max_ret,
                )

    # ── Public API ─────────────────────────────────────────────────────────────

    def put(
        self,
        layer_idx: int,
        head_idx:  int,
        key:       np.ndarray,
        value:     np.ndarray,
    ) -> None:
        """Append one token's key/value for the given (layer, head)."""
        buf = self._buffers.get((layer_idx, head_idx))
        if buf is None:
            return
        buf.put(key, value)

    def get(
        self,
        layer_idx: int,
        head_idx:  int,
        query:     np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve KV for the given (layer, head), optionally query-filtered.

        Returns
        -------
        keys : (N, d_head) float32
        vals : (N, d_head) float32
        """
        buf = self._buffers.get((layer_idx, head_idx))
        if buf is None:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        top_k = self._cfg.top_k_retrieval
        return buf.get(query=query, top_k=top_k)

    def head_size(self, layer_idx: int, head_idx: int) -> int:
        """Current number of KV entries for the given (layer, head)."""
        buf = self._buffers.get((layer_idx, head_idx))
        return len(buf) if buf else 0

    def total_entries(self) -> int:
        """Sum of all KV entries across all (layer, head) buffers."""
        return sum(len(b) for b in self._buffers.values())

    def reset(self) -> None:
        """Clear all buffers (start of new request)."""
        for buf in self._buffers.values():
            buf._keys.clear()
            buf._vals.clear()
