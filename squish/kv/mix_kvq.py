"""
squish/mix_kvq.py

MixKVQ — Query-Aware Per-Channel Mixed-Precision KV Quantization.

Based on:
  "MixKVQ: Mixed-Precision KV Cache Quantization with Query-Aware Channel
  Selection for Reasoning LLMs"
  arXiv:2512.19206 (Dec 22, 2025)

Evaluated on Qwen-32B; achieves 66.04% vs BF16 baseline 67.84% (−1.8% gap)
at ~2.5-bit average vs RotateKV 4-bit (−3.33%) and KIVI 2-bit (−8.95%).

Problem
-------
Key cache quantization error comes from two *independent* factors that prior
methods conflate:

  1. **Intrinsic difficulty**: some Key channels have outlier distributions
     regardless of the query — they are structurally hard to quantize.

  2. **Query relevance**: some channels carry information important for the
     current query's attention pattern — regardless of their quantizability.

A channel that is *both* relevant AND hard should get FP16.
A channel that is *neither* should get INT2.
Channels in between get INT4.

MixKVQ Algorithm (per-query, per-layer)
-----------------------------------------
For each Key head at layer l, query step t:
  1. Compute channel importance from query-key inner products:
     ``importance[c] = |Q_t · K[:, c]| / Z``  (Z = normalisation)
  2. Compute channel difficulty from outlier magnitude:
     ``difficulty[c] = |K[:, c]|.max() / |K[:, c]|.mean()``
  3. Score = importance × difficulty
  4. Assign bits: top-α → FP16, bottom-β → INT2, middle → INT4

Provides
--------
  MixKVQConfig        — configuration parameters
  ChannelScorer       — compute per-channel importance × difficulty scores
  MixKVQQuantizer     — mixed-precision quantize/dequantize with bit-map
  MixKVQStats         — tracking statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MixKVQConfig",
    "ChannelScorer",
    "MixKVQQuantizer",
    "MixKVQStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MixKVQConfig:
    """Configuration for MixKVQ per-channel mixed-precision quantization.

    Parameters
    ----------
    fp16_fraction:
        Fraction of channels assigned FP16 (highest quality tier).
        Default 0.1 (10% of channels).
    int2_fraction:
        Fraction of channels assigned INT2 (maximum compression).
        Default 0.5 (50% of channels).  Remaining channels get INT4.
    importance_weight:
        Weight of query-relevance (importance) relative to difficulty in the
        combined score.  Default 0.5 (equal weight).
    group_size:
        INT4/INT2 quantization group size (default 64).
    calibration_history:
        Number of recent Key vectors tracked for difficulty estimation when
        no explicit Key history is available.  Default 128.
    """

    fp16_fraction:         float = 0.10
    int2_fraction:         float = 0.50
    importance_weight:     float = 0.50
    group_size:            int   = 64
    calibration_history:   int   = 128

    def __post_init__(self) -> None:
        if not 0.0 <= self.fp16_fraction <= 1.0:
            raise ValueError("fp16_fraction must be in [0, 1]")
        if not 0.0 <= self.int2_fraction <= 1.0:
            raise ValueError("int2_fraction must be in [0, 1]")
        if self.fp16_fraction + self.int2_fraction > 1.0:
            raise ValueError("fp16_fraction + int2_fraction must be <= 1.0")
        if not 0.0 <= self.importance_weight <= 1.0:
            raise ValueError("importance_weight must be in [0, 1]")


# ---------------------------------------------------------------------------
# Channel Scorer
# ---------------------------------------------------------------------------

class ChannelScorer:
    """Compute per-channel mixed-precision scores for Key cache vectors.

    Maintains a rolling history of Key vectors to estimate intrinsic channel
    difficulty (outlier magnitude ratio).  Per-query importance is computed
    on-the-fly from the current query vector.

    Parameters
    ----------
    n_channels:
        Number of channels (head_dim) per head.
    config:
        MixKVQ configuration.
    """

    def __init__(
        self,
        n_channels: int,
        config: MixKVQConfig | None = None,
    ) -> None:
        self._n     = n_channels
        self._cfg   = config or MixKVQConfig()
        # Rolling Key history for difficulty estimation: shape (history, n_channels)
        self._hist  = np.zeros(
            (self._cfg.calibration_history, n_channels), dtype=np.float32
        )
        self._pos   = 0
        self._filled = False

    @property
    def n_channels(self) -> int:
        return self._n

    def record(self, key_vector: np.ndarray) -> None:
        """Record a Key vector for difficulty calibration.

        Parameters
        ----------
        key_vector:
            Float32 array of shape ``(n_channels,)``.
        """
        self._hist[self._pos] = key_vector.astype(np.float32)[:self._n]
        self._pos = (self._pos + 1) % self._cfg.calibration_history
        if self._pos == 0:
            self._filled = True

    def difficulty(self) -> np.ndarray:
        """Compute per-channel difficulty scores from Key history.

        Returns
        -------
        np.ndarray of shape ``(n_channels,)`` in [0, 1].
        Scores are 0 before any history is accumulated.
        """
        if not self._filled and self._pos == 0:
            return np.zeros(self._n, dtype=np.float32)

        hist = self._hist[:self._pos] if not self._filled else self._hist
        max_abs = np.abs(hist).max(axis=0).clip(min=1e-7)
        mean_abs = np.abs(hist).mean(axis=0).clip(min=1e-7)
        diff = max_abs / mean_abs        # outlier ratio
        dmax = diff.max()
        return diff / dmax if dmax > 0 else diff

    def importance(self, query: np.ndarray, key_matrix: np.ndarray) -> np.ndarray:
        """Compute per-channel query-relevance importance scores.

        Parameters
        ----------
        query:
            Float32 array of shape ``(n_channels,)`` — the current query vector.
        key_matrix:
            Float32 array of shape ``(seq_len, n_channels)`` — accumulated Key cache.

        Returns
        -------
        np.ndarray of shape ``(n_channels,)`` in [0, 1].
        """
        if key_matrix.shape[0] == 0:
            return np.zeros(self._n, dtype=np.float32)

        q = query.astype(np.float32)[:self._n]
        k = key_matrix.astype(np.float32)[:, :self._n]
        # Inner product of query with each key, summed over sequence → channel importance
        scores = np.abs(k * q[None, :]).mean(axis=0)
        smax = scores.max()
        return scores / smax if smax > 0 else scores

    def score(
        self,
        query: np.ndarray,
        key_matrix: np.ndarray,
    ) -> np.ndarray:
        """Combined importance × difficulty score per channel.

        Parameters
        ----------
        query:
            Current query vector, shape ``(n_channels,)``.
        key_matrix:
            Accumulated Key cache, shape ``(seq_len, n_channels)``.

        Returns
        -------
        np.ndarray of shape ``(n_channels,)`` in [0, 1].
        """
        w = self._cfg.importance_weight
        imp  = self.importance(query, key_matrix)
        diff = self.difficulty()
        combined = w * imp + (1.0 - w) * diff
        cmax = combined.max()
        return combined / cmax if cmax > 0 else combined

    def assign_bits(
        self,
        query: np.ndarray,
        key_matrix: np.ndarray,
    ) -> np.ndarray:
        """Return per-channel bit assignment array.

        Returns
        -------
        np.ndarray of shape ``(n_channels,)`` with values in {2, 4, 16}.
        """
        scores   = self.score(query, key_matrix)
        n        = len(scores)
        cfg      = self._cfg
        n_fp16   = max(1, int(np.round(cfg.fp16_fraction * n)))
        n_int2   = max(1, int(np.round(cfg.int2_fraction * n)))

        sorted_idx = np.argsort(-scores)  # descending
        bits = np.full(n, 4, dtype=np.int32)     # default INT4
        bits[sorted_idx[:n_fp16]]          = 16  # top: FP16
        bits[sorted_idx[n - n_int2:]]      = 2   # bottom: INT2
        return bits


# ---------------------------------------------------------------------------
# MixKVQ Quantizer
# ---------------------------------------------------------------------------

class MixKVQQuantizer:
    """Mixed-precision quantize / dequantize for Key cache vectors.

    Quantizes each channel group according to its assigned bit-width (from
    `ChannelScorer.assign_bits()`).  Three levels: FP16, INT4, INT2.

    Parameters
    ----------
    config:
        MixKVQ configuration.
    """

    def __init__(self, config: MixKVQConfig | None = None) -> None:
        self._cfg = config or MixKVQConfig()

    def quantize(
        self,
        key_vector: np.ndarray,
        bit_map: np.ndarray,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """Quantize a single Key vector with per-channel bit assignment.

        Parameters
        ----------
        key_vector:
            Float32 array of shape ``(n_channels,)``.
        bit_map:
            Int array of shape ``(n_channels,)`` with values in ``{2, 4, 16}``.

        Returns
        -------
        segments : list of np.ndarray  — one per contiguous same-bits segment
        scales   : np.ndarray          — per-group scale (matches segment groups)
        bit_map  : np.ndarray          — stored bit_map for decode
        """
        kv = key_vector.astype(np.float32)
        n  = len(kv)
        gs = self._cfg.group_size

        segments: list[np.ndarray] = []
        scales_list: list[np.ndarray] = []

        # Segment by bit assignment, quantize each segment
        i = 0
        while i < n:
            bits = int(bit_map[i])
            j = i + 1
            while j < n and bit_map[j] == bits:
                j += 1
            chunk = kv[i:j]

            if bits == 16:
                segments.append(chunk.astype(np.float16))
                scales_list.append(np.ones(1, dtype=np.float32))
            else:
                levels = (1 << bits) - 1
                n_chunk = len(chunk)
                n_groups = max(1, (n_chunk + gs - 1) // gs)
                padded = np.zeros(n_groups * gs, dtype=np.float32)
                padded[:n_chunk] = chunk
                groups  = padded.reshape(n_groups, gs)
                vmax    = np.abs(groups).max(axis=1, keepdims=True).clip(min=1e-7)
                scale   = vmax / max(levels // 2, 1)
                q       = np.round(groups / scale).clip(
                    -(levels // 2), levels // 2
                ).astype(np.int8)
                segments.append(q.flatten()[:n_chunk])
                scales_list.append(scale.flatten())

            i = j

        return segments, np.concatenate(scales_list), bit_map.copy()

    def dequantize(
        self,
        segments: list,
        scales: np.ndarray,
        bit_map: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct a float32 Key vector from segments.

        Parameters
        ----------
        segments : list as returned by ``quantize``
        scales   : concatenated scales array
        bit_map  : channel bit-width assignment

        Returns
        -------
        np.ndarray of shape ``(n_channels,)`` dtype float32.
        """
        n   = len(bit_map)
        gs  = self._cfg.group_size
        out = np.empty(n, dtype=np.float32)

        seg_idx   = 0
        scale_pos = 0
        i         = 0

        for seg_arr in segments:
            bits = int(bit_map[i])
            seg_len = len(seg_arr)

            if bits == 16:
                out[i:i + seg_len] = seg_arr.astype(np.float32)
                scale_pos += 1
            else:
                n_groups = max(1, (seg_len + gs - 1) // gs)
                seg_scales = scales[scale_pos:scale_pos + n_groups]
                padded = np.zeros(n_groups * gs, dtype=np.float32)
                padded[:seg_len] = seg_arr.astype(np.float32)
                groups = padded.reshape(n_groups, gs)
                dq = (groups * seg_scales.reshape(-1, 1)).flatten()[:seg_len]
                out[i:i + seg_len] = dq
                scale_pos += n_groups

            i       += seg_len
            seg_idx += 1

        return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class MixKVQStats:
    """Track per-channel precision distribution over requests.

    Attributes
    ----------
    total_quantized_channels  : total channel×token quantizations
    fp16_count                : channels quantized at FP16
    int4_count                : channels quantized at INT4
    int2_count                : channels quantized at INT2
    """

    total_quantized_channels: int = 0
    fp16_count:               int = 0
    int4_count:               int = 0
    int2_count:               int = 0

    def record(self, bit_map: np.ndarray) -> None:
        self.total_quantized_channels += len(bit_map)
        self.fp16_count += int((bit_map == 16).sum())
        self.int4_count += int((bit_map == 4).sum())
        self.int2_count += int((bit_map == 2).sum())

    @property
    def avg_bits(self) -> float:
        total = self.total_quantized_channels
        if total == 0:
            return 0.0
        return (16 * self.fp16_count + 4 * self.int4_count + 2 * self.int2_count) / total

    def reset(self) -> None:
        self.total_quantized_channels = 0
        self.fp16_count = 0
        self.int4_count = 0
        self.int2_count = 0
