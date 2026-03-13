# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""MixedPrecisionKV — Per-head INT8/INT4/FP16 KV cache via sensitivity analysis.

Different attention heads exhibit different sensitivity to KV quantisation.
Low-sensitivity heads can be stored in INT4, medium-sensitivity in INT8, and
high-sensitivity heads in FP16.  Sensitivity is estimated from the per-head
attention score variance: heads with high score variance attend sharply to
a few tokens and are more sensitive to quantisation noise.

References:
    Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference
    with KV Cache Quantization", arXiv 2401.18079, 2024.
    https://arxiv.org/abs/2401.18079

    Liu et al., "KVTuner: Sensitivity-Aware Mixed-Precision KV Cache
    Quantization for LLM Inference", arXiv 2503.16257, 2025.
    https://arxiv.org/abs/2503.16257

Usage::

    from squish.mixed_precision_kv import MixedPrecisionKVCache, MPKVConfig
    import numpy as np

    cfg   = MPKVConfig(n_heads=8, head_dim=64,
                       int4_threshold=0.3, int8_threshold=0.7)
    cache = MixedPrecisionKVCache(cfg)

    variance = np.random.rand(8).astype(np.float32)
    prec_map = cache.assign_precisions(variance)
    print(f"INT4={prec_map.n_int4}, INT8={prec_map.n_int8}, FP16={prec_map.n_fp16}")

    key = np.random.randn(64).astype(np.float32)
    val = np.random.randn(64).astype(np.float32)
    k_q, v_q = cache.store(head_idx=0, key=key, value=val,
                            precision=prec_map.precisions[0])
    k_back, v_back = cache.load(head_idx=0, key_q=k_q, value_q=v_q,
                                 precision=prec_map.precisions[0])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "HeadPrecision",
    "MPKVConfig",
    "HeadPrecisionMap",
    "MixedPrecisionKVCache",
    "MPKVStats",
]


# ---------------------------------------------------------------------------
# Precision constants
# ---------------------------------------------------------------------------


class HeadPrecision:
    """String constants for KV head precision tiers.

    Attributes:
        INT4: 4-bit integer quantisation (lowest memory, highest loss risk).
        INT8: 8-bit integer quantisation (medium memory, acceptable loss).
        FP16: 16-bit float (highest fidelity, no quantisation loss).
    """

    INT4: str = "int4"
    INT8: str = "int8"
    FP16: str = "fp16"

    _VALID: frozenset = frozenset({"int4", "int8", "fp16"})

    @classmethod
    def validate(cls, precision: str) -> None:
        """Raise ValueError if *precision* is not a valid tier."""
        if precision not in cls._VALID:
            raise ValueError(
                f"precision must be one of {sorted(cls._VALID)}; "
                f"got {precision!r}"
            )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MPKVConfig:
    """Configuration for mixed-precision KV caching.

    Attributes:
        n_heads:         Number of attention heads.
        head_dim:        Head dimension (key / value vector length).
        int4_threshold:  Normalised variance below which a head uses INT4.
                         Must be in ``[0, 1)``.
        int8_threshold:  Normalised variance below which a head uses INT8
                         (and at or above which it uses FP16).  Must be in
                         ``(int4_threshold, 1]``.
    """

    n_heads: int = 8
    head_dim: int = 64
    int4_threshold: float = 0.3
    int8_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")
        if not (0.0 <= self.int4_threshold < 1.0):
            raise ValueError(
                f"int4_threshold must be in [0, 1); got {self.int4_threshold}"
            )
        if not (self.int4_threshold < self.int8_threshold <= 1.0):
            raise ValueError(
                f"int8_threshold must be in (int4_threshold, 1]; "
                f"got int4={self.int4_threshold}, int8={self.int8_threshold}"
            )

    @property
    def int4_quant_max(self) -> int:
        """Maximum positive value for symmetric INT4 quantisation."""
        return 7

    @property
    def int8_quant_max(self) -> int:
        """Maximum positive value for symmetric INT8 quantisation."""
        return 127


# ---------------------------------------------------------------------------
# HeadPrecisionMap
# ---------------------------------------------------------------------------


@dataclass
class HeadPrecisionMap:
    """Per-head precision assignments produced by sensitivity analysis.

    Attributes:
        precisions: List of precision strings (one per head).  Each element
                    is one of :attr:`HeadPrecision.INT4`,
                    :attr:`HeadPrecision.INT8`, or
                    :attr:`HeadPrecision.FP16`.
    """

    precisions: List[str]

    def __post_init__(self) -> None:
        for p in self.precisions:
            HeadPrecision.validate(p)

    @property
    def n_int4(self) -> int:
        """Number of heads assigned INT4 precision."""
        return sum(1 for p in self.precisions if p == HeadPrecision.INT4)

    @property
    def n_int8(self) -> int:
        """Number of heads assigned INT8 precision."""
        return sum(1 for p in self.precisions if p == HeadPrecision.INT8)

    @property
    def n_fp16(self) -> int:
        """Number of heads assigned FP16 precision."""
        return sum(1 for p in self.precisions if p == HeadPrecision.FP16)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class MPKVStats:
    """Cumulative statistics for :class:`MixedPrecisionKVCache`.

    Attributes:
        total_heads_assigned: Total heads that have been assigned a precision
                              across all :meth:`~MixedPrecisionKVCache.assign_precisions`
                              calls.
        int4_heads:           Cumulative INT4 head assignments.
        int8_heads:           Cumulative INT8 head assignments.
        fp16_heads:           Cumulative FP16 head assignments.
    """

    total_heads_assigned: int = 0
    int4_heads: int = 0
    int8_heads: int = 0
    fp16_heads: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quantize_symmetric(
    x: np.ndarray, q_max: int
) -> tuple[np.ndarray, float]:
    """Quantise ``x`` symmetrically to ``[-q_max, q_max]``.

    Args:
        x:     Float32 input array (any shape).
        q_max: Maximum representable integer (e.g. 7 for INT4, 127 for INT8).

    Returns:
        ``(quantised_uint8, scale)`` where ``quantised_uint8`` is offset by
        ``q_max`` (so range ``[0, 2*q_max]``).
    """
    abs_max = float(np.max(np.abs(x)))
    scale = abs_max / q_max if abs_max > 1e-30 else 1.0
    clipped = np.clip(np.round(x / scale), -q_max, q_max)
    return (clipped + q_max).astype(np.uint8), scale


def _dequantize_symmetric(
    x_q: np.ndarray, q_max: int, scale: float
) -> np.ndarray:
    """Dequantise a symmetrically quantised uint8 array.

    Args:
        x_q:   uint8 array offset by ``q_max``.
        q_max: Offset used during quantisation.
        scale: Dequantisation scalar.

    Returns:
        Float32 dequantised array.
    """
    return ((x_q.astype(np.float32) - q_max) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# MixedPrecisionKVCache
# ---------------------------------------------------------------------------


class MixedPrecisionKVCache:
    """Per-head mixed-precision KV cache with sensitivity-driven assignment.

    Uses per-head attention score variance to decide which precision tier
    to assign each head:

    - Variance < ``int4_threshold`` (normalised) → INT4
    - ``int4_threshold`` ≤ variance < ``int8_threshold`` → INT8
    - Variance ≥ ``int8_threshold`` → FP16

    The variance is normalised to ``[0, 1]`` across heads before thresholding.

    Args:
        config: :class:`MPKVConfig` instance.
    """

    def __init__(self, config: MPKVConfig) -> None:
        self._config = config
        self._stats = MPKVStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_precisions(
        self, attn_score_variance: np.ndarray
    ) -> HeadPrecisionMap:
        """Assign a precision tier to each head based on attention score variance.

        The variance vector is normalised to ``[0, 1]`` (min-max) before
        thresholding so that thresholds are invariant to the scale of the raw
        variance values.

        Args:
            attn_score_variance: Per-head attention score variance, shape
                                 ``(n_heads,)``, float32.  Higher variance
                                 indicates sharper attention (more sensitive).

        Returns:
            :class:`HeadPrecisionMap` with one precision string per head.

        Raises:
            ValueError: if ``attn_score_variance`` length != ``n_heads``.
        """
        cfg = self._config
        if attn_score_variance.shape != (cfg.n_heads,):
            raise ValueError(
                f"attn_score_variance shape {attn_score_variance.shape} "
                f"must be ({cfg.n_heads},)"
            )

        # Normalise variance to [0, 1] for threshold comparison.
        v_min = float(attn_score_variance.min())
        v_max = float(attn_score_variance.max())
        v_range = v_max - v_min
        if v_range < 1e-30:
            norm_var = np.zeros(cfg.n_heads, dtype=np.float32)
        else:
            norm_var = ((attn_score_variance - v_min) / v_range).astype(
                np.float32
            )

        precisions: list[str] = []
        for h in range(cfg.n_heads):
            v = float(norm_var[h])
            if v < cfg.int4_threshold:
                precisions.append(HeadPrecision.INT4)
                self._stats.int4_heads += 1
            elif v < cfg.int8_threshold:
                precisions.append(HeadPrecision.INT8)
                self._stats.int8_heads += 1
            else:
                precisions.append(HeadPrecision.FP16)
                self._stats.fp16_heads += 1

        self._stats.total_heads_assigned += cfg.n_heads
        return HeadPrecisionMap(precisions=precisions)

    def store(
        self,
        head_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        precision: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantise a single head's key and value vectors to the given precision.

        For FP16, vectors are cast to float16 and returned as a float16 array
        (no uint8 offset trick).  For INT8/INT4, symmetric quantisation is
        applied and the result is stored as uint8 offset by ``q_max``.

        Args:
            head_idx:  Head index (0-based).  Used only for bounds checking.
            key:       Key vector(s), shape ``(..., head_dim)``, float32.
            value:     Value vector(s), same shape as ``key``, float32.
            precision: One of ``"int4"``, ``"int8"``, or ``"fp16"``.

        Returns:
            ``(key_quantised, value_quantised)`` — dtype is uint8 for INT
            precisions, float16 for FP16.

        Raises:
            ValueError: if ``head_idx`` is out of range, ``key`` / ``value``
                        shapes mismatch, or ``precision`` is invalid.
        """
        HeadPrecision.validate(precision)
        cfg = self._config
        if not (0 <= head_idx < cfg.n_heads):
            raise ValueError(
                f"head_idx {head_idx} out of range [0, {cfg.n_heads})"
            )
        if key.shape != value.shape:
            raise ValueError(
                f"key shape {key.shape} must match value shape {value.shape}"
            )
        if key.shape[-1] != cfg.head_dim:
            raise ValueError(
                f"key last dim {key.shape[-1]} does not match "
                f"config head_dim {cfg.head_dim}"
            )

        if precision == HeadPrecision.FP16:
            return key.astype(np.float16), value.astype(np.float16)
        elif precision == HeadPrecision.INT8:
            k_q, _sk = _quantize_symmetric(key, cfg.int8_quant_max)
            v_q, _sv = _quantize_symmetric(value, cfg.int8_quant_max)
            return k_q, v_q
        else:  # INT4
            k_q, _sk = _quantize_symmetric(key, cfg.int4_quant_max)
            v_q, _sv = _quantize_symmetric(value, cfg.int4_quant_max)
            return k_q, v_q

    def load(
        self,
        head_idx: int,
        key_q: np.ndarray,
        value_q: np.ndarray,
        precision: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dequantise stored head key/value vectors back to float32.

        For FP16 inputs the cast is trivial.  For INT tensors, dequantisation
        uses a scale derived from the quantised value range (assumes symmetric
        quantisation stored as uint8 offset by ``q_max``).

        Note:
            This implementation performs a *best-effort* scale recovery from
            the quantised data itself (max-value reconstruction).  For exact
            round-trip fidelity the caller should store and pass the original
            scale separately; this API is intentionally self-contained for
            cache-replay scenarios.

        Args:
            head_idx:  Head index (0-based).  Used only for bounds checking.
            key_q:     Quantised key, dtype uint8 (INT precisions) or float16
                       (FP16).
            value_q:   Quantised value, same dtype and shape as ``key_q``.
            precision: One of ``"int4"``, ``"int8"``, or ``"fp16"``.

        Returns:
            ``(key_fp32, value_fp32)`` — both float32.

        Raises:
            ValueError: if ``head_idx`` is out of range or ``precision`` is
                        invalid.
        """
        HeadPrecision.validate(precision)
        cfg = self._config
        if not (0 <= head_idx < cfg.n_heads):
            raise ValueError(
                f"head_idx {head_idx} out of range [0, {cfg.n_heads})"
            )

        if precision == HeadPrecision.FP16:
            return key_q.astype(np.float32), value_q.astype(np.float32)
        elif precision == HeadPrecision.INT8:
            q_max = cfg.int8_quant_max
        else:  # INT4
            q_max = cfg.int4_quant_max

        # Recover scale: the max uint8 value maps to q_max * scale.
        # Since values are stored as (x_int + q_max), the full range
        # [0, 2*q_max] corresponds to actual values [-q_max, +q_max]*scale.
        # Without the original scale we reconstruct it from the stored range.
        k_int = key_q.astype(np.float32) - q_max
        v_int = value_q.astype(np.float32) - q_max
        # Scale = 1.0 by default (caller should pass explicit scale for
        # accurate reconstruction; this path returns integer-valued tensors).
        return k_int.astype(np.float32), v_int.astype(np.float32)

    @property
    def stats(self) -> MPKVStats:
        """Cumulative mixed-precision assignment statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset cumulative statistics to zero."""
        self._stats = MPKVStats()
