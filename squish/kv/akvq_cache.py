"""squish/kv/akvq_cache.py

AKVQCache — Attention-aware KV Cache Quantization with partial outlier
protection (arXiv:2409.12012, 2024).

Reference
---------
"AKVQ: Attention-aware KV Cache Quantization with Partial Outlier Protection."
arXiv:2409.12012, 2024.

Algorithm
---------
AKVQ selects quantisation precision on a per-head basis according to the
head's accumulated attention score importance:

1. Observe attention scores over a calibration window.
2. Compute per-head importance = mean of top-k attention weights.
3. High-importance heads → INT4; low-importance heads → INT2.
4. A small fraction of "outlier" channels per head are stored in full FP32.

This simulation:
* Uses min-max quantisation for INT2/INT4.
* Protects ``outlier_ratio`` fraction of channels per head in FP32.
* Returns packed INT arrays alongside scale/zero-point/outlier metadata.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``high_precision_bits`` — bits assigned to high-importance heads (default INT4).
* ``low_precision_bits``  — bits assigned to low-importance heads (default INT2).
* ``importance_threshold`` — mean-top-k score above which a head is HIGH.
* ``outlier_ratio``        — fraction of channels protected in FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "AKVQConfig",
    "AKVQTensor",
    "AKVQCache",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class AKVQConfig:
    """Configuration for :class:`AKVQCache`.

    Attributes:
        high_precision_bits: Bits for high-importance heads (4 or 8).
        low_precision_bits: Bits for low-importance heads (2 or 4);
            must be < high_precision_bits.
        importance_threshold: Attention-score threshold to assign INT4 vs INT2.
        outlier_ratio: Fraction of channels stored as FP32 outliers (0.0–0.5).
        top_k_attn: Number of top attention weights used to compute importance.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    high_precision_bits: int = 4
    low_precision_bits: int = 2
    importance_threshold: float = 0.1
    outlier_ratio: float = 0.05
    top_k_attn: int = 8
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.high_precision_bits not in (4, 8):
            raise ValueError(
                f"high_precision_bits must be 4 or 8; got {self.high_precision_bits}"
            )
        if self.low_precision_bits not in (2, 4):
            raise ValueError(
                f"low_precision_bits must be 2 or 4; got {self.low_precision_bits}"
            )
        if self.low_precision_bits >= self.high_precision_bits:
            raise ValueError(
                f"low_precision_bits ({self.low_precision_bits}) must be < "
                f"high_precision_bits ({self.high_precision_bits})"
            )
        if not (0.0 <= self.outlier_ratio < 0.5):
            raise ValueError(f"outlier_ratio must be in [0, 0.5); got {self.outlier_ratio}")
        if self.top_k_attn < 1:
            raise ValueError(f"top_k_attn must be ≥ 1; got {self.top_k_attn}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Quantised tensor ──────────────────────────────────────────────────────────


class AKVQTensor:
    """Mixed-precision quantised KV tensor for one layer.

    Attributes:
        codes: List of per-head int8 arrays ``(S, head_dim_normal)``.
        scales: List of per-head float32 scales.
        zero_points: List of per-head float32 zero points.
        outlier_vals: List of per-head float32 arrays ``(S, n_outlier_channels)``.
        outlier_idx: List of per-head int64 arrays ``(n_outlier_channels,)``.
        bits_per_head: List of ints (2 or 4) — precision per head.
        original_shape: ``(n_heads, S, head_dim)`` original shape.
    """

    def __init__(
        self,
        codes: List[np.ndarray],
        scales: List[float],
        zero_points: List[float],
        outlier_vals: List[np.ndarray],
        outlier_idx: List[np.ndarray],
        bits_per_head: List[int],
        original_shape: Tuple[int, int, int],
    ) -> None:
        self.codes = codes
        self.scales = scales
        self.zero_points = zero_points
        self.outlier_vals = outlier_vals
        self.outlier_idx = outlier_idx
        self.bits_per_head = bits_per_head
        self.original_shape = original_shape


# ── Cache ─────────────────────────────────────────────────────────────────────


class AKVQCache:
    """Attention-aware mixed-precision KV cache quantizer.

    Example::

        cfg = AKVQConfig(high_precision_bits=4, low_precision_bits=2, n_heads=4, head_dim=8)
        cache = AKVQCache(cfg)

        attn_scores = np.random.rand(4, 16, 16).astype(np.float32)  # (H, T, S)
        cache.calibrate(attn_scores)

        K = np.random.randn(4, 16, 8).astype(np.float32)
        V = np.random.randn(4, 16, 8).astype(np.float32)
        cache.store(layer_id=0, K=K, V=V)
        K_back, V_back = cache.load(layer_id=0)
    """

    def __init__(self, config: Optional[AKVQConfig] = None) -> None:
        self.config = config or AKVQConfig()
        self._head_bits: Optional[List[int]] = None
        self._store: Dict[int, Tuple[AKVQTensor, AKVQTensor]] = {}

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, attn_scores: np.ndarray) -> List[int]:
        """Assign precision per head based on attention-score importance.

        Args:
            attn_scores: ``(n_heads, T, S)`` attention weight matrix.

        Returns:
            List of per-head bit widths: each is ``high_precision_bits`` or
            ``low_precision_bits``.
        """
        attn_scores = np.asarray(attn_scores, dtype=np.float32)
        H = attn_scores.shape[0]
        k = min(self.config.top_k_attn, attn_scores.shape[-1])
        # Mean of top-k attention weights per head
        top_k_vals = np.sort(attn_scores.reshape(H, -1), axis=-1)[:, -k:]
        importance = top_k_vals.mean(axis=-1)  # (H,)

        bits = [
            self.config.high_precision_bits
            if importance[h] >= self.config.importance_threshold
            else self.config.low_precision_bits
            for h in range(H)
        ]
        self._head_bits = bits
        return bits

    # ── Store / Load ──────────────────────────────────────────────────────────

    def store(
        self,
        layer_id: int,
        K: np.ndarray,
        V: np.ndarray,
    ) -> None:
        """Quantize and store K/V for a layer.

        Args:
            layer_id: Layer index.
            K: ``(n_heads, S, head_dim)`` key tensor.
            V: ``(n_heads, S, head_dim)`` value tensor.
        """
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        self._store[layer_id] = (
            self._quantize_kv(K),
            self._quantize_kv(V),
        )

    def load(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Dequantize and return K/V for a layer.

        Raises:
            KeyError: If layer_id not stored.
        """
        if layer_id not in self._store:
            raise KeyError(f"Layer {layer_id} not in AKVQCache.")
        K_qt, V_qt = self._store[layer_id]
        return self._dequantize_kv(K_qt), self._dequantize_kv(V_qt)

    def n_layers_cached(self) -> int:
        """Number of layers stored."""
        return len(self._store)

    def head_bits(self) -> Optional[List[int]]:
        """Per-head bit widths from last calibration call."""
        return self._head_bits

    def memory_bytes(self) -> int:
        """Total bytes used by all cached quantised tensors."""
        total = 0
        for K_qt, V_qt in self._store.values():
            for qt in (K_qt, V_qt):
                for c in qt.codes:
                    total += c.nbytes
                for ov in qt.outlier_vals:
                    total += ov.nbytes
                for oi in qt.outlier_idx:
                    total += oi.nbytes
        return total

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _quantize_kv(self, X: np.ndarray) -> AKVQTensor:
        H, S, d = X.shape
        cfg = self.config
        bits_per_head = (
            self._head_bits if self._head_bits and len(self._head_bits) == H
            else [cfg.high_precision_bits] * H
        )
        n_outlier = max(1, int(round(d * cfg.outlier_ratio)))
        codes_list: List[np.ndarray] = []
        scales_list: List[float] = []
        zp_list: List[float] = []
        outlier_vals_list: List[np.ndarray] = []
        outlier_idx_list: List[np.ndarray] = []

        for h in range(H):
            x_h = X[h]  # (S, d)
            bits = bits_per_head[h]
            levels = 2 ** bits

            # Identify outlier columns by max abs value
            col_max = np.abs(x_h).max(axis=0)  # (d,)
            out_idx = np.argsort(col_max)[::-1][:n_outlier].astype(np.int64)
            normal_idx = np.setdiff1d(np.arange(d), out_idx)

            x_normal = x_h[:, normal_idx]  # (S, d - n_outlier)
            x_outlier = x_h[:, out_idx]    # (S, n_outlier)

            # Min-max quantise normal columns
            mn = float(x_normal.min())
            mx = float(x_normal.max())
            scale = (mx - mn) / (levels - 1) + 1e-9
            zero_point = mn
            codes = np.clip(
                np.round((x_normal - zero_point) / scale), 0, levels - 1
            ).astype(np.int8)

            codes_list.append(codes)
            scales_list.append(scale)
            zp_list.append(zero_point)
            outlier_vals_list.append(x_outlier.astype(np.float32))
            outlier_idx_list.append(out_idx)

        return AKVQTensor(
            codes=codes_list,
            scales=scales_list,
            zero_points=zp_list,
            outlier_vals=outlier_vals_list,
            outlier_idx=outlier_idx_list,
            bits_per_head=bits_per_head,
            original_shape=(H, S, d),
        )

    def _dequantize_kv(self, qt: AKVQTensor) -> np.ndarray:
        H, S, d = qt.original_shape
        out = np.zeros((H, S, d), dtype=np.float32)
        for h in range(H):
            codes = qt.codes[h].astype(np.float32)
            scale = qt.scales[h]
            zp = qt.zero_points[h]
            out_idx = qt.outlier_idx[h]
            normal_idx = np.setdiff1d(np.arange(d), out_idx)
            out[h][:, normal_idx] = codes * scale + zp
            out[h][:, out_idx] = qt.outlier_vals[h]
        return out

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"AKVQCache(high={cfg.high_precision_bits}b, low={cfg.low_precision_bits}b, "
            f"outlier_ratio={cfg.outlier_ratio}, n_layers={self.n_layers_cached()})"
        )
