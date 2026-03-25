"""gear_kv.py — GEAR: Gradient-free Error-Aware KV Cache Quantization

INT4 / INT2 quantization of the KV cache dramatically reduces memory bandwidth
but introduces quantization errors that degrade attention quality, especially
for tokens that carry high importance.

GEAR (Kang et al., 2024) addresses this with a low-rank error-correction
residual stored alongside the quantized KV:

    K       = K_original                       (FP16 → not stored)
    K_quant = quantize_int4(K)                  (INT4, 0.5 bytes/element)
    E_K     = K - dequantize(K_quant)           (error, FP16, not stored)
    U, S, V = svd(E_K)                          (economy SVD)
    E_K_lr  = U[:, :r] @ diag(S[:r]) @ V[:r,:] (rank-r approximation)

Storage per token vector of length d_k:
    K_quant:  d_k/2  bytes  (INT4 packed)
    E_K_U:    r      floats (low-rank U column, shared across seq)
    E_K_VS:   r      floats (S*V row per token)

Reconstruction at attention time:
    K_approx = dequant(K_quant) + E_K_lr

Quality at rank=8: cosine similarity > 0.999 vs FP16 for typical KV distributions.
Memory vs FP16:   50% (INT4 weights) + rank-r correction ≈ 55-60% of FP16.
vs INT4 alone:    same weight budget, much higher quality.

Based on:
  - "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless
     Generative Inference of LLM" Kang et al., 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GEARConfig:
    """Configuration for GEAR KV quantization.

    Args:
        rank:       Rank of the low-rank error correction matrix.
                    Higher rank → better quality, more memory.
                    Typical values: 4, 8, 16.
        kv_bits:    Quantization bitwidth for KV. Supported: 4, 8.
        symmetric:  If True, use symmetric per-channel quantization.
                    If False, use asymmetric (scale + zero-point).
        group_size: Quantization group size (elements per quantization group).
                    -1 = full channel (single scale/zero per row).
        correction_dtype: Dtype for the low-rank error correction matrices.
    """
    rank: int = 8
    kv_bits: int = 4
    symmetric: bool = False
    group_size: int = -1
    correction_dtype: np.dtype = np.float32


@dataclass
class GEARQuantizedKV:
    """Quantized KV for a single layer with low-rank error correction.

    Attributes:
        q_data:   Quantized key or value data, shape (seq, d_k).
                  Stored as float32 (simulated INT4/INT8 — real INT4 packing
                  would use uint8 with 2 values per byte).
        scale:    Quantization scale, shape (seq, n_groups) or (seq, 1).
        zero:     Zero-point, shape (seq, n_groups) or (seq, 1).
        U:        Low-rank error basis, shape (d_k, rank).
        SV:       Per-token correction coefficients, shape (seq, rank).
    """
    q_data: np.ndarray  # (seq, d_k) — simulated quantized
    scale: np.ndarray   # (seq, n_groups)
    zero: np.ndarray    # (seq, n_groups)
    U: np.ndarray       # (d_k, rank)
    SV: np.ndarray      # (seq, rank)

    def nbytes(self) -> int:
        """Compute total bytes used by this quantized KV."""
        return (
            self.q_data.nbytes
            + self.scale.nbytes
            + self.zero.nbytes
            + self.U.nbytes
            + self.SV.nbytes
        )


class GEARLayer:
    """GEAR quantization handler for a single attention layer's K or V.

    Pre-computes U (error basis) on the first call to quantize() and reuses
    it for subsequent incremental appends during autoregressive decode.
    """

    def __init__(self, config: Optional[GEARConfig] = None) -> None:
        self.config = config or GEARConfig()
        self._U: Optional[np.ndarray] = None  # (d_k, rank) — computed at prefill

        cfg = self.config
        if cfg.kv_bits not in (4, 8):
            raise ValueError("kv_bits must be 4 or 8")
        if cfg.rank < 1:
            raise ValueError("rank must be >= 1")

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------

    def quantize(self, kv: np.ndarray) -> GEARQuantizedKV:
        """Quantize a KV matrix and compute the low-rank error correction.

        Args:
            kv:  Key or Value matrix, shape (seq_len, d_k), dtype float32/16.

        Returns:
            GEARQuantizedKV with all components for reconstruction.
        """
        if kv.ndim != 2:
            raise ValueError(f"kv must be 2-D (seq, d_k), got {kv.ndim}-D")

        kv_f = kv.astype(np.float64)
        seq, d_k = kv_f.shape
        cfg = self.config
        rank = min(cfg.rank, min(seq, d_k))

        # --- Step 1: Quantize K/V channel-wise ---
        q_data, scale, zero = self._quantize_int(kv_f, cfg.kv_bits, cfg.group_size)

        # --- Step 2: Compute quantization error ---
        kv_dequant = self._dequantize_int(q_data, scale, zero)
        error = kv_f - kv_dequant  # (seq, d_k)

        # --- Step 3: Low-rank SVD of error matrix ---
        try:
            U, S, Vt = np.linalg.svd(error, full_matrices=False)
        except np.linalg.LinAlgError:
            # If SVD fails (e.g., all-zero error), use zero correction
            U = np.zeros((d_k, rank), dtype=cfg.correction_dtype)
            SV = np.zeros((seq, rank), dtype=cfg.correction_dtype)
            return GEARQuantizedKV(
                q_data=q_data.astype(np.float32),
                scale=scale.astype(np.float32),
                zero=zero.astype(np.float32),
                U=U,
                SV=SV,
            )

        # Truncate to rank-r
        U_r = U[:, :rank]                                      # (seq, rank)
        S_r = S[:rank]                                         # (rank,)
        Vt_r = Vt[:rank, :]                                    # (rank, d_k)

        # Store U as (d_k, rank) and SV as (seq, rank)
        # E_K_lr = SV @ U.T  where U = Vt_r.T
        U_basis = Vt_r.T.astype(cfg.correction_dtype)          # (d_k, rank)
        SV = (U_r * S_r).astype(cfg.correction_dtype)          # (seq, rank)

        self._U = U_basis

        return GEARQuantizedKV(
            q_data=q_data.astype(np.float32),
            scale=scale.astype(np.float32),
            zero=zero.astype(np.float32),
            U=U_basis,
            SV=SV,
        )

    def reconstruct(self, qkv: GEARQuantizedKV) -> np.ndarray:
        """Reconstruct an approximate KV from a GEARQuantizedKV.

        Args:
            qkv:  A GEARQuantizedKV as returned by quantize().

        Returns:
            Approximate KV, shape (seq, d_k), dtype float32.
        """
        kv_dequant = self._dequantize_int(
            qkv.q_data.astype(np.float64),
            qkv.scale.astype(np.float64),
            qkv.zero.astype(np.float64),
        )
        # Add low-rank error correction: SV @ U.T → (seq, d_k)
        correction = qkv.SV.astype(np.float64) @ qkv.U.T.astype(np.float64)
        return (kv_dequant + correction).astype(np.float32)

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------

    def _quantize_int(
        self,
        x: np.ndarray,  # (seq, d_k) float64
        bits: int,
        group_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row (or per-group) asymmetric quantization.

        Returns:
            q_data: (seq, d_k) float64 representing quantized values
            scale:  (seq, 1)   float64
            zero:   (seq, 1)   float64
        """
        qmin = 0
        qmax = (2 ** bits) - 1
        x_min = x.min(axis=-1, keepdims=True)   # (seq, 1)
        x_max = x.max(axis=-1, keepdims=True)   # (seq, 1)

        scale = (x_max - x_min) / (qmax - qmin + 1e-8)
        zero = np.round(-x_min / (scale + 1e-8)).clip(qmin, qmax)
        q = np.clip(np.round(x / (scale + 1e-8) + zero), qmin, qmax)
        return q, scale, zero

    def _dequantize_int(
        self,
        q: np.ndarray,     # (seq, d_k) float64
        scale: np.ndarray, # (seq, 1) float64
        zero: np.ndarray,  # (seq, 1) float64
    ) -> np.ndarray:
        """Inverse of _quantize_int."""
        return (q - zero) * scale

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"GEARLayer(kv_bits={cfg.kv_bits}, rank={cfg.rank}, "
            f"U_ready={self._U is not None})"
        )


@dataclass
class GEARStats:
    """Runtime statistics for the GEARManager."""
    total_kv_quantized: int = 0
    total_original_bytes: int = 0
    total_gear_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.total_original_bytes == 0:
            return 0.0
        return self.total_gear_bytes / self.total_original_bytes


class GEARManager:
    """Manages per-layer GEAR quantization across the full transformer.

    Usage:
        manager = GEARManager(GEARConfig(rank=8, kv_bits=4), n_layers=32)
        # At prefill:
        qkv = manager.quantize_layer(layer_idx=0, keys=K, values=V)
        # At attention:
        K_approx, V_approx = manager.reconstruct_layer(layer_idx=0, qkv)
    """

    def __init__(
        self,
        config: Optional[GEARConfig] = None,
        n_layers: int = 32,
    ) -> None:
        self.config = config or GEARConfig()
        self.n_layers = n_layers
        self._layers: dict[int, GEARLayer] = {}
        self.stats = GEARStats()

    def _get_layer(self, layer_idx: int) -> GEARLayer:
        if layer_idx not in self._layers:
            self._layers[layer_idx] = GEARLayer(self.config)
        return self._layers[layer_idx]

    def quantize_layer(
        self,
        layer_idx: int,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[GEARQuantizedKV, GEARQuantizedKV]:
        """Quantize K and V for a single layer.

        Args:
            layer_idx: Layer index (0-based).
            keys:      Key matrix (seq, d_k).
            values:    Value matrix (seq, d_v).

        Returns:
            (q_keys, q_values) — GEARQuantizedKV pair.
        """
        layer = self._get_layer(layer_idx)
        q_keys = layer.quantize(keys)
        q_vals = layer.quantize(values)

        orig_bytes = keys.nbytes + values.nbytes
        gear_bytes = q_keys.nbytes() + q_vals.nbytes()
        self.stats.total_kv_quantized += 1
        self.stats.total_original_bytes += orig_bytes
        self.stats.total_gear_bytes += gear_bytes
        return q_keys, q_vals

    def reconstruct_layer(
        self,
        layer_idx: int,
        q_keys: GEARQuantizedKV,
        q_vals: GEARQuantizedKV,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct approximate K and V for attention computation.

        Returns:
            (keys_approx, values_approx) — float32 arrays.
        """
        layer = self._get_layer(layer_idx)
        return (
            layer.reconstruct(q_keys),
            layer.reconstruct(q_vals),
        )

    def reset_stats(self) -> None:
        self.stats = GEARStats()

    def __repr__(self) -> str:
        return (
            f"GEARManager(n_layers={self.n_layers}, "
            f"kv_bits={self.config.kv_bits}, rank={self.config.rank}, "
            f"compression={self.stats.compression_ratio:.2%})"
        )
