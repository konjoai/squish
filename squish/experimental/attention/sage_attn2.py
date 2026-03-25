"""squish/attention/sage_attn2.py

SageAttn2 — Per-Thread INT4 Q/K Attention with Outlier Smoothing.

Reference
---------
Zhang et al. "SageAttention 2: Efficient Attention with Thorough Outlier
Smoothing and Per-thread INT4 Quantization."
ICLR 2025 (arXiv:2411.10958).

Algorithm
---------
SageAttention 2 accelerates attention by quantizing Q and K to INT4:

1. **Outlier smoothing** — subtract per-channel mean from Q and K to
   suppress extreme activation outliers before quantization.
2. **Per-thread INT4 Q/K quantization** — quantize the smoothed Q and K
   to symmetric INT4 with a per-vector scale.
3. **INT4 dot-product accumulation** — compute scores using integer
   arithmetic; re-scale before softmax.
4. **FP16/FP32 V accumulation** — values are kept in higher precision
   to preserve output quality.

This module provides a NumPy simulation with the same API as standard
scaled dot-product attention.

Key properties
--------------
* NumPy-only simulation (integer arithmetic approximated in float).
* ``n_heads`` — number of attention heads.
* ``head_dim`` — dimension per head.
* ``smooth_outliers`` — apply per-channel mean subtraction.
* ``quantize_qk`` — enable INT4 Q/K quantization (set False to disable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "SageAttn2Config",
    "SageAttn2",
]


@dataclass
class SageAttn2Config:
    """Configuration for :class:`SageAttn2`.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        smooth_outliers: If True, subtract per-channel mean before quant.
        quantize_qk: If True, quantize Q and K to INT4.
    """

    n_heads: int = 32
    head_dim: int = 128
    smooth_outliers: bool = True
    quantize_qk: bool = True


class SageAttn2:
    """SageAttention 2 — per-thread INT4 Q/K with outlier smoothing.

    Parameters
    ----------
    config:
        SageAttn2 configuration.
    """

    def __init__(self, config: Optional[SageAttn2Config] = None) -> None:
        self._cfg = config or SageAttn2Config()
        # Running per-channel mean for Q and K (shape: head_dim)
        self._q_channel_mean: Optional[np.ndarray] = None
        self._k_channel_mean: Optional[np.ndarray] = None
        self._calibration_count: int = 0

    @property
    def _q_mean(self) -> Optional[np.ndarray]:
        """Alias for _q_channel_mean (used by calibrate_sets_mean tests)."""
        return self._q_channel_mean

    @property
    def config(self) -> SageAttn2Config:
        return self._cfg

    @property
    def calibration_count(self) -> int:
        return self._calibration_count

    def _quantize_int4(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric INT4 quantization.

        Parameters
        ----------
        x:
            Input tensor, any shape.

        Returns
        -------
        Tuple of (quantized, scale) where quantized is in [-8, 7] and
        scale is the per-vector dequantisation factor.
        """
        x = x.astype(np.float32)
        orig_shape = x.shape
        flat = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
        scale = np.abs(flat).max(axis=1, keepdims=True).clip(min=1e-8) / 7.0
        q = np.round(flat / scale).clip(-8, 7).astype(np.int8)
        return q.reshape(orig_shape), scale.reshape(x.shape[0])

    def calibrate(self, query: np.ndarray, key: np.ndarray) -> None:
        """Update running per-channel mean from calibration samples.

        Parameters
        ----------
        query: ``(n_heads, seq_len, head_dim)`` or ``(n_heads, head_dim)``
        key: Same shape as query.
        """
        q = np.asarray(query, dtype=np.float32)
        k = np.asarray(key, dtype=np.float32)
        if q.ndim == 2:
            q = q[:, None, :]
            k = k[:, None, :]
        # Mean over (heads, tokens)
        q_mean = q.mean(axis=(0, 1))
        k_mean = k.mean(axis=(0, 1))
        n = self._calibration_count
        if n == 0:
            self._q_channel_mean = q_mean
            self._k_channel_mean = k_mean
        else:
            self._q_channel_mean = (self._q_channel_mean * n + q_mean) / (n + 1)
            self._k_channel_mean = (self._k_channel_mean * n + k_mean) / (n + 1)
        self._calibration_count += 1

    def forward(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Compute SageAttn2 attention.

        Parameters
        ----------
        query:
            Shape ``(n_heads, head_dim)`` — single decode step.
        keys:
            Shape ``(n_heads, seq_len, head_dim)``.
        values:
            Shape ``(n_heads, seq_len, head_dim)`` — kept in FP32.

        Returns
        -------
        np.ndarray
            Output tensor, shape ``(n_heads, head_dim)``.
        """
        q_in = np.asarray(query, dtype=np.float32)
        K = np.asarray(keys, dtype=np.float32)   # (n_heads, seq_len, head_dim)
        V = np.asarray(values, dtype=np.float32)  # (n_heads, seq_len, head_dim)
        # Handle optional q_len dimension: accept (n_heads, head_dim) or (n_heads, q_len, head_dim)
        has_q_len = q_in.ndim == 3
        q = q_in.squeeze(1) if has_q_len else q_in  # (n_heads, head_dim)
        scale = float(self._cfg.head_dim ** -0.5)

        # Step 1: outlier smoothing
        if self._cfg.smooth_outliers and self._q_channel_mean is not None:
            q = q - self._q_channel_mean[None, :]  # broadcast over heads
            K = K - self._k_channel_mean[None, None, :]

        # Step 2: INT4 quantization of Q and K (simulated in float)
        if self._cfg.quantize_qk:
            # Quantize each head's query independently
            q_q, q_scale = self._quantize_int4(q)   # q_q: (n_heads, head_dim)
            # Quantize each key token independently
            K_flat = K.reshape(-1, self._cfg.head_dim)
            K_q, K_scale = self._quantize_int4(K_flat)
            K_q = K_q.reshape(K.shape)
            K_scale = K_scale.reshape(K.shape[0], -1)  # (n_heads, seq_len)
            # De-quantize back to float for the dot product
            q_deq = (q_q * q_scale[:, None]).astype(np.float32)  # (n_heads, head_dim)
            K_deq = (K_q * K_scale[:, :, None]).astype(np.float32)
        else:
            q_deq = q
            K_deq = K

        # Step 3: scaled dot-product scores
        scores = np.einsum("hd,hsd->hs", q_deq, K_deq) * scale  # (n_heads, seq_len)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)

        # Step 4: FP32 V accumulation
        output = np.einsum("hs,hsd->hd", w, V)  # (n_heads, head_dim)
        if has_q_len:
            output = output[:, None, :]  # (n_heads, 1, head_dim)
        return output.astype(np.float32)

    def reset_calibration(self) -> None:
        self._q_channel_mean = None
        self._k_channel_mean = None
        self._calibration_count = 0
