"""squish/kv/kvquant.py

KVQuant — Per-Vector NF4 + Per-Channel Calibrated KV Quantization
(Hooper et al., NeurIPS 2024 / arXiv:2401.18079).

Reference
---------
"KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
Quantization." Hooper et al., NeurIPS 2024 (arXiv:2401.18079).

Algorithm
---------
KVQuant quantizes K and V caches to low bit-width using:

* **Per-channel calibrated scales** — collected from a rolling calibration
  window of ``calibration_window`` recent K/V vectors.
* **Uniform quantization** to ``bits`` bit-width (symmetric, zero-centered).

Quantization:
    codes = round(clip(x / scale, -q_max, q_max) + q_max)

Dequantization:
    x_hat = (codes - q_max) * scale

The scale per channel ``c`` is ``max(|mean_k[c]|, eps)`` where ``mean_k`` is
the running mean of the absolute value of K/V entries in that channel,
estimated over the calibration window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "KVQuantConfig",
    "KVQuantCache",
]

_VALID_BITS = frozenset({2, 4, 8})

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class KVQuantConfig:
    """Configuration for :class:`KVQuantCache`.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        bits: Quantization bit-width (2, 4, or 8).
        calibration_window: Number of samples used to estimate channel scales.
    """

    n_heads: int = 8
    head_dim: int = 64
    bits: int = 4
    calibration_window: int = 64

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")
        if self.bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {sorted(_VALID_BITS)}; got {self.bits}")
        if self.calibration_window < 1:
            raise ValueError(f"calibration_window must be ≥ 1; got {self.calibration_window}")


# ── KVQuantCache ──────────────────────────────────────────────────────────────


class KVQuantCache:
    """Calibrated low-bit KV cache.

    Example::

        cfg = KVQuantConfig(n_heads=2, head_dim=8, bits=4, calibration_window=32)
        cache = KVQuantCache(cfg)

        rng = np.random.default_rng(0)
        K = rng.standard_normal((2, 16, 8)).astype(np.float32)
        V = rng.standard_normal((2, 16, 8)).astype(np.float32)
        cache.calibrate(K, V)
        cache.quantize(0, K, V)
        K2, V2 = cache.dequantize(0)
    """

    def __init__(self, config: Optional[KVQuantConfig] = None) -> None:
        self.config = config or KVQuantConfig()
        cfg = self.config
        d = cfg.n_heads * cfg.head_dim
        # Running absolute mean for scale estimation.
        self._k_abs_acc = np.zeros(d, dtype=np.float64)
        self._v_abs_acc = np.zeros(d, dtype=np.float64)
        self._calib_count = 0
        # Calibrated scales.
        self._k_scale: Optional[np.ndarray] = None  # (n_heads, head_dim)
        self._v_scale: Optional[np.ndarray] = None
        # Quantized storage: layer_id → (K_codes, V_codes, original_shape)
        self._store: Dict[int, Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]] = {}

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, K: np.ndarray, V: np.ndarray) -> None:
        """Update channel scale estimates from ``K`` and ``V``.

        Args:
            K: ``(n_heads, T, head_dim)`` or ``(n_heads * head_dim, T)`` float32.
            V: Same shape as K.
        """
        K_flat = self._flatten(K)  # (d, T)
        V_flat = self._flatten(V)
        window = self.config.calibration_window
        n_samples = min(K_flat.shape[1], window - self._calib_count)
        if n_samples <= 0:
            return
        self._k_abs_acc += np.abs(K_flat[:, :n_samples]).mean(axis=1)
        self._v_abs_acc += np.abs(V_flat[:, :n_samples]).mean(axis=1)
        self._calib_count += 1
        if self._calib_count > 0:
            self._k_scale = (
                self._k_abs_acc / self._calib_count
            ).reshape(self.config.n_heads, self.config.head_dim).astype(np.float32)
            self._v_scale = (
                self._v_abs_acc / self._calib_count
            ).reshape(self.config.n_heads, self.config.head_dim).astype(np.float32)
            self._k_scale = np.maximum(self._k_scale, 1e-7)
            self._v_scale = np.maximum(self._v_scale, 1e-7)

    # ── Quantize / Dequantize ──────────────────────────────────────────────────

    def quantize(self, layer_id: int, K: np.ndarray, V: np.ndarray) -> None:
        """Quantize and store KV tensors for ``layer_id``.

        Args:
            layer_id: Integer layer index.
            K: ``(n_heads, T, head_dim)`` float32.
            V: Same shape as K.
        """
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        shape = K.shape

        k_scale, v_scale = self._get_scales(K, V)
        K_codes = self._quantize_tensor(K, k_scale[..., np.newaxis, :])
        V_codes = self._quantize_tensor(V, v_scale[..., np.newaxis, :])
        self._store[layer_id] = (K_codes, V_codes, shape)

    def dequantize(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Dequantize and return KV tensors for ``layer_id``.

        Raises:
            KeyError: If ``layer_id`` has not been quantized.
        """
        if layer_id not in self._store:
            raise KeyError(f"No quantized KV for layer_id={layer_id}")
        K_codes, V_codes, shape = self._store[layer_id]
        tmp_K = np.ones(shape, dtype=np.float32)
        tmp_V = np.ones(shape, dtype=np.float32)
        k_scale, v_scale = self._get_scales(tmp_K, tmp_V)
        K_hat = self._dequantize_tensor(K_codes, k_scale[..., np.newaxis, :], shape)
        V_hat = self._dequantize_tensor(V_codes, v_scale[..., np.newaxis, :], shape)
        return K_hat, V_hat

    # ── Stats ─────────────────────────────────────────────────────────────────

    def memory_bytes(self) -> int:
        """Approximate bytes used by quantized codes."""
        total = 0
        bits_per_entry = self.config.bits
        for K_codes, V_codes, _ in self._store.values():
            # Each code is stored as uint8 but logically uses `bits` bits.
            total += K_codes.nbytes + V_codes.nbytes
        return total

    def n_layers_cached(self) -> int:
        """Number of layers currently in the quantized store."""
        return len(self._store)

    def relative_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Mean relative L2 error."""
        denom = np.linalg.norm(original) + 1e-9
        return float(np.linalg.norm(original - reconstructed) / denom)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        """Reshape ``(H, T, d)`` → ``(H*d, T)``."""
        x = np.asarray(x, dtype=np.float32)
        H, T, d = x.shape
        return x.transpose(0, 2, 1).reshape(H * d, T)

    def _get_scales(
        self, K: np.ndarray, V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._k_scale is not None:
            return self._k_scale, self._v_scale  # type: ignore[return-value]
        # Per-token fallback scale if no calibration has been done yet.
        k_scale = np.maximum(np.abs(K).mean(axis=1, keepdims=False), 1e-7)
        v_scale = np.maximum(np.abs(V).mean(axis=1, keepdims=False), 1e-7)
        return k_scale, v_scale  # (H, d) effectively

    def _quantize_tensor(
        self, x: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        bits = self.config.bits
        q_max = (1 << bits) - 1
        half_q = q_max // 2
        x_scaled = x / scale
        codes = np.clip(np.round(x_scaled + half_q), 0, q_max).astype(np.uint8)
        return codes

    def _dequantize_tensor(
        self, codes: np.ndarray, scale: np.ndarray, shape: tuple
    ) -> np.ndarray:
        bits = self.config.bits
        q_max = (1 << bits) - 1
        half_q = q_max // 2
        return (codes.astype(np.float32) - half_q) * scale

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"KVQuantCache(n_heads={cfg.n_heads}, head_dim={cfg.head_dim}, "
            f"bits={cfg.bits}, calibration_window={cfg.calibration_window})"
        )
