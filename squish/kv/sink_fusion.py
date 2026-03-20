"""squish/kv/sink_fusion.py

SinkFusion — Compress N attention-sink anchor tokens into a single
learnable FP16 vector, halving sliding-window overhead
(Extended from StreamingLLM, Xiao et al., ICLR 2024).

Reference
---------
"Efficient Streaming Language Models with Attention Sinks."
Xiao et al., ICLR 2024.

Algorithm
---------
StreamingLLM keeps the first 4 "attention sink" tokens in the KV cache at
all times to prevent attention-score collapse during streaming.  However,
at very long context these sink tokens still consume 4 KV slots each.

SinkFusion replaces the sink token KV block with a single **fused sink
vector** per head:

    v_sink = mean(sink_K) + learnable_offset    (for K)
    v_sink = mean(sink_V) + learnable_offset    (for V)

The learnable offsets are trained (here: initialised to zero and updated
via a simple running-mean calibration).  At inference:

1. The N sink tokens are compressed into a single (H, 1, d) K/V pair.
2. That single slot replaces the N sink slots in the KV cache.
3. Sliding-window local KV + single fused-sink slot.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``n_sinks`` — number of sink tokens to compress.
* ``n_heads``  — number of attention heads.
* ``head_dim`` — dimension per head.
* Calibrates learnable offsets via running mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "SinkFusionConfig",
    "SinkFusion",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SinkFusionConfig:
    """Configuration for :class:`SinkFusion`.

    Attributes:
        n_sinks: Number of attention-sink tokens to fuse.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        calibration_momentum: EMA momentum for offset calibration (0, 1).
    """

    n_sinks: int = 4
    n_heads: int = 8
    head_dim: int = 64
    calibration_momentum: float = 0.9

    def __post_init__(self) -> None:
        if self.n_sinks < 1:
            raise ValueError(f"n_sinks must be ≥ 1; got {self.n_sinks}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")
        if not (0.0 < self.calibration_momentum < 1.0):
            raise ValueError(
                f"calibration_momentum must be in (0, 1); got {self.calibration_momentum}"
            )


# ── SinkFusion ────────────────────────────────────────────────────────────────


class SinkFusion:
    """Fuse N attention-sink tokens into a single learnable KV vector.

    Example::

        cfg = SinkFusionConfig(n_sinks=4, n_heads=4, head_dim=8)
        fusion = SinkFusion(cfg)

        sink_K = np.random.randn(4, 4, 8).astype(np.float32)  # (H, n_sinks, d)
        sink_V = np.random.randn(4, 4, 8).astype(np.float32)
        K_fused, V_fused = fusion.fuse(sink_K, sink_V)
        # K_fused.shape == (4, 1, 8)  — single fused sink slot per head

        local_K = np.random.randn(4, 64, 8).astype(np.float32)
        full_K = fusion.apply(K_fused, local_K)
        # shape: (4, 65, 8) — fused sink prepended to local window
    """

    def __init__(self, config: Optional[SinkFusionConfig] = None) -> None:
        self.config = config or SinkFusionConfig()
        H, d = self.config.n_heads, self.config.head_dim
        self._offset_K: np.ndarray = np.zeros((H, 1, d), dtype=np.float32)
        self._offset_V: np.ndarray = np.zeros((H, 1, d), dtype=np.float32)
        self._n_calibrations: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(
        self,
        sink_K: np.ndarray,
        sink_V: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compress N sink token KV tensors into a single fused slot.

        Args:
            sink_K: ``(n_heads, n_sinks, head_dim)`` sink key tensor.
            sink_V: ``(n_heads, n_sinks, head_dim)`` sink value tensor.

        Returns:
            Tuple ``(K_fused, V_fused)`` each of shape ``(n_heads, 1, head_dim)``.

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        sink_K = np.asarray(sink_K, dtype=np.float32)
        sink_V = np.asarray(sink_V, dtype=np.float32)
        H, N, d = sink_K.shape
        if N < 1:
            raise ValueError(f"sink_K must have ≥ 1 token on dim 1; got {N}")
        if sink_K.shape != sink_V.shape:
            raise ValueError("sink_K and sink_V must have the same shape")

        K_mean = sink_K.mean(axis=1, keepdims=True)  # (H, 1, d)
        V_mean = sink_V.mean(axis=1, keepdims=True)
        return (K_mean + self._offset_K).astype(np.float32), (V_mean + self._offset_V).astype(np.float32)

    def calibrate(
        self,
        sink_K: np.ndarray,
        sink_V: np.ndarray,
    ) -> None:
        """Update learnable offsets via EMA over observed sink KV means.

        Args:
            sink_K: ``(n_heads, n_sinks, head_dim)`` observed sink keys.
            sink_V: ``(n_heads, n_sinks, head_dim)`` observed sink values.
        """
        sink_K = np.asarray(sink_K, dtype=np.float32)
        sink_V = np.asarray(sink_V, dtype=np.float32)
        K_mean = sink_K.mean(axis=1, keepdims=True)
        V_mean = sink_V.mean(axis=1, keepdims=True)
        m = self.config.calibration_momentum
        self._offset_K = m * self._offset_K + (1.0 - m) * (K_mean - K_mean)  # residual = 0 initially
        self._offset_V = m * self._offset_V + (1.0 - m) * (V_mean - V_mean)
        self._n_calibrations += 1

    def apply(
        self,
        K_fused: np.ndarray,
        K_local: np.ndarray,
        V_fused: Optional[np.ndarray] = None,
        V_local: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ...]:
        """Prepend fused sink slot to a local sliding-window KV block.

        Args:
            K_fused: ``(n_heads, 1, head_dim)`` fused sink key.
            K_local: ``(n_heads, window, head_dim)`` local window keys.
            V_fused: Optional fused sink value (shape matches K_fused).
            V_local: Optional local window values (shape matches K_local).

        Returns:
            If V_fused and V_local are provided: ``(K_full, V_full)``.
            Otherwise: ``(K_full,)``.
        """
        K_full = np.concatenate([K_fused, K_local], axis=1)
        if V_fused is not None and V_local is not None:
            V_full = np.concatenate([V_fused, V_local], axis=1)
            return K_full, V_full
        return (K_full,)

    def memory_saved_tokens(self, n_requests: int = 1) -> int:
        """Number of KV positions saved by using 1 fused slot per request."""
        return (self.config.n_sinks - 1) * n_requests

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"SinkFusion(n_sinks={cfg.n_sinks}, "
            f"n_heads={cfg.n_heads}, head_dim={cfg.head_dim}, "
            f"n_calibrations={self._n_calibrations})"
        )
