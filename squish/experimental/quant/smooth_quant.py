"""SmoothQuantActivation — per-channel activation smoothing for W8A8 INT8.

Implements the SmoothQuant algorithm (Xiao et al., ICML 2023).  The core idea
is to *migrate* quantisation difficulty from activations (which have large
per-token outliers) to weights (which are much smoother) by rescaling:

    Y = (X diag(s)^{-1}) · (diag(s) W)
           ↑ smoothed        ↑ absorbed

where ``s`` is a per-channel scale vector computed from calibration statistics.
Both the rescaled activation and rescaled weight can then be quantised to INT8
without significant accuracy loss.

The module is self-contained and NumPy-only so it runs on every Squish platform
(Apple Silicon, CUDA, CPU-only) without GPU dependencies.

Reference:
    Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
    Quantization for Large Language Models", ICML 2023.
    arXiv:2211.10438
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "SmoothQuantConfig",
    "SmoothQuantActivation",
]

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class SmoothQuantConfig:
    """Configuration for SmoothQuantActivation.

    Attributes:
        alpha: Migration strength in [0, 1].  ``alpha=0`` puts all difficulty
            on weights; ``alpha=1`` puts all difficulty on activations.
            The SmoothQuant paper recommends ``alpha=0.5`` as a robust default.
        epsilon: Small constant added when computing scales to prevent
            division by zero.
        per_token_dynamic: If True, re-compute per-token activation scale at
            runtime (dynamic quantisation); if False, use calibrated static
            scales.
        bits: Target bit-width for INT quantisation (8 gives INT8 range).
    """

    alpha: float = 0.5
    epsilon: float = 1e-5
    per_token_dynamic: bool = False
    bits: int = 8

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive; got {self.epsilon}")
        if self.bits not in (4, 8, 16):
            raise ValueError(f"bits must be 4, 8, or 16; got {self.bits}")


# ── Core class ───────────────────────────────────────────────────────────────


class SmoothQuantActivation:
    """Per-channel activation-to-weight difficulty migration.

    Call :meth:`calibrate` once on a representative set of activations and
    weights.  Afterwards, :meth:`smooth_weight` returns the pre-absorbed
    weight matrix and :meth:`smooth_activation` re-scales incoming
    activations at runtime (either statically or dynamically).

    Example::

        smoother = SmoothQuantActivation()
        smoother.calibrate(act_samples, weight)   # one-time
        W_smooth = smoother.smooth_weight(weight)
        x_smooth = smoother.smooth_activation(x)
        # W_smooth and x_smooth can now be INT8-quantised safely

    Args:
        config: Optional :class:`SmoothQuantConfig`.  Defaults to
            ``SmoothQuantConfig()`` (alpha=0.5, INT8).
    """

    def __init__(self, config: Optional[SmoothQuantConfig] = None) -> None:
        self.config: SmoothQuantConfig = config or SmoothQuantConfig()
        self._scales: Optional[np.ndarray] = None  # shape (C_in,)
        self._calibrated: bool = False

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(
        self,
        activations: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        """Compute per-channel smooth scales from calibration data.

        Args:
            activations: Representative activation tensor of shape
                ``(..., C_in)``; typically a few hundred tokens from the
                calibration set.
            weight: Weight matrix of shape ``(C_out, C_in)`` (row-major,
                as returned by a linear layer).

        Returns:
            Scale vector ``s`` of shape ``(C_in,)``.

        Raises:
            ValueError: If ``activations`` and ``weight`` have mismatched
                ``C_in`` dimensions.
        """
        act = np.asarray(activations, dtype=np.float32)
        W = np.asarray(weight, dtype=np.float32)

        if act.ndim < 2:
            raise ValueError(
                f"activations must have at least 2 dimensions; got shape {act.shape}"
            )
        if W.ndim != 2:
            raise ValueError(
                f"weight must be 2-D (C_out, C_in); got shape {W.shape}"
            )
        c_in_act = act.shape[-1]
        c_in_w = W.shape[1]
        if c_in_act != c_in_w:
            raise ValueError(
                f"C_in mismatch: activations={c_in_act}, weight={c_in_w}"
            )

        alpha = self.config.alpha
        eps = self.config.epsilon

        # Per-channel activation max-abs across all tokens / batch dims
        flat = act.reshape(-1, c_in_act)
        act_max = np.abs(flat).max(axis=0) + eps  # (C_in,)

        # Per-channel weight max-abs across output channels
        w_max = np.abs(W).max(axis=0) + eps  # (C_in,)

        # s = act_max^alpha / w_max^(1-alpha)
        scales = (act_max ** alpha) / (w_max ** (1.0 - alpha) + eps)
        self._scales = scales.astype(np.float32)
        self._calibrated = True
        return self._scales

    # ── Apply ─────────────────────────────────────────────────────────────────

    def smooth_weight(self, weight: np.ndarray) -> np.ndarray:
        """Absorb the scale into the weight matrix (``W ← diag(s) W``).

        Args:
            weight: ``(C_out, C_in)`` weight matrix.

        Returns:
            Rescaled weight of the same shape.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        self._require_calibrated()
        W = np.asarray(weight, dtype=np.float32)
        # Broadcast (C_in,) over rows: multiply each column by s[i]
        return (W * self._scales[np.newaxis, :]).astype(np.float32)

    def smooth_activation(self, activation: np.ndarray) -> np.ndarray:
        """Divide the activation by the scale (``X ← X diag(s)^{-1}``).

        If ``config.per_token_dynamic`` is True, the per-token dynamic
        scaling is applied on top of the calibrated channel scale.

        Args:
            activation: ``(..., C_in)`` activation tensor.

        Returns:
            Rescaled activation of the same shape as input.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        self._require_calibrated()
        x = np.asarray(activation, dtype=np.float32)
        x_smooth = x / (self._scales + self.config.epsilon)
        if self.config.per_token_dynamic:
            flat = x_smooth.reshape(-1, x_smooth.shape[-1])
            tok_scale = np.abs(flat).max(axis=1, keepdims=True) + self.config.epsilon
            x_smooth = x_smooth / tok_scale.reshape(
                x_smooth.shape[:-1] + (1,)
            )
        return x_smooth.astype(np.float32)

    # ── Quantisation helpers ──────────────────────────────────────────────────

    def quantise_int8(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Round-to-nearest INT8 quantisation with per-tensor scale.

        Args:
            x: Float32 array of any shape.

        Returns:
            Tuple ``(x_int8, scale)`` where ``x_int8`` is ``np.int8`` and
            ``scale`` is a float32 scalar such that
            ``x ≈ x_int8 * scale``.
        """
        x = np.asarray(x, dtype=np.float32)
        qmax = float(2 ** (self.config.bits - 1) - 1)
        amax = float(np.abs(x).max()) + self.config.epsilon
        scale = amax / qmax
        x_q = np.clip(np.round(x / scale), -qmax - 1, qmax).astype(np.int8)
        return x_q, np.float32(scale)

    def dequantise_int8(
        self, x_int8: np.ndarray, scale: float
    ) -> np.ndarray:
        """Dequantise INT8 back to float32.

        Args:
            x_int8: Integer array (INT8).
            scale: Scale factor returned by :meth:`quantise_int8`.

        Returns:
            Float32 array.
        """
        return x_int8.astype(np.float32) * float(scale)

    def forward_smoothed(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Smooth, quantise, and perform INT8 matmul.

        Replicates a single linear layer step end-to-end:
            1. Smooth ``x`` by dividing by calibrated channels.
            2. Quantise ``x_smooth`` and ``weight_smooth`` to INT8.
            3. Dequantise and multiply.
            4. Add optional bias.

        Args:
            x: ``(..., C_in)`` float32 input activation.
            weight: ``(C_out, C_in)`` float32 weight.
            bias: Optional ``(C_out,)`` float32 bias.

        Returns:
            Output ``(..., C_out)`` float32 tensor.
        """
        self._require_calibrated()
        x_s = self.smooth_activation(x)
        w_s = self.smooth_weight(weight)
        x_q, sx = self.quantise_int8(x_s)
        w_q, sw = self.quantise_int8(w_s)
        # INT8 matmul in int32 accumulator
        batch_shape = x.shape[:-1]
        flat = x_q.astype(np.int32).reshape(-1, x.shape[-1])
        out_int32 = flat @ w_q.astype(np.int32).T  # (N, C_out)
        out = out_int32.astype(np.float32) * (sx * sw)
        out = out.reshape(batch_shape + (weight.shape[0],))
        if bias is not None:
            out = out + np.asarray(bias, dtype=np.float32)
        return out

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def scales(self) -> Optional[np.ndarray]:
        """Calibrated scale vector, or None if not yet calibrated."""
        return self._scales

    @property
    def is_calibrated(self) -> bool:
        """True after :meth:`calibrate` has been called successfully."""
        return self._calibrated

    # ── Internals ────────────────────────────────────────────────────────────

    def _require_calibrated(self) -> None:
        if not self._calibrated:
            raise RuntimeError(
                "SmoothQuantActivation: calibrate() must be called before "
                "smooth_weight() or smooth_activation()."
            )

    def __repr__(self) -> str:
        status = "calibrated" if self._calibrated else "uncalibrated"
        return (
            f"SmoothQuantActivation(alpha={self.config.alpha}, "
            f"bits={self.config.bits}, {status})"
        )
