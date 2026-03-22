"""AutoRound quantization (arXiv 2309.05516, EMNLP 2024).

Sign-projected AdamW rounding optimiser per linear layer.
No Hessian required; beats GPTQ INT2/INT3 by 0.3–0.5 PPL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AutoRoundConfig:
    """Configuration for AutoRound quantization."""

    bits: int = 3
    group_size: int = 128
    n_steps: int = 512
    lr: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.bits < 1 or self.bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


@dataclass
class AutoRoundResult:
    """Output of AutoRound quantization."""

    quantized: np.ndarray      # (rows, cols) int8
    scales: np.ndarray         # (n_groups,) float32 per-group scale
    zeros: np.ndarray          # (n_groups,) float32 per-group zero-point
    shape: Tuple[int, int]
    group_size: int
    bits: int
    loss_history: List[float] = field(default_factory=list)


class AutoRoundQuantizer:
    """AutoRound: sign-projected gradient rounding optimiser.

    Iteratively refines rounding decisions over ``n_steps`` Adam steps
    using only the weight matrix itself (or optional calibration data)
    without computing a Hessian.
    """

    def __init__(self, config: Optional[AutoRoundConfig] = None) -> None:
        self._config = config or AutoRoundConfig()

    @property
    def config(self) -> AutoRoundConfig:
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize_groups(
        self,
        W: np.ndarray,
        rounding_offsets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize W with learned rounding offsets.

        Parameters
        ----------
        W:
            ``(rows, cols)`` float32 weight matrix (already padded to
            a multiple of ``group_size``).
        rounding_offsets:
            ``(rows, cols)`` float32 values in ``[‑0.5, 0.5]`` that are
            added to the scaled weight *before* floor-rounding so the
            optimiser can flip rounding decisions.

        Returns
        -------
        quantized, scales, zeros
            ``quantized`` is int8; ``scales`` and ``zeros`` are float32
            per-group arrays of shape ``(rows * n_col_groups,)``.
        """
        cfg = self._config
        rows, cols = W.shape
        n_col_groups = cols // cfg.group_size

        q_min = -(1 << (cfg.bits - 1))
        q_max = (1 << (cfg.bits - 1)) - 1

        scales = np.empty(rows * n_col_groups, dtype=np.float32)
        zeros = np.empty(rows * n_col_groups, dtype=np.float32)
        quantized = np.empty((rows, cols), dtype=np.int8)

        for r in range(rows):
            for gc in range(n_col_groups):
                col_start = gc * cfg.group_size
                col_end = col_start + cfg.group_size
                g = W[r, col_start:col_end].astype(np.float32)
                g_off = rounding_offsets[r, col_start:col_end]

                g_min = float(g.min())
                g_max = float(g.max())
                scale = (g_max - g_min) / ((1 << cfg.bits) - 1) if g_max != g_min else 1.0
                zero = -g_min / scale

                q = np.floor((g / scale + zero) + g_off + 0.5).astype(np.float32)
                q = np.clip(q, q_min, q_max).astype(np.int8)

                idx = r * n_col_groups + gc
                scales[idx] = scale
                zeros[idx] = zero
                quantized[r, col_start:col_end] = q

        return quantized, scales, zeros

    def _dequant_groups(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
    ) -> np.ndarray:
        rows, cols = quantized.shape
        cfg = self._config
        n_col_groups = cols // cfg.group_size
        W_hat = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            for gc in range(n_col_groups):
                col_start = gc * cfg.group_size
                col_end = col_start + cfg.group_size
                idx = r * n_col_groups + gc
                W_hat[r, col_start:col_end] = (
                    quantized[r, col_start:col_end].astype(np.float32) - zeros[idx]
                ) * scales[idx]
        return W_hat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(
        self,
        W: np.ndarray,
        calibration_data: Optional[np.ndarray] = None,
    ) -> AutoRoundResult:
        """Quantize weight matrix W using sign-projected AdamW rounding.

        Parameters
        ----------
        W:
            2-D float weight matrix ``(rows, cols)``.
        calibration_data:
            Optional ``(n_samples, cols)`` float32 activation matrix.
            When provided the reconstruction error is measured relative
            to calibration outputs; otherwise the plain weight-matrix
            Frobenius error is used.

        Returns
        -------
        AutoRoundResult
        """
        cfg = self._config
        rng = np.random.default_rng(cfg.seed)

        W = np.asarray(W, dtype=np.float32)
        rows, cols = W.shape

        # Pad cols to multiple of group_size
        pad = (-cols) % cfg.group_size
        if pad:
            W = np.pad(W, ((0, 0), (0, pad)))
        padded_cols = W.shape[1]

        # Initialise rounding offsets to zero (floor rounding baseline)
        offsets = np.zeros((rows, padded_cols), dtype=np.float32)

        # Adam state
        m = np.zeros_like(offsets)
        v = np.zeros_like(offsets)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        loss_history: List[float] = []

        for step in range(1, cfg.n_steps + 1):
            q, scales, zeros = self._quantize_groups(W, offsets)
            W_hat = self._dequant_groups(q, scales, zeros)

            if calibration_data is not None:
                cal = np.asarray(calibration_data, dtype=np.float32)
                # cal: (n_samples, rows); W: (rows, padded_cols)
                # out = cal @ W  →  (n_samples, padded_cols)
                out_true = cal @ W
                out_hat = cal @ W_hat
                grad_W_hat = cal.T @ (out_hat - out_true) / cal.shape[0]
                # gradient w.r.t. offsets via chain rule (jacobian = scales broadcast)
                grad = grad_W_hat * _broadcast_scales(scales, rows, padded_cols, cfg.group_size)
            else:
                grad = W_hat - W

            # Sign-project gradient then Adam update
            sign_grad = np.sign(grad)
            m = beta1 * m + (1 - beta1) * sign_grad
            v = beta2 * v + (1 - beta2) * (sign_grad ** 2)
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            offsets -= cfg.lr * m_hat / (np.sqrt(v_hat) + eps)
            # Clamp offsets to valid rounding range
            offsets = np.clip(offsets, -0.5, 0.5)

            loss = float(np.mean((W_hat - W) ** 2))
            loss_history.append(loss)

        # Final quantization with optimised offsets
        quantized, scales, zeros = self._quantize_groups(W, offsets)
        # Trim padding back to original cols
        quantized = quantized[:, :cols]

        return AutoRoundResult(
            quantized=quantized,
            scales=scales,
            zeros=zeros,
            shape=(rows, cols),
            group_size=cfg.group_size,
            bits=cfg.bits,
            loss_history=loss_history,
        )

    def dequantize(self, result: AutoRoundResult) -> np.ndarray:
        """Reconstruct float32 weight from AutoRoundResult."""
        cfg = self._config
        rows, cols = result.shape
        pad = (-cols) % cfg.group_size
        padded_cols = cols + pad

        q_padded = np.pad(result.quantized, ((0, 0), (0, pad))) if pad else result.quantized
        W_hat = self._dequant_groups(q_padded, result.scales, result.zeros)
        return W_hat[:, :cols]

    def forward(self, x: np.ndarray, result: AutoRoundResult) -> np.ndarray:
        """Matrix-multiply input ``x`` by the dequantized weight.

        Parameters
        ----------
        x:
            ``(*, rows)`` float32 input tensor.
        result:
            Quantized weight produced by :meth:`quantize`.

        Returns
        -------
        np.ndarray
            ``(*, cols)`` output.
        """
        W_hat = self.dequantize(result)
        return np.tensordot(x, W_hat, axes=([-1], [0]))


# ---------------------------------------------------------------------------
# Module-level helper (not part of public API)
# ---------------------------------------------------------------------------


def _broadcast_scales(
    scales: np.ndarray,
    rows: int,
    cols: int,
    group_size: int,
) -> np.ndarray:
    """Expand per-group scales to the full (rows, cols) shape."""
    n_col_groups = cols // group_size
    s = scales.reshape(rows, n_col_groups)
    return np.repeat(s, group_size, axis=1)
