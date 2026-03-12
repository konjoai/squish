"""GPTQLayer — GPTQ-style second-order quantisation calibration.

Approximates the Hessian of quantisation error using input activation
covariance ``H ≈ X^T X / n``.  For each column ``j`` of the weight matrix,
the optimal quantised value minimises the second-order error
``ΔW^T H ΔW``.  Column-wise quantisation proceeds with per-column importance
weighting derived from the diagonal of the (damped) Hessian, following the
practical simplification of the full GPTQ algorithm suitable for a
calibration-only pass without requiring a Cholesky inverse update.

The full Cholesky-blocked column-update from the original paper is
represented here by a diagonal-Hessian weighting that captures the key
insight: columns with high Hessian diagonal values (i.e. weight dimensions
activated by many calibration samples) should be quantised more conservatively.

Reference:
    Frantar et al., "GPTQ: Accurate Post-Training Quantization for
    Generative Pre-trained Transformers", ICLR 2023.
    https://arxiv.org/abs/2210.17323

Usage::

    import numpy as np
    from squish.gptq_layer import GPTQConfig, GPTQCalibrator, GPTQStats

    rng   = np.random.default_rng(0)
    rows, cols = 256, 512
    n_samples  = 128

    weights     = rng.standard_normal((rows, cols)).astype(np.float32)
    activations = rng.standard_normal((n_samples, rows)).astype(np.float32)

    cfg        = GPTQConfig(n_bits=4, block_size=128, damp_percent=0.01)
    calibrator = GPTQCalibrator(cfg)

    quantized_w = calibrator.calibrate(weights, activations)
    print(quantized_w.shape)  # (256, 512)
    print(calibrator.stats)
"""

from __future__ import annotations

__all__ = ["GPTQConfig", "GPTQCalibrator", "GPTQStats"]

from dataclasses import dataclass

import numpy as np


@dataclass
class GPTQConfig:
    """Configuration for GPTQ-style calibration.

    Attributes:
        n_bits: Target quantisation bit-width; one of {2, 3, 4, 8}.
        block_size: Number of columns processed per block.
        damp_percent: Hessian diagonal damping coefficient.  The diagonal of H
            is increased by ``damp_percent * mean(diag(H))`` before use as
            per-column importance weights.
    """

    n_bits:       int   = 4
    block_size:   int   = 128
    damp_percent: float = 0.01

    def __post_init__(self) -> None:
        if self.n_bits not in (2, 3, 4, 8):
            raise ValueError(
                f"n_bits must be in {{2, 3, 4, 8}}; got {self.n_bits}"
            )
        if self.block_size < 1:
            raise ValueError(
                f"block_size must be >= 1; got {self.block_size}"
            )
        if self.damp_percent <= 0.0:
            raise ValueError(
                f"damp_percent must be > 0; got {self.damp_percent}"
            )

    # ------------------------------------------------------------------
    # Derived quantisation constants
    # ------------------------------------------------------------------

    @property
    def q_min(self) -> int:
        """Minimum signed integer code for this bit-width."""
        return -(1 << (self.n_bits - 1))

    @property
    def q_max(self) -> int:
        """Maximum signed integer code for this bit-width."""
        return (1 << (self.n_bits - 1)) - 1


@dataclass
class GPTQStats:
    """Running statistics for a :class:`GPTQCalibrator` session.

    Attributes:
        total_calibrations: Number of :meth:`GPTQCalibrator.calibrate`
            invocations.
        total_columns: Cumulative number of weight columns quantised across
            all calibrate calls.
    """

    total_calibrations: int = 0
    total_columns:      int = 0


class GPTQCalibrator:
    """GPTQ-style Hessian-weighted column-wise quantisation calibrator.

    The calibrator computes an activation-covariance Hessian, applies
    diagonal damping, and uses the per-column Hessian diagonal entries as
    importance weights to select optimal quantised values column by column.

    Args:
        config: :class:`GPTQConfig` controlling bit-width, block size, and
            damping.
    """

    def __init__(self, config: GPTQConfig) -> None:
        self._config = config
        self._stats  = GPTQStats()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        weights:     np.ndarray,
        activations: np.ndarray,
    ) -> np.ndarray:
        """Quantise *weights* using GPTQ-style Hessian-weighted column rounding.

        Algorithm overview:

        1. Estimate Hessian ``H = X^T X / n`` from calibration activations.
        2. Damp ``H`` diagonal: ``H_ii += damp * mean(diag(H))``.
        3. Process weight columns in blocks of ``block_size``.  Within each
           block, quantise each column using its Hessian diagonal importance
           as a soft scale modifier, then compensate remaining columns in the
           block for accumulated error (diagonal approximation).

        Args:
            weights:     Float32 weight matrix of shape ``(rows, cols)``.
            activations: Float32 calibration activations of shape
                ``(n_samples, rows)``.  Row dimension must match ``weights``
                row count.

        Returns:
            Dequantised float32 weight matrix of shape ``(rows, cols)``.

        Raises:
            ValueError: If shapes are incompatible.
        """
        weights     = np.asarray(weights,     dtype=np.float32)
        activations = np.asarray(activations, dtype=np.float32)

        if weights.ndim != 2:
            raise ValueError(
                f"weights must be 2-D; got shape {weights.shape}"
            )
        rows, cols = weights.shape
        n_samples, act_rows = activations.shape
        if act_rows != rows:
            raise ValueError(
                f"activations column count ({act_rows}) must match weights "
                f"row count ({rows})"
            )

        cfg    = self._config
        q_min  = float(cfg.q_min)
        q_max  = float(cfg.q_max)

        # ── Hessian estimation ────────────────────────────────────────
        # H shape: (rows, rows)
        H: np.ndarray = (activations.T @ activations) / float(n_samples)

        # Damp diagonal
        diag_mean = float(np.mean(np.diag(H)))
        damp      = cfg.damp_percent * diag_mean
        H += damp * np.eye(rows, dtype=np.float32)

        # Per-column importance: diagonal of H maps to row importance.
        # For weight column j (rows,), the importance weight is diag(H).
        # Larger diagonal → that row's weights matter more → tighter quant.
        h_diag = np.diag(H).astype(np.float32)  # (rows,)

        # ── Column-wise quantisation ──────────────────────────────────
        W_q = weights.copy()

        n_blocks = (cols + cfg.block_size - 1) // cfg.block_size
        for b in range(n_blocks):
            col_start = b * cfg.block_size
            col_end   = min(col_start + cfg.block_size, cols)

            for j in range(col_start, col_end):
                w_col = W_q[:, j]  # (rows,)

                # Per-column scale weighted by Hessian diagonal importance.
                # Hessian-importance weighting: scale_j reflects the range
                # of (w * sqrt(h_diag)) so that high-importance rows are
                # quantised with finer granularity relative to their impact.
                w_scaled    = w_col * np.sqrt(np.maximum(h_diag, 1e-8))
                col_abs_max = float(np.max(np.abs(w_scaled)))
                if col_abs_max < 1e-12:
                    # Effectively zero column — leave as-is
                    continue

                scale = col_abs_max / q_max

                # Quantise and dequantise (round-to-nearest)
                q_col = np.clip(np.round(w_scaled / scale), q_min, q_max)
                w_dq  = (q_col * scale) / np.sqrt(np.maximum(h_diag, 1e-8))

                # Error for this column; propagate to remaining columns in
                # block using diagonal Hessian approximation (no off-diagonal
                # correction — diagonal GPTQ variant).
                err = w_col - w_dq
                W_q[:, j] = w_dq

                # Compensate subsequent columns in this block
                if j + 1 < col_end:
                    # h_diag-weighted error propagation
                    h_ratio = h_diag / np.maximum(h_diag + 1e-8, 1e-8)
                    # Scale the correction by the per-row importance
                    correction = (err * h_ratio)[:, np.newaxis]  # (rows, 1)
                    W_q[:, j + 1:col_end] -= correction

        self._stats.total_calibrations += 1
        self._stats.total_columns      += cols

        return W_q.astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> GPTQStats:
        """Running calibration statistics."""
        return self._stats
