"""squish/kernels/rs_adaround.py — Rust-backed AdaRound optimisation step.

Wraps ``squish_quant_rs.adaround_step_f32`` with a NumPy fallback.

Reference: Nagel et al., "Up or Down? Adaptive Rounding for Post-Training
Quantization." ICML 2020 (arXiv 2004.10568).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "AdaRoundConfig",
    "RustAdaRound",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "adaround_step_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_step(
    v: np.ndarray,
    w: np.ndarray,
    w_quant_floor: np.ndarray,
    q_scale: np.ndarray,
    lr: float,
    beta: float,
    lambda_reg: float,
    zeta: float,
    gamma: float,
) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-(beta * (v - zeta))))
    h = np.clip(sig, 0.0, 1.0)
    w_soft = (w_quant_floor + h) * q_scale
    grad_recon = (w_soft - w) * q_scale
    h_prime = beta * h * (1.0 - h) * (gamma - zeta)
    grad_reg = lambda_reg * h_prime
    return (v - lr * (grad_recon + grad_reg)).astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class AdaRoundConfig:
    """Configuration for :class:`RustAdaRound`.

    Attributes:
        lr:          Learning rate.
        lambda_reg:  Regularisation coefficient.
        zeta:        Lower stretch bound (default −0.1 → we use 0.1 for abs).
        gamma:       Upper stretch bound (default 1.1).
    """

    lr: float = 1e-3
    lambda_reg: float = 0.01
    zeta: float = 0.1
    gamma: float = 1.1


class RustAdaRound:
    """Rust-accelerated AdaRound single optimisation step.

    Vectorises the rounding-parameter gradient update over all weight
    elements in parallel.  Falls back to NumPy when ``squish_quant_rs``
    is unavailable.
    """

    def __init__(self, config: Optional[AdaRoundConfig] = None) -> None:
        self._cfg = config or AdaRoundConfig()

    def step(
        self,
        v: np.ndarray,
        w: np.ndarray,
        w_quant_floor: np.ndarray,
        q_scale: np.ndarray,
        beta: float,
        lr: Optional[float] = None,
        lambda_reg: Optional[float] = None,
    ) -> np.ndarray:
        """Compute one AdaRound gradient update.

        Args:
            v:             Rounding parameters (flat float32, len = N).
            w:             Original weights (flat float32, len = N).
            w_quant_floor: Floor-quantized weights (flat float32, len = N).
            q_scale:       Per-element quantisation scale (float32, len = N).
            beta:          Current annealing β.
            lr:            Override config learning rate.
            lambda_reg:    Override config regularisation coefficient.

        Returns:
            Updated V vector ``(N,)`` float32.
        """
        v_f = np.ascontiguousarray(v, dtype=np.float32).ravel()
        w_f = np.ascontiguousarray(w, dtype=np.float32).ravel()
        wf_f = np.ascontiguousarray(w_quant_floor, dtype=np.float32).ravel()
        qs_f = np.ascontiguousarray(q_scale, dtype=np.float32).ravel()
        if v_f.shape != w_f.shape or v_f.shape != wf_f.shape or v_f.shape != qs_f.shape:
            raise ValueError("v, w, w_quant_floor, q_scale must all have the same shape")
        lr_val = float(lr) if lr is not None else self._cfg.lr
        lam = float(lambda_reg) if lambda_reg is not None else self._cfg.lambda_reg
        zeta = self._cfg.zeta
        gamma = self._cfg.gamma
        if _HAS_RUST:
            res = _sq.adaround_step_f32(
                v_f, w_f, wf_f, qs_f, lr_val, float(beta), lam, zeta, gamma
            )
            return np.asarray(res, dtype=np.float32)
        return _numpy_step(v_f, w_f, wf_f, qs_f, lr_val, float(beta), lam, zeta, gamma)

    # ── properties ───────────────────────────────────────────────────────────

    def lr(self) -> float:
        """Default learning rate."""
        return self._cfg.lr

    def lambda_reg(self) -> float:
        """Default regularisation coefficient."""
        return self._cfg.lambda_reg

    def backend(self) -> str:
        """Return ``'rust'`` or ``'numpy'``."""
        return "rust" if _HAS_RUST else "numpy"
