"""squish/quant/ada_round.py

AdaRound — Adaptive Rounding for Post-Training Weight Quantization.

AdaRound learns the optimal rounding decision for every weight element by
optimising a continuous relaxation.  Instead of always rounding to nearest,
each weight can round *up* or *down* based on how much that decision reduces
the activation reconstruction error on a small calibration set.

The V parameter is a real-valued matrix (same shape as W).  The
soft-quantized weight at each optimisation step is:

    W_soft_q = floor(W / delta) * delta + delta * h(V)

where ``h(V) = clip(sigmoid(V) * (ζ − γ) + γ, 0, 1)`` is the stretched
sigmoid from the original paper, and ``delta`` is the per-channel step size.

After calibration, the hard rounding decision is:

    W_hard_q = (floor(W / delta) + (sigmoid(V) > 0.5)) * delta

Reference
---------
Nagel, M. et al. "Up or Down? Adaptive Rounding for Post-Training
Quantization." ICML 2020. arXiv:2004.10568.
"""

from __future__ import annotations

__all__ = ["AdaRoundConfig", "AdaRoundState", "AdaRoundQuantizer"]

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AdaRoundConfig:
    """Configuration for AdaRoundQuantizer.

    Parameters
    ----------
    bits:
        Weight quantization bit-width (e.g. 4).
    n_iters:
        Number of gradient descent iterations during calibration.
    lr:
        Learning rate for optimising V.
    beta_warmup:
        Fraction of n_iters for the first β half of the annealing schedule.
        β anneals from 20 → 2 over ``beta_warmup * n_iters`` steps and
        stays at 2 after that (controls regularisation sharpness).
    seed:
        RNG seed.
    """

    bits: int = 4
    n_iters: int = 500
    lr: float = 1e-2
    beta_warmup: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        if self.n_iters < 1:
            raise ValueError("n_iters must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 < self.beta_warmup < 1.0:
            raise ValueError("beta_warmup must be in (0, 1)")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class AdaRoundState:
    """Mutable optimisation state for AdaRound.

    Attributes
    ----------
    V:
        Continuous relaxation variable, same shape as W.  Values are real
        numbers; ``sigmoid(V) > 0.5`` gives the hard round-up decision.
    n_iters_done:
        Number of calibration iterations completed.
    """

    V: ndarray
    n_iters_done: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: ndarray) -> ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _soft_round(V: ndarray, beta: float = 20.0) -> ndarray:
    """Stretched sigmoid h(V; b) ∈ [0, 1]."""
    # h(V) = clip(sigmoid(V * b) * (zeta - gamma) + gamma, 0, 1)
    # We use zeta=1.1, gamma=-0.1 (from the original paper)
    zeta, gamma = 1.1, -0.1
    return np.clip(_sigmoid(V) * (zeta - gamma) + gamma, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class AdaRoundQuantizer:
    """Post-training quantizer using adaptive rounding (AdaRound).

    Parameters
    ----------
    config:
        ``AdaRoundConfig`` instance.
    """

    def __init__(self, config: AdaRoundConfig | None = None) -> None:
        self.config = config or AdaRoundConfig()

    def _compute_delta(self, W: ndarray) -> ndarray:
        """Per-output-channel quantization step size delta."""
        q_levels = float((1 << self.config.bits) - 1)
        w_max = np.abs(W).max(axis=1, keepdims=True)  # (out, 1)
        delta = 2.0 * w_max / q_levels
        return np.where(delta > 1e-8, delta, np.ones_like(delta) * 1e-8)

    def new_state(self, W: ndarray) -> AdaRoundState:
        """Initialise V to 0 (neutral: round to nearest on first hard_round).

        Parameters
        ----------
        W:
            Weight matrix, shape ``(out_features, in_features)``.
        """
        W = np.asarray(W, dtype=np.float32)
        V = np.zeros(W.shape, dtype=np.float32)
        return AdaRoundState(V=V, n_iters_done=0)

    def hard_round(
        self, W: ndarray, state: AdaRoundState
    ) -> Tuple[ndarray, ndarray]:
        """Apply hard rounding decisions learned in V.

        Parameters
        ----------
        W:
            Original float weight matrix.
        state:
            ``AdaRoundState`` with learned V.

        Returns
        -------
        W_rounded:
            Rounded weight matrix, float32, same shape as W.
        scales:
            Per-output-channel scale (delta), shape ``(out_features,)``.
        """
        W = np.asarray(W, dtype=np.float32)
        delta = self._compute_delta(W)
        # Hard decision: round up if sigmoid(V) > 0.5
        up = (_sigmoid(state.V) > 0.5).astype(np.float32)
        W_rounded = (np.floor(W / delta) + up) * delta
        return W_rounded.astype(np.float32), delta.squeeze(axis=1).astype(np.float32)

    def calibrate(
        self, W: ndarray, X_calib: ndarray, state: AdaRoundState
    ) -> AdaRoundState:
        """Optimise V to minimise activation reconstruction error.

        Minimises:
            L = ||X_calib @ W.T − X_calib @ W_soft_q.T||_F^2
              + λ_reg * Σ_ij (1 − |2h(V_ij) − 1|^β)

        Parameters
        ----------
        W:
            Weight matrix, shape ``(out_features, in_features)``.
        X_calib:
            Calibration activations, shape ``(n_samples, in_features)``.
        state:
            Initial ``AdaRoundState`` (typically from :meth:`new_state`).

        Returns
        -------
        Updated ``AdaRoundState`` with optimised V.
        """
        W = np.asarray(W, dtype=np.float32)
        X_calib = np.asarray(X_calib, dtype=np.float32)
        delta = self._compute_delta(W)  # (out, 1)

        V = state.V.copy()
        ref_output = X_calib @ W.T  # (n_samples, out)

        lr = float(self.config.lr)
        n_iters = self.config.n_iters
        beta_steps = max(1, int(n_iters * self.config.beta_warmup))

        for t in range(n_iters):
            # Anneal beta from 20 -> 2
            beta = 20.0 - 18.0 * min(1.0, t / beta_steps)

            h = _soft_round(V)
            W_soft_q = (np.floor(W / delta) + h) * delta  # (out, in)

            residual = X_calib @ W_soft_q.T - ref_output  # (n_samples, out)

            # Gradient wrt W_soft_q: dL/dW_soft_q_ij = 2 * sum_k(residual_ki * X_calib_kj)
            # Then chain rule through W_soft_q → h → V
            # dL/dh_ij = (2/n) * sum_k residual_ki * X_calib_kj * delta_i
            n_s = X_calib.shape[0]
            grad_W_q = (2.0 / n_s) * X_calib.T @ residual  # (in, out)
            grad_h = (grad_W_q * delta.T).T  # (out, in), multiplied by delta

            # Regularisation: λ*(1 - |2h-1|^β)  → gradient wrt h
            # d/dh: λ * β * |2h-1|^(β-1) * sign(2h-1) * 2
            lam = 0.01
            abs_2h1 = np.abs(2 * h - 1)
            reg_grad_h = lam * beta * np.where(
                abs_2h1 > 0,
                abs_2h1 ** (beta - 1.0) * np.sign(2 * h - 1) * 2.0,
                0.0,
            )

            # Chain through soft_round: dh/dV via stretched sigmoid derivative
            sig = _sigmoid(V)
            zeta, gamma = 1.1, -0.1
            dh_dV = sig * (1.0 - sig) * (zeta - gamma)

            total_grad = (grad_h + reg_grad_h) * dh_dV
            V -= lr * total_grad

        return AdaRoundState(V=V, n_iters_done=state.n_iters_done + n_iters)

    def quantize(
        self, W: ndarray, state: AdaRoundState
    ) -> Tuple[ndarray, ndarray]:
        """Apply hard rounding to produce the final quantized weights.

        Parameters
        ----------
        W:
            Original float weight matrix.
        state:
            Calibrated ``AdaRoundState``.

        Returns
        -------
        W_q:
            AdaRound-quantized weights, float32.
        scales:
            Per-output-channel delta values.
        """
        return self.hard_round(W, state)
