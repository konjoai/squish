"""
squish/token/deja_vu_sparse.py

DejaVuSparseFFN — Activation-Sparsity Predictor for FFN Compute Reduction.

Based on:
  "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time"
  Liu et al. — ICML 2023  —  arXiv:2310.17157

  "DejaVu+: Exploiting Predictable Attention and FFN Sparsity for Fast LLM
   Inference" — Extended version with adaptive thresholds — 2025

Background
----------
In transformer FFN (feed-forward) blocks:

    output = act(x @ W1) @ W2    (SwiGLU or GELU)

At each step, only a **sparse subset of neurons** (typically 30–70%) has
non-negligible activations after the activation function.  The "dead"
neurons could be skipped entirely.

Deja Vu trains a lightweight predictor (a small MLP or linear classifier)
that, given the current hidden state x, predicts *which neurons will be
active before running the expensive W1 matmul*.  If the predictor is
accurate, we skip the matmul rows/columns for predicted-inactive neurons.

Result: 25–50% FFN FLOP reduction with < 0.5% quality loss (ICML 2023).

This module provides:
  - ``FFNPredictor``: a lightweight 1-hidden-layer classifier that predicts
    binary neuron activation given a hidden state vector.
  - ``DejaVuSparseFFN``: wraps any FFN callable, applies predictor masking,
    and tracks savings.

Training/Calibration:
  - An offline calibration step collects (hidden_state, activations) pairs.
  - ``calibrate()`` trains the predictor via logistic regression-style
    mini-batch gradient descent.

Classes
-------
``DejaVuConfig``          — predictor dims, threshold, update frequency
``FFNPredictor``          — lightweight binary activation predictor
``DejaVuStats``           — call counts, sparsity ratio, savings
``DejaVuSparseFFN``       — wrapper that applies predictor at forward time

Usage::

    from squish.token.deja_vu_sparse import DejaVuConfig, DejaVuSparseFFN
    import numpy as np

    def my_ffn(x):  # x: (hidden,) → (hidden,)
        w1 = np.eye(64)  # stub
        return np.maximum(x @ w1, 0)

    cfg = DejaVuConfig(hidden_size=64, ffn_size=64, predictor_hidden=32)
    dv = DejaVuSparseFFN(cfg, ffn_fn=my_ffn)

    # Calibrate with sample data
    samples = np.random.randn(100, 64).astype(np.float32)
    dv.calibrate(samples)

    out = dv.forward(samples[0])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

__all__ = [
    "DejaVuConfig",
    "FFNPredictor",
    "DejaVuStats",
    "DejaVuSparseFFN",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DejaVuConfig:
    """Configuration for DejaVu contextual sparsity.

    Attributes:
        hidden_size:       Input hidden state dimension.
        ffn_size:          FFN intermediate dimension (neurons to predict).
        predictor_hidden:  Hidden dimension of the predictor MLP.
        threshold:         Activation threshold — neurons with predicted
                           activation probability below this threshold are
                           skipped.  Default: 0.3.
        learning_rate:     Predictor learning rate during calibration.
        n_calibration_epochs: Epochs over calibration samples.
        seed:              RNG seed for weight initialisation.
    """

    hidden_size: int = 512
    ffn_size: int = 2048
    predictor_hidden: int = 128
    threshold: float = 0.3
    learning_rate: float = 1e-3
    n_calibration_epochs: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.ffn_size < 1:
            raise ValueError(f"ffn_size must be >= 1, got {self.ffn_size}")
        if self.predictor_hidden < 1:
            raise ValueError(f"predictor_hidden must be >= 1, got {self.predictor_hidden}")
        if not (0.0 <= self.threshold < 1.0):
            raise ValueError(f"threshold must be in [0, 1), got {self.threshold}")
        if self.n_calibration_epochs < 1:
            raise ValueError(
                f"n_calibration_epochs must be >= 1, got {self.n_calibration_epochs}"
            )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class FFNPredictor:
    """Lightweight 2-layer MLP that predicts binary neuron activation.

    Architecture:
        x: (hidden_size,)
        → Linear(hidden_size, predictor_hidden) + ReLU
        → Linear(predictor_hidden, ffn_size)    + Sigmoid
        → probabilities: (ffn_size,)

    Parameters
    ----------
    hidden_size:      Input dimension.
    ffn_size:         Output (per-neuron probabilities).
    predictor_hidden: MLP hidden width.
    seed:             Weight init seed.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        predictor_hidden: int,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale1 = (2.0 / hidden_size) ** 0.5
        scale2 = (2.0 / predictor_hidden) ** 0.5
        self.W1 = rng.normal(0, scale1, (hidden_size, predictor_hidden)).astype(np.float32)
        self.b1 = np.zeros(predictor_hidden, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (predictor_hidden, ffn_size)).astype(np.float32)
        self.b2 = np.zeros(ffn_size, dtype=np.float32)
        self._is_calibrated = False

    def predict_probs(self, hidden: np.ndarray) -> np.ndarray:
        """Predict neuron activation probabilities.

        Parameters
        ----------
        hidden: (hidden_size,) float32

        Returns
        -------
        probs: (ffn_size,) float32 in [0, 1]
        """
        h = _relu(hidden @ self.W1 + self.b1)
        return _sigmoid(h @ self.W2 + self.b2)

    def predict_mask(self, hidden: np.ndarray, threshold: float) -> np.ndarray:
        """Predict binary activation mask.

        Parameters
        ----------
        hidden:    (hidden_size,) float32
        threshold: Probability threshold.

        Returns
        -------
        mask: (ffn_size,) bool — True = neuron predicted active.
        """
        return self.predict_probs(hidden) >= threshold

    def train_step(
        self, hidden: np.ndarray, labels: np.ndarray, lr: float
    ) -> float:
        """One mini-batch gradient descent step (binary cross-entropy).

        Parameters
        ----------
        hidden: (batch, hidden_size)
        labels: (batch, ffn_size) float32 in {0, 1}
        lr:     Learning rate.

        Returns
        -------
        loss: Binary cross-entropy loss.
        """
        # Forward
        z1 = hidden @ self.W1 + self.b1  # (B, ph)
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2  # (B, ffn_size)
        probs = _sigmoid(z2)

        # BCE loss
        eps = 1e-7
        loss = -np.mean(
            labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)
        )

        # Backward
        batch = hidden.shape[0]
        d_z2 = (probs - labels) / batch  # (B, ffn_size)
        d_W2 = a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (z1 > 0).astype(np.float32)
        d_W1 = hidden.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

        return float(loss)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def __repr__(self) -> str:
        return (
            f"FFNPredictor("
            f"in={self.W1.shape[0]}, "
            f"hidden={self.W1.shape[1]}, "
            f"out={self.W2.shape[1]}, "
            f"calibrated={self._is_calibrated})"
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class DejaVuStats:
    """Runtime statistics for DejaVuSparseFFN.

    Attributes:
        total_forward_calls:    Number of ``forward()`` calls after calibration.
        total_neurons:          Total neuron evaluations (calls × ffn_size).
        total_neurons_skipped:  Neurons skipped due to sparsity prediction.
        calibration_loss:       Final calibration loss.
    """

    total_forward_calls: int = 0
    total_neurons: int = 0
    total_neurons_skipped: int = 0
    calibration_loss: float = float("inf")

    @property
    def mean_sparsity(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        return self.total_neurons_skipped / self.total_neurons

    @property
    def compute_saved_pct(self) -> float:
        return self.mean_sparsity * 100.0

    def __repr__(self) -> str:
        return (
            f"DejaVuStats("
            f"calls={self.total_forward_calls}, "
            f"sparsity={self.mean_sparsity:.2%}, "
            f"saved={self.compute_saved_pct:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DejaVuSparseFFN:
    """Apply contextual sparsity to an FFN function using a learned predictor.

    Parameters
    ----------
    config:
        DejaVu configuration.
    ffn_fn:
        The underlying FFN callable: ``(hidden: np.ndarray) → np.ndarray``.
        Should handle input of shape ``(hidden_size,)`` and return
        ``(hidden_size,)`` (same as input shape, as in residual-stream FFN).
    """

    def __init__(
        self,
        config: Optional[DejaVuConfig] = None,
        ffn_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._cfg = config or DejaVuConfig()
        self._ffn_fn = ffn_fn
        self._predictor = FFNPredictor(
            hidden_size=self._cfg.hidden_size,
            ffn_size=self._cfg.ffn_size,
            predictor_hidden=self._cfg.predictor_hidden,
            seed=self._cfg.seed,
        )
        self.stats = DejaVuStats()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        hidden_samples: np.ndarray,
        ffn_fn: Optional[Callable] = None,
    ) -> List[float]:
        """Train the predictor from sample hidden states.

        Parameters
        ----------
        hidden_samples:
            (N, hidden_size) float32 — sample hidden states from the target
            layer.
        ffn_fn:
            Optional override for the FFN callable (defaults to self._ffn_fn).
            Used to collect activation labels.

        Returns
        -------
        List of per-epoch BCE losses.
        """
        fn = ffn_fn or self._ffn_fn
        if fn is None:
            raise ValueError("ffn_fn must be provided for calibration")

        cfg = self._cfg
        X = np.asarray(hidden_samples, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]

        # Collect binary activation labels
        labels = np.zeros((len(X), cfg.ffn_size), dtype=np.float32)
        for i, x in enumerate(X):
            out = fn(x)
            # Treat any non-zero output as active (neuron contributes)
            labels[i] = (np.abs(out[:cfg.ffn_size]) > 0).astype(np.float32)

        losses: List[float] = []
        rng = np.random.default_rng(cfg.seed)
        for epoch in range(cfg.n_calibration_epochs):
            idx = rng.permutation(len(X))
            epoch_losses: List[float] = []
            batch_size = max(1, min(32, len(X)))
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_X = X[idx[start:end]]
                batch_y = labels[idx[start:end]]
                loss = self._predictor.train_step(batch_X, batch_y, cfg.learning_rate)
                epoch_losses.append(loss)
            losses.append(float(np.mean(epoch_losses)))

        self._predictor._is_calibrated = True
        self.stats.calibration_loss = losses[-1] if losses else float("inf")
        return losses

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, hidden: np.ndarray, ffn_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Run FFN with contextual sparsity masking.

        If the predictor is calibrated, neurons predicted inactive are zeroed
        in the output (simulating skipped compute).  If the predictor is not
        calibrated, falls back to the dense FFN.

        Parameters
        ----------
        hidden:  (hidden_size,) float32 input.
        ffn_fn:  Optional override for the FFN callable.

        Returns
        -------
        (hidden_size,) float32 output.
        """
        fn = ffn_fn or self._ffn_fn
        if fn is None:
            raise ValueError("ffn_fn must be provided")

        x = np.asarray(hidden, dtype=np.float32).ravel()
        cfg = self._cfg

        if not self._predictor.is_calibrated:
            return fn(x)

        # Predict which neurons to skip
        mask = self._predictor.predict_mask(x, cfg.threshold)  # (ffn_size,)

        # Run the dense FFN (in a real Metal implementation, only active rows
        # of W1/W2 would be computed; here we zero the inactive outputs)
        out = fn(x)
        out_arr = np.asarray(out, dtype=np.float32).ravel()
        ffn_out_size = min(len(out_arr), cfg.ffn_size)
        out_arr[:ffn_out_size] = out_arr[:ffn_out_size] * mask[:ffn_out_size]

        n_skipped = int((~mask[:ffn_out_size]).sum())
        self.stats.total_forward_calls += 1
        self.stats.total_neurons += ffn_out_size
        self.stats.total_neurons_skipped += n_skipped

        return out_arr

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def predictor(self) -> FFNPredictor:
        return self._predictor

    @property
    def is_calibrated(self) -> bool:
        return self._predictor.is_calibrated

    def __repr__(self) -> str:
        return (
            f"DejaVuSparseFFN("
            f"hidden={self._cfg.hidden_size}, "
            f"ffn={self._cfg.ffn_size}, "
            f"threshold={self._cfg.threshold}, "
            f"{self.stats})"
        )
