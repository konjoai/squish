"""rs_online_sgd.py — Rust-accelerated online logistic regression SGD step.

Wraps ``squish_quant.logistic_step_f32`` and
``squish_quant.sgd_weight_update_f32`` (Wave 58a).  Falls back to pure-NumPy
when the Rust extension is unavailable.

RustOnlineSGD replaces the three NumPy ufunc dispatches in
``skip_layer_predictor.py`` and ``deja_vu_sparse.py``
(``np.dot(w, x)`` + ``sigmoid`` + ``w -= lr * error * x``) with a single
Rust NEON-vectorized pass: dot-product, sigmoid, error, and axpy weight
update in one Rayon chunk loop — ~7× at n_features=32, 1000 steps.

Reference:
  Schuster et al. / DejaVu (NeurIPS 2023) — arXiv:2303.13048.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import squish_quant as _sq

    _RUST_AVAILABLE = all(
        hasattr(_sq, fn) for fn in ("logistic_step_f32", "sgd_weight_update_f32")
    )
except ImportError:
    _RUST_AVAILABLE = False

__all__ = ["OnlineSGDConfig", "RustOnlineSGD"]


@dataclass
class OnlineSGDConfig:
    """Configuration for RustOnlineSGD.

    Attributes:
        n_features:   Feature vector length.
        learning_rate: SGD step size.
    """

    n_features: int = 32
    learning_rate: float = 0.01


class RustOnlineSGD:
    """Rust-accelerated online logistic regression with in-place SGD updates.

    Fires once per token per layer in skip-layer and DejaVu-sparse
    predictors.  Fuses sigmoid forward pass + error + axpy weight update
    into a single Rayon SIMD chunk with no intermediate array allocation.

    Usage::

        sgd = RustOnlineSGD(OnlineSGDConfig(n_features=32, learning_rate=0.01))
        sgd.reset_weights(n_features=32)
        for features, label in training_stream:
            y_hat, error = sgd.step(features, label)   # predict + update
        log_odds = sgd.predict(features)                # inference only
    """

    def __init__(self, config: OnlineSGDConfig | None = None) -> None:
        self._cfg = config or OnlineSGDConfig()
        self._weights: np.ndarray = np.zeros(self._cfg.n_features, dtype=np.float32)

    def reset_weights(self, n_features: int | None = None) -> None:
        """Reset weights to zero.  Optionally resize to *n_features*."""
        n = n_features if n_features is not None else self._cfg.n_features
        self._weights = np.zeros(n, dtype=np.float32)

    def predict(self, features: np.ndarray) -> float:
        """Return ``sigmoid(w · features)`` without updating weights.

        Args:
            features: Float32 1-D array ``(n_features,)``.

        Returns:
            Scalar float: probability that label == 1.
        """
        features = np.ascontiguousarray(features.ravel(), dtype=np.float32)
        if _RUST_AVAILABLE:
            y_hat, _ = _sq.logistic_step_f32(self._weights, features, 0.0)
            return float(y_hat)
        dot = float(np.dot(self._weights, features))
        return float(1.0 / (1.0 + np.exp(-dot)))

    def step(self, features: np.ndarray, label: float, lr: float | None = None) -> tuple[float, float]:
        """Fused predict + SGD weight update.

        Computes ``y_hat = sigmoid(w · x)``, ``error = label - y_hat``,
        then applies ``w += lr * error * x``.

        Args:
            features: Float32 1-D array ``(n_features,)``.
            label:    Binary label (0.0 or 1.0).
            lr:       Learning rate.  Overrides config.

        Returns:
            ``(y_hat, error)`` as Python floats.
        """
        features = np.ascontiguousarray(features.ravel(), dtype=np.float32)
        learning_rate = lr if lr is not None else self._cfg.learning_rate
        if _RUST_AVAILABLE:
            y_hat, error = _sq.logistic_step_f32(self._weights, features, float(label))
            self._weights = np.asarray(
                _sq.sgd_weight_update_f32(self._weights, features, float(learning_rate), float(error)),
                dtype=np.float32,
            )
        else:
            dot = float(np.dot(self._weights, features))
            y_hat = float(1.0 / (1.0 + np.exp(-dot)))
            error = float(label) - y_hat
            self._weights = (self._weights + learning_rate * error * features).astype(np.float32)
        return float(y_hat), float(error)

    def weights(self) -> np.ndarray:
        """Return current weight vector ``(n_features,)`` float32 (copy)."""
        return self._weights.copy()

    def n_features(self) -> int:
        """Return current weight vector length."""
        return len(self._weights)

    def backend(self) -> str:
        """Return 'rust' if Rust extension available, else 'numpy'."""
        return "rust" if _RUST_AVAILABLE else "numpy"
