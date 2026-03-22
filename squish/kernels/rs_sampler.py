"""Rust-backed fused sampler kernel: softmax · top-p · min-p.

Wraps ``squish_quant.{softmax_logits,top_p_filter,min_p_filter}_f32`` from the
maturin-compiled Rust extension.  Falls back to a pure-NumPy implementation when
the extension is unavailable.

All operations work on 1-D logit / probability arrays (vocabulary size).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.random import default_rng

__all__ = ["SamplerKernelConfig", "RustSamplerKernel"]

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class SamplerKernelConfig:
    """Configuration for :class:`RustSamplerKernel`.

    Attributes
    ----------
    temperature:
        Logit temperature scale (1.0 = no scaling).
    top_p:
        Nucleus sampling probability mass (0–1].
    min_p:
        Min-P filter threshold: drop tokens with ``p < min_p * p_max``.
    seed:
        Seed for NumPy RNG (used only in the NumPy fallback path).
    """

    temperature: float = 1.0
    top_p: float = 0.9
    min_p: float = 0.0
    seed: int = 0


class RustSamplerKernel:
    """Fused sampling kernel (softmax + top-p + min-p) backed by Rust.

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`SamplerKernelConfig`.
    """

    def __init__(self, config: SamplerKernelConfig | None = None) -> None:
        self.config = config or SamplerKernelConfig()
        self._rng = default_rng(self.config.seed)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute numerically stable softmax probabilities.

        Parameters
        ----------
        logits:
            1-D float32 array of shape ``(vocab_size,)``.

        Returns
        -------
        float32 probability array of the same shape.
        """
        logits_f32 = logits.astype(np.float32, copy=False).ravel()

        if self.config.temperature != 1.0:
            logits_f32 = logits_f32 / max(self.config.temperature, 1e-8)

        if _RUST_AVAILABLE:
            try:
                return _sq.softmax_logits_f32(logits_f32)
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_softmax(logits_f32)

    def top_p_filter(self, probs: np.ndarray, p: float | None = None) -> np.ndarray:
        """Apply top-p (nucleus) filter to a probability distribution.

        Parameters
        ----------
        probs:
            1-D float32 probability array (should sum to 1).
        p:
            Nucleus probability mass.  Defaults to ``config.top_p``.

        Returns
        -------
        Re-normalised float32 array with low-probability tokens zeroed.
        """
        p_val = self.config.top_p if p is None else p
        probs_f32 = probs.astype(np.float32, copy=False).ravel()

        if _RUST_AVAILABLE:
            try:
                return _sq.top_p_filter_f32(probs_f32, float(p_val))
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_top_p(probs_f32, float(p_val))

    def min_p_filter(self, probs: np.ndarray, p_min: float | None = None) -> np.ndarray:
        """Apply min-p filter to a probability distribution.

        Parameters
        ----------
        probs:
            1-D float32 probability array.
        p_min:
            Threshold: tokens with ``prob < p_min * max_prob`` are zeroed.
            Defaults to ``config.min_p``.

        Returns
        -------
        Re-normalised float32 array.
        """
        p_val = self.config.min_p if p_min is None else p_min
        probs_f32 = probs.astype(np.float32, copy=False).ravel()

        if _RUST_AVAILABLE:
            try:
                return _sq.min_p_filter_f32(probs_f32, float(p_val))
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_min_p(probs_f32, float(p_val))

    def sample(self, logits: np.ndarray) -> int:
        """Full fused pipeline: temperature → softmax → top-p → min-p → sample.

        Parameters
        ----------
        logits:
            1-D float32 logit array of shape ``(vocab_size,)``.

        Returns
        -------
        Sampled token index.
        """
        probs = self.softmax(logits)
        if self.config.top_p < 1.0:
            probs = self.top_p_filter(probs)
        if self.config.min_p > 0.0:
            probs = self.min_p_filter(probs)

        # Multinomial draw
        total = probs.sum()
        if total <= 0:
            return int(self._rng.integers(len(probs)))
        probs_norm = probs / total
        cumpr = np.cumsum(probs_norm)
        u = self._rng.random()
        idx = int(np.searchsorted(cumpr, u))
        return min(idx, len(probs) - 1)

    # ------------------------------------------------------------------ #
    #  NumPy fallback implementations                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        exp_vals = np.exp(shifted)
        return (exp_vals / exp_vals.sum()).astype(np.float32)

    @staticmethod
    def _numpy_top_p(probs: np.ndarray, p: float) -> np.ndarray:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        # Keep tokens until cumulative mass exceeds p
        mask_sorted = np.zeros(len(probs), dtype=bool)
        mask_sorted[sorted_idx] = True
        cutoff = np.searchsorted(cumsum, p, side="right")
        keep_idx = sorted_idx[: cutoff + 1]
        out = np.zeros_like(probs)
        out[keep_idx] = probs[keep_idx]
        total = out.sum()
        if total > 1e-10:
            out /= total
        return out

    @staticmethod
    def _numpy_min_p(probs: np.ndarray, min_p: float) -> np.ndarray:
        threshold = min_p * probs.max()
        out = np.where(probs >= threshold, probs, 0.0).astype(np.float32)
        if out.max() == 0:
            best = int(probs.argmax())
            out[best] = probs[best]
        total = out.sum()
        if total > 1e-10:
            out /= total
        return out
