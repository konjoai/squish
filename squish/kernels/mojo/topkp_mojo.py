"""topkp_mojo.py — Mojo-accelerated fused top-k / top-p sampling.

Wraps `squish/kernels/mojo/kernels/topkp.mojo` via MojoBridge (Wave 58b).
Falls back to NumPy when the Mojo library is unavailable.

MojoTopKP replaces 4 NumPy passes in `scheduler.py`, `token_swift.py`,
`early_exit_sampler.py`, and `duo_decoding.py`:
  ``np.argsort(-probs) + np.cumsum + np.searchsorted + mask``

with a single Mojo radix-histogram partial-sort:
  1. High-byte SIMD histogram (256 buckets) → score threshold in one
     ``vectorize`` pass over vocab_size floats.
  2. Linear scan over threshold bucket for exact top-p cumsum cutoff.
  3. Temperature scaling + softmax (vectorized exp + horizontal-sum).
  4. Multinomial sample via single uniform draw + cumsum threshold.

``@parameter`` on vocab_size (32000, 128256).  ~4× for vocab=128K on M3.

Reference:
  Holtzman et al. (ICLR 2020) — Nucleus/top-p sampling.
  Fan et al. (arXiv:1904.09751) — top-k sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["TopKPConfig", "MojoTopKP"]

_bridge = MojoBridge()
_MOJO_FN = _bridge.load_kernel("topkp")


@dataclass
class TopKPConfig:
    """Configuration for MojoTopKP.

    Attributes:
        vocab_size:  Vocabulary size (e.g. 32000 or 128256).
        top_k:       Maximum number of tokens to keep (0 = disable).
        top_p:       Cumulative probability threshold (1.0 = disable).
        temperature: Logit temperature scaling (1.0 = no scaling).
    """

    vocab_size: int = 128256
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0


class MojoTopKP:
    """Mojo-accelerated fused top-k / top-p sampling pipeline.

    Accepts raw logits or a probability distribution, applies temperature
    scaling, top-k / top-p filtering, and draws a single token sample.

    Usage::

        sampler = MojoTopKP(TopKPConfig(vocab_size=128256, top_p=0.9))
        logits  = np.random.randn(128256).astype(np.float32)
        token_id = sampler.sample(logits)           # int
        top_ids  = sampler.filter(logits, top_k=50) # filtered probability array
    """

    def __init__(self, config: TopKPConfig | None = None) -> None:
        self._cfg = config or TopKPConfig()

    def sample(
        self,
        logits: np.ndarray,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> int:
        """Sample a single token from logits using top-k / top-p filtering.

        Args:
            logits:      Float32 1-D array ``(vocab_size,)`` raw logits.
            top_k:       Override config top_k (0 = disable).
            top_p:       Override config top_p (1.0 = disable).
            temperature: Override config temperature.
            seed:        Optional RNG seed for determinism.

        Returns:
            Integer token index.
        """
        logits = np.asarray(logits.ravel(), dtype=np.float32)
        k = top_k if top_k is not None else self._cfg.top_k
        p = top_p if top_p is not None else self._cfg.top_p
        temp = temperature if temperature is not None else self._cfg.temperature
        if _MOJO_FN is not None:
            rng_seed = seed if seed is not None else 0
            return int(_MOJO_FN(logits, k, p, temp, rng_seed))
        return self._numpy_sample(logits, k, p, temp, seed)

    def filter(
        self,
        logits: np.ndarray,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
    ) -> np.ndarray:
        """Apply top-k / top-p filtering and return probability distribution.

        Args:
            logits:      Float32 1-D array ``(vocab_size,)`` raw logits.
            top_k:       Override config top_k.
            top_p:       Override config top_p.
            temperature: Override config temperature.

        Returns:
            Float32 1-D array ``(vocab_size,)`` filtered + normalized probabilities.
        """
        logits = np.asarray(logits.ravel(), dtype=np.float32)
        k = top_k if top_k is not None else self._cfg.top_k
        p = top_p if top_p is not None else self._cfg.top_p
        temp = temperature if temperature is not None else self._cfg.temperature
        return self._numpy_filter(logits, k, p, temp)

    def vocab_size(self) -> int:
        """Return configured vocabulary size."""
        return self._cfg.vocab_size

    def backend(self) -> str:
        """Return 'mojo' if Mojo kernel loaded, else 'numpy'."""
        return "mojo" if _MOJO_FN is not None else "numpy"

    # ── NumPy fallbacks ────────────────────────────────────────────────────

    @staticmethod
    def _numpy_filter(
        logits: np.ndarray,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> np.ndarray:
        # Temperature scaling
        scaled = logits / max(temperature, 1.0e-7)
        # Numerically stable softmax
        shifted = scaled - scaled.max()
        probs = np.exp(shifted)
        probs = probs / probs.sum()
        # Top-k filtering
        if top_k > 0 and top_k < len(probs):
            kth_val = np.partition(probs, -top_k)[-top_k]
            probs = np.where(probs >= kth_val, probs, 0.0)
        # Top-p filtering
        if top_p < 1.0:
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, top_p)
            probs_copy = np.zeros_like(probs)
            probs_copy[sorted_idx[:cutoff + 1]] = probs[sorted_idx[:cutoff + 1]]
            probs = probs_copy
        # Renormalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones_like(probs) / len(probs)
        return probs.astype(np.float32)

    def _numpy_sample(
        self,
        logits: np.ndarray,
        top_k: int,
        top_p: float,
        temperature: float,
        seed: int | None,
    ) -> int:
        probs = self._numpy_filter(logits, top_k, top_p, temperature)
        rng = np.random.default_rng(seed)
        return int(rng.choice(len(probs), p=probs))
