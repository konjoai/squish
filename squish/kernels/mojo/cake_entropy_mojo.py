"""squish/kernels/mojo/cake_entropy_mojo.py — Mojo-backed CAKE per-head entropy.

CAKE (Entropy-Aware KV Cache Eviction) computes normalised attention-entropy
per head over a sliding observation window.  When the Mojo kernel is not
available the wrapper falls back transparently to a NumPy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["CakeEntropyMojoConfig", "MojoCakeEntropy"]

_bridge = MojoBridge()
_cake_entropy_kernel = _bridge.load_kernel("cake_entropy")


@dataclass
class CakeEntropyMojoConfig:
    """Configuration for :class:`MojoCakeEntropy`.

    Attributes:
        n_heads:     Number of attention heads.
        head_dim:    Head dimension (key/query vectors).
        obs_window:  Number of observation queries used to estimate entropy.
        temperature: Softmax temperature for score normalisation.
    """

    n_heads: int = 32
    head_dim: int = 128
    obs_window: int = 4
    temperature: float = 1.0


class MojoCakeEntropy:
    """Mojo-accelerated per-head CAKE entropy computation.

    For each head *h* the method selects ``obs_window`` query positions,
    computes softmax attention scores against all *T* key tokens and returns
    the normalised Shannon entropy averaged over the observation queries.

    Args:
        config: :class:`CakeEntropyMojoConfig` controlling n_heads, head_dim,
                obs_window and temperature.

    Example::

        ce = MojoCakeEntropy(CakeEntropyMojoConfig(n_heads=4, head_dim=8))
        q_obs = np.random.randn(2, 4, 8).astype(np.float32)   # (obs, H, D)
        k     = np.random.randn(16, 4, 8).astype(np.float32)  # (T,  H, D)
        entropies = ce.compute(q_obs, k)  # (4,)
    """

    def __init__(self, config: CakeEntropyMojoConfig) -> None:
        self._config = config
        self._kernel = _cake_entropy_kernel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        q_obs: np.ndarray,
        k: np.ndarray,
        n_heads: Optional[int] = None,
        obs_window: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Return normalised per-head entropy over the observation window.

        Args:
            q_obs:       Query observations, shape ``(obs_window, n_heads, head_dim)``
                         or ``(obs_window * n_heads, head_dim)`` (both accepted).
            k:           Key cache, shape ``(T, n_heads, head_dim)``
                         or ``(T * n_heads, head_dim)`` (both accepted).
            n_heads:     Override; defaults to ``config.n_heads``.
            obs_window:  Override; defaults to ``config.obs_window``.
            temperature: Override; defaults to ``config.temperature``.

        Returns:
            Float32 array of shape ``(n_heads,)`` with normalised entropy in
            [0, 1] for each head.
        """
        nH = n_heads if n_heads is not None else self._config.n_heads
        ow = obs_window if obs_window is not None else self._config.obs_window
        temp = temperature if temperature is not None else self._config.temperature
        hd = self._config.head_dim

        q_obs = np.asarray(q_obs, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)

        # Normalise to 3-D: (obs, n_heads, head_dim)
        if q_obs.ndim == 2:
            q_obs = q_obs.reshape(ow, nH, hd)
        # Normalise to 3-D: (T, n_heads, head_dim)
        if k.ndim == 2:
            T = k.shape[0] // nH
            k = k.reshape(T, nH, hd)

        if self._kernel is not None:
            try:
                return self._kernel(q_obs, k, nH, ow, temp)
            except Exception:
                pass

        return self._numpy_entropy(q_obs, k, nH, ow, temp)

    def n_heads(self) -> int:
        """Return configured number of heads."""
        return self._config.n_heads

    def head_dim(self) -> int:
        """Return configured head dimension."""
        return self._config.head_dim

    def obs_window(self) -> int:
        """Return observation window size."""
        return self._config.obs_window

    def temperature(self) -> float:
        """Return softmax temperature."""
        return self._config.temperature

    def backend(self) -> str:
        """Return ``'mojo'`` when the kernel is loaded, else ``'numpy'``."""
        return "mojo" if self._kernel is not None else "numpy"

    # ------------------------------------------------------------------
    # NumPy fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _numpy_entropy(
        q_obs: np.ndarray,
        k: np.ndarray,
        n_heads: int,
        obs_window: int,
        temperature: float,
    ) -> np.ndarray:
        """Pure-NumPy reference: normalised Shannon entropy per head.

        Args:
            q_obs: ``(obs_window, n_heads, head_dim)`` float32.
            k:     ``(T,         n_heads, head_dim)`` float32.
            n_heads:    Number of heads.
            obs_window: Number of observation queries.
            temperature: Softmax temperature.

        Returns:
            Float32 array ``(n_heads,)``.
        """
        T = k.shape[0]
        hd = q_obs.shape[-1]
        inv_scale = 1.0 / (math.sqrt(hd) * temperature)

        entropies = np.zeros(n_heads, dtype=np.float32)
        for h in range(n_heads):
            head_ent = 0.0
            for q_pos in range(obs_window):
                q_vec = q_obs[q_pos, h]          # (head_dim,)
                scores = (k[:, h, :] @ q_vec) * inv_scale  # (T,)
                scores -= scores.max()            # numerical stability
                exp_s = np.exp(scores, dtype=np.float64)
                p = exp_s / exp_s.sum()
                # Shannon entropy, normalised by ln(T)
                ent = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
                if T > 1:
                    ent /= math.log(T)
                head_ent += ent
            entropies[h] = head_ent / obs_window

        return entropies.astype(np.float32)


import math  # noqa: E402  (placed after class to keep top of file clean)
