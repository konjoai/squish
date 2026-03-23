"""squish/kernels/rs_cake_entropy.py — Rust-backed CAKE attention entropy.

Wraps ``squish_quant_rs.cake_entropy_f32`` with a NumPy fallback.

Reference: Cai et al., "CAKE: Cascading and Adaptive KV Cache Eviction with
Layer-Budget Allocation for Efficient LLM Inference." arXiv 2404.09904.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "CakeEntropyConfig",
    "RustCakeEntropy",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "cake_entropy_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_entropy(
    q_obs: np.ndarray,      # (obs_window * n_heads, head_dim)
    k: np.ndarray,          # (T * n_heads, head_dim)
    n_heads: int,
    obs_window: int,
    temperature: float,
) -> np.ndarray:
    head_dim = q_obs.shape[1]
    t_len = k.shape[0] // n_heads
    scale = 1.0 / (np.sqrt(head_dim) * temperature)
    entropies = np.empty(n_heads, dtype=np.float32)
    for h in range(n_heads):
        acc = 0.0
        for q_pos in range(obs_window):
            q_row = q_obs[q_pos * n_heads + h]
            k_h = k[h::n_heads]  # (T, head_dim)
            scores = (k_h @ q_row) * scale
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            ent = -float(np.sum(weights * np.where(weights > 1e-9, np.log(weights + 1e-9), 0.0)))
            acc += ent / max(np.log(t_len), 1.0)
        entropies[h] = acc / max(obs_window, 1)
    return entropies


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class CakeEntropyConfig:
    """Configuration for :class:`RustCakeEntropy`.

    Attributes:
        n_heads:     Number of attention heads.
        head_dim:    Head dimension.
        obs_window:  Recent query positions used for entropy estimation.
        temperature: Softmax temperature (higher → more uniform attention).
    """

    n_heads: int = 32
    head_dim: int = 128
    obs_window: int = 4
    temperature: float = 1.0


class RustCakeEntropy:
    """Rust-accelerated CAKE per-head attention entropy computation.

    Computes mean normalised attention entropy over ``obs_window`` recent
    queries to guide KV-cache budget allocation.  Falls back to NumPy when
    ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[CakeEntropyConfig] = None) -> None:
        self._cfg = config or CakeEntropyConfig()

    def compute(
        self,
        q_obs: np.ndarray,
        k: np.ndarray,
        n_heads: Optional[int] = None,
        obs_window: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Compute per-head mean normalised attention entropy.

        Args:
            q_obs:       Recent query tensor ``(obs_window, n_heads, head_dim)``
                         or ``(obs_window * n_heads, head_dim)`` float32.
            k:           Key cache ``(T, n_heads, head_dim)`` or
                         ``(T * n_heads, head_dim)`` float32.
            n_heads:     Override config n_heads.
            obs_window:  Override config obs_window.
            temperature: Override config temperature.

        Returns:
            Per-head entropy ``(n_heads,)`` float32 in [0, 1].
        """
        nh = int(n_heads) if n_heads is not None else self._cfg.n_heads
        ow = int(obs_window) if obs_window is not None else self._cfg.obs_window
        temp = float(temperature) if temperature is not None else self._cfg.temperature
        hd = self._cfg.head_dim
        q_f = np.ascontiguousarray(q_obs, dtype=np.float32).reshape(ow * nh, hd)
        k_flat = k.shape[0] // nh if k.ndim == 3 else None
        if k.ndim == 3:
            k_f = np.ascontiguousarray(k, dtype=np.float32).reshape(-1, hd)
        else:
            k_f = np.ascontiguousarray(k, dtype=np.float32)
        if _HAS_RUST:
            raw = _sq.cake_entropy_f32(q_f, k_f, nh, ow, temp)
            return np.asarray(raw, dtype=np.float32)
        return _numpy_entropy(q_f, k_f, nh, ow, temp)

    # ── properties ───────────────────────────────────────────────────────────

    def n_heads(self) -> int:
        return self._cfg.n_heads

    def obs_window(self) -> int:
        return self._cfg.obs_window

    def temperature(self) -> float:
        return self._cfg.temperature

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
