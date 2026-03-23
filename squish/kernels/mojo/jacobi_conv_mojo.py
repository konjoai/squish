"""squish/kernels/mojo/jacobi_conv_mojo.py — Mojo-backed Jacobi convergence check.

Wraps the ``jacobi_convergence`` Mojo kernel via MojoBridge with a NumPy
fallback.  Performs per-position argmax or Gumbel-max sampling over
vocabulary logits to generate the next Jacobi iteration's token guesses,
then counts converged positions.

Reference: Santilli et al., "Accelerating Transformer Inference for
Translation via Parallel Decoding," ACL 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "JacobiConvMojoConfig",
    "MojoJacobiConv",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("jacobi_convergence")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_check(
    pos_logits: np.ndarray,
    guesses: np.ndarray,
    temperature: float,
    seed: int,
) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    if temperature <= 0.0:
        new_g = pos_logits.argmax(axis=1).astype(np.int32)
    else:
        scaled = pos_logits / max(temperature, 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-9
        new_g = np.array(
            [rng.choice(pos_logits.shape[1], p=probs[i]) for i in range(len(probs))],
            dtype=np.int32,
        )
    return new_g, int((new_g == guesses).sum())


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class JacobiConvMojoConfig:
    """Configuration for :class:`MojoJacobiConv`.

    Attributes:
        temperature: Sampling temperature (0 = greedy argmax).
    """

    temperature: float = 0.0


class MojoJacobiConv:
    """Mojo-backed Jacobi fixed-point convergence checker.

    SIMD-vectorised argmax / Gumbel-max over vocabulary per position,
    ``parallelize`` over N positions.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[JacobiConvMojoConfig] = None) -> None:
        self._cfg = config or JacobiConvMojoConfig()

    def check(
        self,
        pos_logits: np.ndarray,
        guesses: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """Generate updated token guesses and count converged positions.

        Args:
            pos_logits:  Per-position logits ``(N, vocab)`` float32.
            guesses:     Previous guesses ``(N,)`` int32.
            temperature: Sampling temperature (None → config, 0 = greedy).
            seed:        RNG seed.

        Returns:
            ``(new_guesses (N,) int32, n_fixed int)``.
        """
        logits = np.ascontiguousarray(pos_logits, dtype=np.float32)
        g = np.ascontiguousarray(guesses, dtype=np.int32).ravel()
        temp = float(temperature) if temperature is not None else self._cfg.temperature
        if _kernel is not None:
            n, vocab = logits.shape
            new_g = np.zeros(n, dtype=np.int32)
            n_fixed_buf = np.zeros(1, dtype=np.int32)
            _kernel(
                logits.ctypes.data, g.ctypes.data, new_g.ctypes.data,
                n_fixed_buf.ctypes.data, n, vocab, temp, int(seed),
            )
            return new_g, int(n_fixed_buf[0])
        return _numpy_check(logits, g, temp, int(seed))

    def converged(
        self,
        pos_logits: np.ndarray,
        guesses: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> bool:
        """Return True when all positions have converged.

        Calls :meth:`check` and returns ``True`` iff ``n_fixed == N``.

        Args:
            pos_logits:  ``(N, vocab)`` float32.
            guesses:     ``(N,)`` int32.
            temperature: Sampling temperature (None → config).
            seed:        RNG seed.

        Returns:
            ``True`` if every position's guess matches the sampled token.
        """
        _, n_fixed = self.check(pos_logits, guesses, temperature=temperature, seed=seed)
        return n_fixed == pos_logits.shape[0]

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
