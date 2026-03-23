"""squish/kernels/rs_jacobi_conv.py — Rust-backed Jacobi convergence check.

Wraps ``squish_quant_rs.jacobi_conv_check_f32`` with a NumPy fallback.

Jacobi decoding generates the next token guesses for all N speculative
positions in parallel by sampling (or argmax-ing) from per-position
logits, then reports how many positions have converged (guess unchanged).
Rayon parallelises the per-position sampling.

Reference: Santilli et al., "Accelerating Transformer Inference for
Translation via Parallel Decoding," ACL 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "JacobiConvConfig",
    "RustJacobiConv",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "jacobi_conv_check_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_jacobi_check(
    pos_logits: np.ndarray,
    guesses: np.ndarray,
    temperature: float,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """Argmax or temperature-sample per-position logits."""
    rng = np.random.default_rng(seed)
    if temperature <= 0.0:
        new_guesses = pos_logits.argmax(axis=1).astype(np.int32)
    else:
        scaled = pos_logits / max(temperature, 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-9
        new_guesses = np.array(
            [rng.choice(pos_logits.shape[1], p=probs[i]) for i in range(pos_logits.shape[0])],
            dtype=np.int32,
        )
    n_fixed = int((new_guesses == guesses).sum())
    return new_guesses, n_fixed


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class JacobiConvConfig:
    """Configuration for :class:`RustJacobiConv`.

    Attributes:
        temperature: Sampling temperature (0 = greedy argmax).
    """

    temperature: float = 0.0


class RustJacobiConv:
    """Rust-accelerated Jacobi fixed-point convergence checker.

    Generates updated token guesses for N speculative positions in
    parallel (argmax when temperature=0, Gumbel-max sampling otherwise),
    and counts how many positions have converged compared with the
    previous iteration.  Rayon parallelises per-position sampling.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[JacobiConvConfig] = None) -> None:
        self._cfg = config or JacobiConvConfig()

    def check(
        self,
        pos_logits: np.ndarray,
        guesses: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """Generate new guesses and count converged positions.

        Args:
            pos_logits:  Per-position logits ``(N, vocab)`` float32.
            guesses:     Previous token guesses ``(N,)`` int32.
            temperature: Sampling temperature (None → use config, 0 = greedy).
            seed:        RNG seed for reproducible sampling.

        Returns:
            Tuple of
            - ``new_guesses`` — updated token ids ``(N,)`` int32
            - ``n_fixed`` — number of positions where guess is unchanged

        Raises:
            ValueError: If logits and guesses have incompatible shapes.
        """
        logits = np.ascontiguousarray(pos_logits, dtype=np.float32)
        g = np.ascontiguousarray(guesses, dtype=np.int32).ravel()
        if logits.shape[0] != g.shape[0]:
            raise ValueError(
                f"pos_logits N={logits.shape[0]} != guesses N={g.shape[0]}"
            )
        temp = float(temperature) if temperature is not None else self._cfg.temperature
        if _HAS_RUST:
            ng, nf = _sq.jacobi_conv_check_f32(logits, g, temp, int(seed))
            return np.asarray(ng, dtype=np.int32), int(nf)
        return _numpy_jacobi_check(logits, g, temp, int(seed))

    def converged(
        self,
        pos_logits: np.ndarray,
        guesses: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> bool:
        """Return True if all positions have converged."""
        _, n_fixed = self.check(pos_logits, guesses, temperature, seed)
        return n_fixed == len(guesses)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
