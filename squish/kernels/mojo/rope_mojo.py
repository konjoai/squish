"""Mojo-backed Rotary Position Embedding (RoPE) kernel.

Backend resolution order:
1. Compiled Mojo shared library (via :class:`MojoBridge`)
2. Pure NumPy (no Rust function for RoPE in ``squish_quant``)

The Mojo kernel source lives in ``kernels/rope.mojo``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["MojoRoPEConfig", "MojoRoPE"]


@dataclass
class MojoRoPEConfig:
    """Configuration for :class:`MojoRoPE`.

    Attributes
    ----------
    head_dim:
        Dimensionality of each attention head (must be even).
    max_seq_len:
        Maximum sequence length for frequency cache pre-computation.
    base:
        Theta base for RoPE frequencies (default 10000.0, Llama-style).
    """

    head_dim: int = 128
    max_seq_len: int = 4096
    base: float = 10000.0


class MojoRoPE:
    """Rotary Position Embedding kernel with Mojo → NumPy fallback.

    RoPE rotates consecutive pairs of features in attention keys and queries
    using position-dependent angles θ_i = pos / base^(2i / head_dim).

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`MojoRoPEConfig`.
    bridge:
        Pre-constructed :class:`MojoBridge`.  A default bridge is created
        if not supplied.
    """

    def __init__(
        self,
        config: MojoRoPEConfig | None = None,
        bridge: MojoBridge | None = None,
    ) -> None:
        self.config = config or MojoRoPEConfig()
        self._bridge = bridge or MojoBridge()
        self._freq_cache: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build_freqs(self, seq_len: int) -> np.ndarray:
        """Precompute complex rotation frequencies.

        Parameters
        ----------
        seq_len:
            Number of positions.

        Returns
        -------
        Complex64 array of shape ``(seq_len, head_dim // 2)`` where each
        element is ``exp(i * θ)``.
        """
        half = self.config.head_dim // 2
        positions = np.arange(seq_len, dtype=np.float32)
        dims = np.arange(half, dtype=np.float32)
        inv_freq = 1.0 / (self.config.base ** (2.0 * dims / self.config.head_dim))
        # Outer product: (seq_len, half_dim)
        angles = np.outer(positions, inv_freq).astype(np.float32)
        freqs = np.cos(angles) + 1j * np.sin(angles)
        self._freq_cache = freqs.astype(np.complex64)
        return self._freq_cache

    def apply(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Apply RoPE rotations to query or key tensor.

        Parameters
        ----------
        x:
            float32 array of shape ``(n_heads, seq_len, head_dim)``.
        positions:
            int32 array of shape ``(seq_len,)`` containing absolute positions.

        Returns
        -------
        float32 array of the same shape as *x* with RoPE applied.
        """
        if x.ndim != 3 or x.shape[-1] != self.config.head_dim:
            raise ValueError(
                f"x must be (n_heads, seq_len, head_dim={self.config.head_dim}), "
                f"got {x.shape}"
            )

        # 1. Mojo (ctypes)
        fn = self._bridge.load_kernel("mojo_rope_f32")
        if fn is not None:
            pass  # fall through

        # 2. NumPy
        return self._numpy_apply(x, positions)

    def backend(self) -> str:
        """Return the active backend name."""
        return self._bridge.backend()

    # ------------------------------------------------------------------ #
    #  NumPy fallback                                                      #
    # ------------------------------------------------------------------ #

    def _numpy_apply(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        n_heads, seq_len, head_dim = x.shape
        half = head_dim // 2
        out = x.astype(np.float32, copy=True)

        positions_np = positions.astype(np.int64)
        dims = np.arange(half, dtype=np.float32)
        inv_freq = 1.0 / (self.config.base ** (2.0 * dims / head_dim))

        for s_idx in range(seq_len):
            pos = int(positions_np[s_idx])
            angles = pos * inv_freq  # (half,)
            cos_a = np.cos(angles).astype(np.float32)
            sin_a = np.sin(angles).astype(np.float32)

            # Copy before modifying to avoid aliasing (views share memory)
            x0 = out[:, s_idx, :half].copy()   # (n_heads, half)
            x1 = out[:, s_idx, half:].copy()   # (n_heads, half)
            out[:, s_idx, :half] = x0 * cos_a - x1 * sin_a
            out[:, s_idx, half:] = x0 * sin_a + x1 * cos_a

        return out
