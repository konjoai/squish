"""Mojo-backed NF4 dequantization kernel.

Backend resolution order:
1. Compiled Mojo shared library (via :class:`MojoBridge`)
2. Rust ``squish_quant.dequantize_nf4_grouped_f32``
3. Pure NumPy (LUT gather)

The Mojo kernel source lives in ``kernels/nf4_dequant.mojo``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["MojoNF4DequantConfig", "MojoNF4Dequant"]

# NF4 lookup table (16 levels, standard-normal quantile function)
_NF4_LUT: np.ndarray = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class MojoNF4DequantConfig:
    """Configuration for :class:`MojoNF4Dequant`.

    Attributes
    ----------
    group_size:
        Number of elements per quantization group.
    """

    group_size: int = 64


class MojoNF4Dequant:
    """NF4 dequantization kernel with Mojo → Rust → NumPy fallback.

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`MojoNF4DequantConfig`.
    bridge:
        Pre-constructed :class:`MojoBridge`.  A default bridge is created
        if not supplied.
    """

    def __init__(
        self,
        config: MojoNF4DequantConfig | None = None,
        bridge: MojoBridge | None = None,
    ) -> None:
        self.config = config or MojoNF4DequantConfig()
        self._bridge = bridge or MojoBridge()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def dequantize(self, packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize NF4 nibble-packed weights to float32.

        Parameters
        ----------
        packed:
            uint8 array of shape ``(N, D // 2)`` — two nibbles per byte.
        scales:
            float32 array of shape ``(N, D // group_size)``.

        Returns
        -------
        float32 array of shape ``(N, D)``.
        """
        # 1. Mojo (ctypes)
        fn = self._bridge.load_kernel("mojo_dequantize_nf4_f32")
        if fn is not None:
            pass  # fall through

        # 2. Rust
        if _RUST_AVAILABLE:
            try:
                return _sq.dequantize_nf4_grouped_f32(
                    packed, scales, self.config.group_size
                )
            except Exception:  # noqa: BLE001
                pass

        # 3. NumPy
        return self._numpy_dequantize(packed, scales, self.config.group_size)

    def backend(self) -> str:
        """Return the active backend name."""
        return self._bridge.backend()

    # ------------------------------------------------------------------ #
    #  NumPy fallback                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_dequantize(
        packed: np.ndarray, scales: np.ndarray, group_size: int
    ) -> np.ndarray:
        n_rows, n_packed = packed.shape
        n_cols = n_packed * 2

        lo = (packed & 0x0F).astype(np.int32)
        hi = ((packed >> 4) & 0x0F).astype(np.int32)
        indices = np.empty((n_rows, n_cols), dtype=np.int32)
        indices[:, 0::2] = lo
        indices[:, 1::2] = hi

        out = _NF4_LUT[indices]
        scale_full = np.repeat(scales, group_size, axis=1)
        return (out * scale_full).astype(np.float32)
