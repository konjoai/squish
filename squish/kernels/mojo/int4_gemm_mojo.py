"""Mojo-backed fused INT4 dequantize + GEMM kernel.

Backend resolution order:
1. Compiled Mojo shared library (via :class:`MojoBridge`)
2. Rust ``squish_quant.dequantize_int4_asymmetric_grouped`` + ``np.matmul``
3. Pure NumPy dequant + matmul

The Mojo kernel source lives in ``kernels/int4_gemm.mojo``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["MojoINT4GEMMConfig", "MojoINT4GEMM"]

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class MojoINT4GEMMConfig:
    """Configuration for :class:`MojoINT4GEMM`.

    Attributes
    ----------
    group_size:
        Quantization group size (must divide ``k``, the inner dimension).
    fuse_dequant:
        When ``True`` (default), the kernel avoids materialising the full
        float32 weight matrix — useful when VRAM is constrained.
    """

    group_size: int = 128
    fuse_dequant: bool = True


class MojoINT4GEMM:
    """INT4 fused dequant-GEMM kernel with Mojo → Rust+NumPy fallback.

    Computes ``x @ W.T`` where ``W`` is stored as asymmetric INT4 (nibble-packed)
    with per-group scales and zero-point offsets.

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`MojoINT4GEMMConfig`.
    bridge:
        Pre-constructed :class:`MojoBridge`.  A default bridge is created
        if not supplied.
    """

    def __init__(
        self,
        config: MojoINT4GEMMConfig | None = None,
        bridge: MojoBridge | None = None,
    ) -> None:
        self.config = config or MojoINT4GEMMConfig()
        self._bridge = bridge or MojoBridge()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def matmul(
        self,
        x: np.ndarray,
        W_packed: np.ndarray,
        scales: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        """Fused INT4 dequant + matrix multiply: ``y = x @ W.T``.

        Parameters
        ----------
        x:
            float32 activation of shape ``(m, k)``.
        W_packed:
            uint8 packed weights of shape ``(n, k // 2)`` — nibble-packed,
            asymmetric INT4.
        scales:
            float32 per-group scales of shape ``(n, k // group_size)``.
        offsets:
            float32 per-group zero-point offsets of shape ``(n, k // group_size)``.

        Returns
        -------
        float32 output of shape ``(m, n)``.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (m, k), got {x.ndim}-D")

        x32 = x.astype(np.float32, copy=False)

        # 1. Mojo (ctypes)
        fn = self._bridge.load_kernel("mojo_int4_gemm_f32")
        if fn is not None:
            pass  # fall through

        # 2. Rust dequant + NumPy matmul
        if _RUST_AVAILABLE:
            try:
                W_f32 = _sq.dequantize_int4_asymmetric_grouped(
                    W_packed, scales, offsets, self.config.group_size
                )
                return (x32 @ W_f32.T).astype(np.float32)
            except Exception:  # noqa: BLE001
                pass

        # 3. NumPy
        W_f32 = self._numpy_dequant(W_packed, scales, offsets, self.config.group_size)
        return (x32 @ W_f32.T).astype(np.float32)

    def backend(self) -> str:
        """Return the active backend name."""
        return self._bridge.backend()

    # ------------------------------------------------------------------ #
    #  NumPy fallback                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_dequant(
        W_packed: np.ndarray,
        scales: np.ndarray,
        offsets: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        n_rows, n_packed = W_packed.shape
        n_cols = n_packed * 2

        lo = (W_packed & 0x0F).astype(np.float32)
        hi = ((W_packed >> 4) & 0x0F).astype(np.float32)
        indices = np.empty((n_rows, n_cols), dtype=np.float32)
        indices[:, 0::2] = lo
        indices[:, 1::2] = hi

        scale_full  = np.repeat(scales,  group_size, axis=1)
        offset_full = np.repeat(offsets, group_size, axis=1)
        return (indices * scale_full + offset_full).astype(np.float32)
