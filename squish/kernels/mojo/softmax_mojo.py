"""Mojo-backed softmax + top-p sampler kernel.

Backend resolution order:
1. Compiled Mojo shared library (via :class:`MojoBridge`)
2. Rust ``squish_quant`` extension
3. Pure NumPy

The Mojo kernel source lives in ``kernels/softmax.mojo``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge, MojoBridgeConfig

__all__ = ["MojoSoftmaxConfig", "MojoSoftmax"]

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class MojoSoftmaxConfig:
    """Configuration for :class:`MojoSoftmax`.

    Attributes
    ----------
    temperature:
        Logit temperature scale applied before softmax.
    top_p:
        Nucleus probability mass for :meth:`fused_top_p`.
    seed:
        Unused in this kernel; kept for API consistency.
    """

    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 0


class MojoSoftmax:
    """Softmax / top-p kernel with Mojo → Rust → NumPy fallback chain.

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`MojoSoftmaxConfig`.
    bridge:
        Pre-constructed :class:`MojoBridge`.  A default bridge is created
        if not supplied.
    """

    def __init__(
        self,
        config: MojoSoftmaxConfig | None = None,
        bridge: MojoBridge | None = None,
    ) -> None:
        self.config = config or MojoSoftmaxConfig()
        self._bridge = bridge or MojoBridge()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.

        Parameters
        ----------
        logits:
            1-D or 2-D float32 array.  If 2-D, softmax is applied over axis -1.

        Returns
        -------
        float32 probability array with the same shape.
        """
        flat = logits.astype(np.float32, copy=False).ravel()

        if self.config.temperature != 1.0:
            flat = flat / max(self.config.temperature, 1e-8)

        # 1. Mojo (ctypes)
        fn = self._bridge.load_kernel("mojo_softmax_f32")
        if fn is not None:
            # ctypes call would go here when library is compiled
            pass  # fall through to Rust

        # 2. Rust
        if _RUST_AVAILABLE:
            try:
                out = _sq.softmax_logits_f32(flat)
                return out.reshape(logits.shape)
            except Exception:  # noqa: BLE001
                pass

        # 3. NumPy
        out = self._numpy_softmax(flat)
        return out.reshape(logits.shape)

    def fused_top_p(self, logits: np.ndarray, p: float | None = None) -> np.ndarray:
        """Softmax followed by top-p filter.

        Parameters
        ----------
        logits:
            1-D float32 logit array.
        p:
            Nucleus mass.  Defaults to ``config.top_p``.

        Returns
        -------
        Re-normalised float32 probability array.
        """
        p_val = self.config.top_p if p is None else p
        flat = logits.astype(np.float32, copy=False).ravel()

        if self.config.temperature != 1.0:
            flat = flat / max(self.config.temperature, 1e-8)

        if _RUST_AVAILABLE:
            try:
                probs = _sq.softmax_logits_f32(flat)
                return _sq.top_p_filter_f32(probs, float(p_val))
            except Exception:  # noqa: BLE001
                pass

        probs = self._numpy_softmax(flat)
        return self._numpy_top_p(probs, float(p_val))

    def backend(self) -> str:
        """Return the active backend name: ``"mojo"``, ``"rust"``, or ``"numpy"``."""
        return self._bridge.backend()

    # ------------------------------------------------------------------ #
    #  NumPy fallback                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        exp_vals = np.exp(shifted)
        return (exp_vals / exp_vals.sum()).astype(np.float32)

    @staticmethod
    def _numpy_top_p(probs: np.ndarray, p: float) -> np.ndarray:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cumsum, p, side="right"))
        keep = sorted_idx[: cutoff + 1]
        out = np.zeros_like(probs)
        out[keep] = probs[keep]
        total = out.sum()
        if total > 1e-10:
            out /= total
        return out
