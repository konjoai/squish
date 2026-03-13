# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""DynamicNTK — Per-request runtime RoPE base auto-scaling.

NTK-aware RoPE scaling can be applied dynamically: when the actual sequence
length exceeds a configurable fraction of the trained context window, the base
theta is automatically scaled up to extend the effective context without any
fine-tuning.

References:
    bloc97, "NTK-Aware Scaled RoPE", Reddit / HuggingFace blog, 2023.
    https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5

    Peng et al., "YaRN: Efficient Context Window Extension of Large Language
    Models", arXiv 2309.00071, 2023.  https://arxiv.org/abs/2309.00071

Usage::

    from squish.attention.dynamic_ntk import DynamicNTKScaler, DynamicNTKConfig
    import numpy as np

    cfg    = DynamicNTKConfig(base_theta=10000.0, max_trained_len=4096,
                              trigger_ratio=0.8, alpha=8.0, head_dim=128)
    scaler = DynamicNTKScaler(cfg)
    freqs  = scaler.get_freqs(seq_len=3500)
    print(f"scaled={scaler.state.is_scaled}, base={scaler.effective_base:.0f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DynamicNTKConfig",
    "NTKState",
    "DynamicNTKScaler",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DynamicNTKConfig:
    """Configuration for dynamic NTK-aware RoPE base scaling.

    Attributes:
        base_theta:       Original RoPE base frequency (e.g. 10 000 for
                          LLaMA-1/2).
        max_trained_len:  Maximum sequence length the model was trained on.
        trigger_ratio:    Fraction of ``max_trained_len`` at which automatic
                          rescaling is activated.  Must be in ``(0, 1]``.
        alpha:            NTK scaling alpha factor.  The effective base becomes
                          ``base_theta * (alpha * seq_len / max_trained_len -
                          (alpha - 1)) ** (head_dim / (head_dim - 2))``.
        head_dim:         Attention head dimension.  Must be even.
    """

    base_theta: float = 10000.0
    max_trained_len: int = 4096
    trigger_ratio: float = 0.8
    alpha: float = 8.0
    head_dim: int = 128

    def __post_init__(self) -> None:
        if self.base_theta <= 0.0:
            raise ValueError(
                f"base_theta must be > 0; got {self.base_theta}"
            )
        if self.max_trained_len < 1:
            raise ValueError(
                f"max_trained_len must be >= 1; got {self.max_trained_len}"
            )
        if not (0.0 < self.trigger_ratio <= 1.0):
            raise ValueError(
                f"trigger_ratio must be in (0, 1]; got {self.trigger_ratio}"
            )
        if self.alpha <= 1.0:
            raise ValueError(f"alpha must be > 1; got {self.alpha}")
        if self.head_dim < 2 or self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even and >= 2; got {self.head_dim}"
            )

    @property
    def trigger_len(self) -> int:
        """Sequence length at which NTK rescaling activates."""
        return int(self.max_trained_len * self.trigger_ratio)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class NTKState:
    """Mutable runtime state for a :class:`DynamicNTKScaler`.

    Attributes:
        current_len:  Most recently seen sequence length.
        current_base: Active RoPE base theta (equals ``base_theta`` before
                      any scaling, dynamically adjusted thereafter).
        is_scaled:    Whether the base has been upscaled from its original
                      value at least once.
    """

    current_len: int = 0
    current_base: float = 0.0
    is_scaled: bool = False


# ---------------------------------------------------------------------------
# DynamicNTKScaler
# ---------------------------------------------------------------------------


class DynamicNTKScaler:
    """Per-request runtime RoPE base frequency auto-scaler.

    Monitors the running sequence length and, when it exceeds
    ``config.trigger_len``, recomputes the base theta using the NTK scaling
    formula so that the effective context window is extended.

    The NTK scaling formula:

    .. code-block:: text

        ratio     = seq_len / max_trained_len
        new_base  = base_theta * (alpha * ratio - (alpha - 1))
                    ** (head_dim / (head_dim - 2))

    When ``seq_len <= trigger_len`` the original ``base_theta`` is used
    unchanged.

    Args:
        config: :class:`DynamicNTKConfig` instance.
    """

    def __init__(self, config: DynamicNTKConfig) -> None:
        self._config = config
        self._state = NTKState(
            current_len=0,
            current_base=config.base_theta,
            is_scaled=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_freqs(self, seq_len: int) -> np.ndarray:
        """Return RoPE angular frequencies for the given sequence length.

        The effective base theta is recomputed via NTK scaling if
        ``seq_len > trigger_len``.

        Args:
            seq_len: Current sequence length (number of token positions).

        Returns:
            1-D float32 array of shape ``(head_dim // 2,)`` containing the
            angular frequency for each dimension pair.

        Raises:
            ValueError: if ``seq_len < 1``.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1; got {seq_len}")

        self.update(seq_len)
        base = self._state.current_base
        half_dim = self._config.head_dim // 2

        # Standard RoPE: theta_i = base^(-2i / head_dim)
        indices = np.arange(half_dim, dtype=np.float32)
        freqs = 1.0 / (base ** (indices * 2.0 / self._config.head_dim))
        return freqs.astype(np.float32)

    def update(self, new_len: int) -> None:
        """Update the internal state when the sequence grows to ``new_len``.

        If ``new_len`` exceeds the trigger length, the NTK scaling formula
        is applied to compute the new base theta.  Scaling is always
        recomputed from the latest ``new_len`` rather than cached, so
        the base continuously adjusts as the sequence grows.

        Args:
            new_len: New (or current) sequence length.

        Raises:
            ValueError: if ``new_len < 1``.
        """
        if new_len < 1:
            raise ValueError(f"new_len must be >= 1; got {new_len}")

        cfg = self._config
        self._state.current_len = new_len

        if new_len > cfg.trigger_len:
            new_base = self._compute_ntk_base(new_len)
            self._state.current_base = new_base
            self._state.is_scaled = True
        else:
            self._state.current_base = cfg.base_theta
            self._state.is_scaled = False

    @property
    def state(self) -> NTKState:
        """Current runtime NTK state."""
        return self._state

    @property
    def effective_base(self) -> float:
        """Currently active RoPE base theta."""
        return self._state.current_base

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_ntk_base(self, seq_len: int) -> float:
        """Apply the NTK scaling formula for a given sequence length.

        Args:
            seq_len: Sequence length exceeding the trigger threshold.

        Returns:
            Scaled base theta as a Python float.
        """
        cfg = self._config
        ratio = seq_len / cfg.max_trained_len
        # NTK-aware scaling: stretch the RoPE base so that the effective
        # context window covers at least seq_len positions.
        exponent = cfg.head_dim / (cfg.head_dim - 2)
        scale_factor = cfg.alpha * ratio - (cfg.alpha - 1.0)
        # Guard against sub-1 scale factors (should not occur when seq_len
        # > trigger_len and alpha > 1, but defend anyway).
        scale_factor = max(scale_factor, 1.0)
        return cfg.base_theta * (scale_factor ** exponent)
