# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""RetentionAttn — Retention-style recurrent attention for O(1) per-step inference.

RetNet (Sun et al., 2023) replaces softmax attention with a linear recurrent
formulation.  At each decode step the per-head state matrix S is updated as::

    S_t = γ · S_{t-1} + k_t^T ⊗ v_t

and the output for head h is::

    o_t = q_t · S_t

where γ ∈ (0, 1) is a learnable (here fixed) exponential decay factor.  This
gives O(1) per-step memory cost — there is no KV cache growth — and O(n) total
compute for a sequence of length n.  The per-head state ``S ∈ R^{d × d}``
captures the entire history implicitly.

Reference:
    Sun et al., "Retentive Network: A Successor to Transformer for Large
    Language Models", arXiv 2023.
    https://arxiv.org/abs/2307.08621

Usage::

    import numpy as np
    from squish.attention.retention_attn import RetentionKernel, RetentionConfig

    cfg    = RetentionConfig(hidden_dim=512, n_heads=8, gamma=0.9)
    kernel = RetentionKernel(cfg)
    state  = kernel.init_state()

    rng = np.random.default_rng(0)
    for _ in range(64):
        q = rng.standard_normal((8, 64)).astype(np.float32)
        k = rng.standard_normal((8, 64)).astype(np.float32)
        v = rng.standard_normal((8, 64)).astype(np.float32)
        output, state = kernel.step(q, k, v, state)

    print(f"State step: {state.step}, stats: {kernel.stats}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["RetentionConfig", "RetentionState", "RetentionKernel", "RetentionStats"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RetentionConfig:
    """Configuration for the retention attention kernel.

    Attributes:
        hidden_dim: Model hidden dimension.  Must be divisible by ``n_heads``.
        n_heads: Number of retention heads.
        gamma: Exponential decay factor applied to the recurrent state at each
            step.  Must be in ``(0, 1)``.  Values close to 1 preserve long-
            range context; values close to 0 emphasise recent tokens.
    """

    hidden_dim: int = 512
    n_heads: int = 8
    gamma: float = 0.9

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1; got {self.hidden_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(
                f"gamma must be in (0, 1); got {self.gamma}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each retention head (``hidden_dim // n_heads``)."""
        return self.hidden_dim // self.n_heads


# ---------------------------------------------------------------------------
# State / Stats
# ---------------------------------------------------------------------------


@dataclass
class RetentionState:
    """Recurrent state for :class:`RetentionKernel`.

    Attributes:
        S: Per-head state matrices, shape
            ``(n_heads, head_dim, head_dim)`` float32.
            ``S[h]`` encodes the accumulated context for head ``h``.
        step: Number of retention steps that have been applied.
    """

    S: np.ndarray
    step: int = 0


@dataclass
class RetentionStats:
    """Accumulated statistics for :class:`RetentionKernel`.

    Attributes:
        total_steps: Total number of :meth:`RetentionKernel.step` calls.
        total_states_init: Number of times :meth:`RetentionKernel.init_state`
            was called.
    """

    total_steps: int = 0
    total_states_init: int = 0


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class RetentionKernel:
    """Recurrent retention attention kernel with O(1) per-step memory.

    Implements the per-step recurrence for each head h::

        S_t[h] = γ · S_{t-1}[h] + k[h] ⊗ v[h]   (outer product update)
        o_t[h] = q[h] · S_t[h]                    (state query)

    where ``q, k, v ∈ R^{head_dim}`` per head, and
    ``S ∈ R^{head_dim × head_dim}`` per head.

    All operations are vectorised across heads using NumPy broadcasting.

    Args:
        config: :class:`RetentionConfig` instance.
    """

    def __init__(self, config: RetentionConfig) -> None:
        self.config = config
        self._stats = RetentionStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_state(self) -> RetentionState:
        """Initialise a zeroed recurrent state.

        Returns:
            :class:`RetentionState` with all-zero state matrices and
            ``step=0``, ready for the first decode step.
        """
        cfg = self.config
        S = np.zeros(
            (cfg.n_heads, cfg.head_dim, cfg.head_dim), dtype=np.float32
        )
        self._stats.total_states_init += 1
        return RetentionState(S=S, step=0)

    def step(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        state: RetentionState,
    ) -> tuple[np.ndarray, RetentionState]:
        """Apply one retention step.

        Args:
            q: Query vectors, shape ``(n_heads, head_dim)`` float32.
            k: Key vectors, shape ``(n_heads, head_dim)`` float32.
            v: Value vectors, shape ``(n_heads, head_dim)`` float32.
            state: Current :class:`RetentionState`.

        Returns:
            Tuple ``(output, new_state)`` where:

            * ``output`` has shape ``(n_heads, head_dim)`` float32.
            * ``new_state`` is the updated :class:`RetentionState` with
              ``step`` incremented by 1.

        Raises:
            ValueError: If any input tensor has an unexpected shape.
        """
        cfg = self.config
        expected = (cfg.n_heads, cfg.head_dim)

        for name, arr in (("q", q), ("k", k), ("v", v)):
            if arr.shape != expected:
                raise ValueError(
                    f"Expected {name} shape {expected}; got {arr.shape}"
                )

        if state.S.shape != (cfg.n_heads, cfg.head_dim, cfg.head_dim):
            raise ValueError(
                f"state.S must have shape "
                f"({cfg.n_heads}, {cfg.head_dim}, {cfg.head_dim}); "
                f"got {state.S.shape}"
            )

        q_f = q.astype(np.float32)
        k_f = k.astype(np.float32)
        v_f = v.astype(np.float32)

        # Outer product update per head:
        # outer[h, i, j] = k[h, i] * v[h, j]  → shape (n_heads, head_dim, head_dim)
        outer = k_f[:, :, np.newaxis] * v_f[:, np.newaxis, :]

        # State update: S_t = γ · S_{t-1} + k ⊗ v
        new_S = (cfg.gamma * state.S + outer).astype(np.float32)

        # Output: o[h, j] = sum_i q[h, i] * S[h, i, j]
        output = np.einsum("hi,hij->hj", q_f, new_S).astype(np.float32)

        new_state = RetentionState(S=new_S, step=state.step + 1)
        self._stats.total_steps += 1
        return output, new_state

    def reset_stats(self) -> None:
        """Reset accumulated statistics to zero."""
        self._stats = RetentionStats()

    @property
    def stats(self) -> RetentionStats:
        """Current accumulated :class:`RetentionStats`."""
        return self._stats
