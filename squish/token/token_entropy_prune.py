"""squish/token/token_entropy_prune.py

TokenEntropyPruner — Per-token residual-stream entropy pruning for
activation FLOP reduction (SirLLM, Yao et al., ACL 2024 / arXiv:2405.12528).

Reference
---------
"SirLLM: Streaming Infinite Retentive LLM."
Yao et al., ACL 2024 (arXiv:2405.12528).

Algorithm
---------
At each layer the residual stream carries a hidden state per token.
Lower-information tokens can be identified by the entropy of their
softmax-normalised hidden state:

    H(t) = -sum_d p(d) * log(p(d))    where p = softmax(h_t)

Tokens with the lowest entropy are least informative and are pruned so
that subsequent layers process fewer positions (reducing FLOP).

Pruning is done by a **keep-ratio** or a fixed **budget**:

* ``keep_ratio`` — fraction of tokens to retain per layer.
* ``min_tokens`` — minimum tokens retained regardless of keep_ratio.
* Pruned token positions are filled with a zero vector in the output so
  downstream layers that need positional alignment can still function.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* Returns the pruned hidden states alongside kept indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "TokenEntropyConfig",
    "TokenEntropyPruner",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class TokenEntropyConfig:
    """Configuration for :class:`TokenEntropyPruner`.

    Attributes:
        keep_ratio: Fraction of tokens to keep per layer (0, 1].
        min_tokens: Minimum tokens always retained regardless of keep_ratio.
        fill_pruned: If True, replace pruned positions with zero vectors in
            the output. If False, return only the kept subset.
    """

    keep_ratio: float = 0.5
    min_tokens: int = 1
    fill_pruned: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1]; got {self.keep_ratio}")
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be ≥ 1; got {self.min_tokens}")


# ── Pruner ────────────────────────────────────────────────────────────────────


class TokenEntropyPruner:
    """Token pruner based on residual-stream entropy.

    Example::

        cfg = TokenEntropyConfig(keep_ratio=0.5, min_tokens=4)
        pruner = TokenEntropyPruner(cfg)
        hidden = np.random.randn(32, 128).astype(np.float32)  # (T, d)
        kept, indices = pruner.prune(hidden)
        # kept.shape == (16, 128) if keep_ratio=0.5 and T=32
    """

    def __init__(self, config: Optional[TokenEntropyConfig] = None) -> None:
        self.config = config or TokenEntropyConfig()
        self._total_pruned: int = 0
        self._total_tokens: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def prune(
        self,
        hidden: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prune low-entropy tokens from the residual stream.

        Args:
            hidden: ``(T, d)`` residual hidden states.

        Returns:
            Tuple ``(kept_hidden, kept_indices)`` where:

            * ``kept_hidden``: ``(K, d)`` if ``fill_pruned=False``, else ``(T, d)``
              with pruned rows zeroed.
            * ``kept_indices``: ``(K,)`` int64 indices of retained positions.

        Raises:
            ValueError: If ``hidden`` is not 2-D.
        """
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.ndim != 2:
            raise ValueError(f"hidden must be 2-D (T, d); got shape {hidden.shape}")
        T, d = hidden.shape
        n_keep = max(self.config.min_tokens, int(round(T * self.config.keep_ratio)))
        n_keep = min(n_keep, T)

        entropies = self._token_entropy(hidden)  # (T,)
        # Keep tokens with HIGHEST entropy (most informative)
        kept_idx = np.argsort(entropies)[::-1][:n_keep]
        kept_idx = np.sort(kept_idx).astype(np.int64)

        self._total_pruned += T - n_keep
        self._total_tokens += T

        if self.config.fill_pruned:
            out = np.zeros_like(hidden)
            out[kept_idx] = hidden[kept_idx]
            return out, kept_idx
        return hidden[kept_idx], kept_idx

    def compression_ratio(self) -> float:
        """Average fraction of tokens retained over all calls."""
        if self._total_tokens == 0:
            return 1.0
        return 1.0 - self._total_pruned / self._total_tokens

    def reset_stats(self) -> None:
        """Reset accumulated pruning statistics."""
        self._total_pruned = 0
        self._total_tokens = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _token_entropy(hidden: np.ndarray) -> np.ndarray:
        """Compute per-token entropy of the softmax-normalised hidden state.

        Args:
            hidden: ``(T, d)`` float32 array.

        Returns:
            ``(T,)`` float32 entropy values.
        """
        e = np.exp(hidden - hidden.max(axis=-1, keepdims=True))
        p = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
        return -np.sum(p * np.log(p + 1e-9), axis=-1)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"TokenEntropyPruner(keep_ratio={cfg.keep_ratio}, "
            f"min_tokens={cfg.min_tokens}, "
            f"compression_ratio={self.compression_ratio():.3f})"
        )
