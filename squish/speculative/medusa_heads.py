"""squish/speculative/medusa_heads.py

MedusaHeads — Multiple decoding heads for parallel speculative token generation
(Cai et al., ICML 2024 / arXiv:2401.10774).

Reference
---------
"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding
Heads." Cai et al., ICML 2024 (arXiv:2401.10774).

Algorithm
---------
Medusa attaches K extra "draft heads" to the base model's final hidden state.
Each head independently predicts the i-th future token (head 1 → token t+1,
head 2 → token t+2, …).  The K head predictions are expanded into a tree of
candidate continuations, which are then verified in a single batched forward
pass of the full model.  Accepted tokens are returned; rejected tokens trigger
fallback sampling from the base model.

This simulation:
* Each head = a linear projection over hidden_dim → vocab_size.
* ``step()`` generates K single-token drafts (one per head), assembles and
  verifies them with ``target_fn``, and returns the accepted sequence.
* No actual weight matrices needed — draft_fn / target_fn are callbacks.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``n_heads`` draft heads (default 2) beyond the base model.
* ``tree_width`` candidates per head at each level (default 3).
* ``accept_threshold`` specifies minimum acceptance probability (default 0.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "MedusaConfig",
    "MedusaDraftResult",
    "MedusaHeads",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class MedusaConfig:
    """Configuration for :class:`MedusaHeads`.

    Attributes:
        n_heads: Number of Medusa draft heads (default 2).
        tree_width: Top-k candidates considered per head (default 3).
        accept_threshold: Minimum acceptance probability per candidate (default 0.0).
        temperature: Sampling temperature for draft heads (default 1.0).
        seed: RNG seed for reproducibility.
    """

    n_heads: int = 2
    tree_width: int = 3
    accept_threshold: float = 0.0
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.tree_width < 1:
            raise ValueError(f"tree_width must be ≥ 1; got {self.tree_width}")
        if not (0.0 <= self.accept_threshold < 1.0):
            raise ValueError(
                f"accept_threshold must be in [0, 1); got {self.accept_threshold}"
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")


@dataclass
class MedusaDraftResult:
    """Return value of :meth:`MedusaHeads.step`.

    Attributes:
        accepted_tokens: List of accepted token IDs (always ≥ 1).
        n_accepted: Number of accepted tokens.
        n_drafted: Total draft tokens proposed across all heads.
        acceptance_rate: n_accepted / n_drafted.
    """

    accepted_tokens: List[int]
    n_accepted: int
    n_drafted: int
    acceptance_rate: float


# ── MedusaHeads ───────────────────────────────────────────────────────────────


class MedusaHeads:
    """Medusa multi-head speculative decoder.

    Example::

        cfg = MedusaConfig(n_heads=2, tree_width=3)
        medusa = MedusaHeads(cfg)

        def draft_fn(head_idx, last_token, context):
            return np.random.dirichlet(np.ones(32)).astype(np.float32)

        def target_fn(last_token, context):
            return np.random.dirichlet(np.ones(32)).astype(np.float32)

        result = medusa.step([1, 2, 3], draft_fn, target_fn)
    """

    def __init__(self, config: Optional[MedusaConfig] = None) -> None:
        self.config = config or MedusaConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._total_accepted = 0
        self._total_drafted = 0
        self._n_steps = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        context_ids: List[int],
        draft_fn: Callable[[int, int, List[int]], np.ndarray],
        target_fn: Callable[[int, List[int]], np.ndarray],
    ) -> MedusaDraftResult:
        """Run one Medusa decode step.

        Args:
            context_ids: Current context token IDs.
            draft_fn: ``(head_idx, last_token, context) → probs(vocab_size)``
                called once per draft head.
            target_fn: ``(last_token, context) → probs(vocab_size)``
                base model probabilities used for verification.

        Returns:
            :class:`MedusaDraftResult` with accepted tokens and stats.
        """
        last_token = context_ids[-1]
        cfg = self.config

        # Each head independently predicts the next+i token
        draft_tokens: List[int] = []
        draft_probs: List[float] = []
        context = list(context_ids)

        for h in range(cfg.n_heads):
            head_probs = np.asarray(draft_fn(h, last_token, context), dtype=np.float32)
            head_probs = self._safe_normalize(head_probs / cfg.temperature)
            top_idx = int(np.argmax(head_probs))
            draft_tokens.append(top_idx)
            draft_probs.append(float(head_probs[top_idx]))
            context = context + [top_idx]

        # Verify draft tokens against the target model
        accepted: List[int] = []
        verify_context = list(context_ids)

        for i, (dt, dp) in enumerate(zip(draft_tokens, draft_probs)):
            target_probs = np.asarray(
                target_fn(verify_context[-1], verify_context), dtype=np.float32
            )
            target_probs = self._safe_normalize(target_probs)
            # Speculative acceptance: accept with prob min(1, target[dt] / draft[dt])
            target_p = float(target_probs[dt])
            accept_p = min(1.0, target_p / max(dp, 1e-9))
            if self._rng.random() < accept_p:
                accepted.append(dt)
                verify_context.append(dt)
            else:
                # Residual correction — sample from modified distribution
                residual = target_probs - self._safe_normalize(
                    np.eye(len(target_probs))[dt] * dp
                )
                residual = np.clip(residual, 0.0, None)
                s = residual.sum()
                if s > 1e-9:
                    residual /= s
                    fallback = int(self._rng.choice(len(residual), p=residual))
                else:
                    fallback = int(np.argmax(target_probs))
                accepted.append(fallback)
                break

        # Ensure at least one token (base-model fallback if no draft accepted)
        if not accepted:
            target_probs = np.asarray(
                target_fn(context_ids[-1], list(context_ids)), dtype=np.float32
            )
            target_probs = self._safe_normalize(target_probs)
            accepted.append(int(self._rng.choice(len(target_probs), p=target_probs)))

        n_drafted = len(draft_tokens)
        n_accepted = len(accepted)
        self._total_accepted += n_accepted
        self._total_drafted += n_drafted
        self._n_steps += 1

        return MedusaDraftResult(
            accepted_tokens=accepted,
            n_accepted=n_accepted,
            n_drafted=n_drafted,
            acceptance_rate=n_accepted / max(n_drafted, 1),
        )

    @property
    def mean_acceptance_rate(self) -> float:
        """Mean acceptance rate across all steps."""
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted

    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self._total_accepted = 0
        self._total_drafted = 0
        self._n_steps = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_normalize(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 0.0, None)
        s = p.sum()
        return p / s if s > 1e-9 else np.ones_like(p) / len(p)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"MedusaHeads(n_heads={cfg.n_heads}, tree_width={cfg.tree_width}, "
            f"steps={self._n_steps}, mean_ar={self.mean_acceptance_rate:.3f})"
        )
