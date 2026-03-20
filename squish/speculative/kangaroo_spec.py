"""squish/speculative/kangaroo_spec.py

KangarooSpec — Lossless Self-Speculative Decoding via Double Early Exiting.

Reference
---------
Liu et al. "Kangaroo: Lossless Self-Speculative Decoding via Double Early
Exiting." arXiv:2404.18911, 2024.

Algorithm
---------
Kangaroo eliminates the need for a separate draft model by using a **shallow
subnetwork** of the original model as the drafter:

1. A small adapter (one linear projection) is attached after the
   ``n_draft_layers``-th transformer layer.
2. During draft generation the subnetwork (first ``n_draft_layers`` layers +
   adapter) autoregressively produces ``draft_length`` candidate tokens.
3. The full model then verifies all draft tokens in a single parallel forward
   pass using speculative decode acceptance-rejection sampling.

This module simulates the algorithm with configurable draft/verify functions.
No actual neural network is required — the draft_fn / target_fn interfaces
accept arbitrary callables that return ``(vocab_size,)`` logits.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``n_draft_layers`` — number of layers in the shallow subnetwork.
* ``draft_length`` — tokens generated per speculation step.
* ``temperature`` — sampling temperature for both draft and target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "KangarooConfig",
    "KangarooDraftResult",
    "KangarooSpec",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class KangarooConfig:
    """Configuration for :class:`KangarooSpec`.

    Attributes:
        n_draft_layers: Depth of the shallow subnetwork drafter.
        draft_length: Number of tokens to draft per speculation step.
        temperature: Sampling temperature (applied to both draft and target).
        seed: RNG seed for reproducible sampling.
    """

    n_draft_layers: int = 4
    draft_length: int = 5
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_draft_layers < 1:
            raise ValueError(
                f"n_draft_layers must be ≥ 1; got {self.n_draft_layers}"
            )
        if self.draft_length < 1:
            raise ValueError(
                f"draft_length must be ≥ 1; got {self.draft_length}"
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0; got {self.temperature}"
            )


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class KangarooDraftResult:
    """Result of one Kangaroo speculation step.

    Attributes:
        accepted_tokens: Token ids that were accepted (length ≤ draft_length + 1).
        n_accepted: Number of accepted tokens.
        n_drafted: Number of drafted tokens.
        acceptance_rate: n_accepted / n_drafted.
    """

    accepted_tokens: List[int]
    n_accepted: int
    n_drafted: int
    acceptance_rate: float


# ── Core class ─────────────────────────────────────────────────────────────────


class KangarooSpec:
    """Shallow-subnetwork self-speculative decoder.

    Example::

        cfg    = KangarooConfig(n_draft_layers=2, draft_length=4)
        spec   = KangarooSpec(cfg)

        def draft_fn(token_id, context):
            return np.random.randn(50257)  # logits from shallow subnetwork

        def target_fn(token_id, context):
            return np.random.randn(50257)  # logits from full model

        result = spec.step(context_ids=[1, 2, 3], draft_fn=draft_fn, target_fn=target_fn)
    """

    def __init__(self, config: Optional[KangarooConfig] = None) -> None:
        self.config = config or KangarooConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._n_steps: int = 0
        self._total_accepted: int = 0
        self._total_drafted: int = 0

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        context_ids: List[int],
        draft_fn: Callable[[int, List[int]], np.ndarray],
        target_fn: Callable[[int, List[int]], np.ndarray],
    ) -> KangarooDraftResult:
        """Run one speculation step.

        Args:
            context_ids: Current token context (list of ints).
            draft_fn: Callable ``(last_token, context) -> logits (vocab_size,)``
                representing the shallow subnetwork.
            target_fn: Callable ``(last_token, context) -> logits (vocab_size,)``
                representing the full model.

        Returns:
            :class:`KangarooDraftResult` with accepted tokens and statistics.
        """
        ctx = list(context_ids)
        temp = self.config.temperature

        # ── Draft phase ───────────────────────────────────────────────────────
        draft_tokens: List[int] = []
        draft_probs: List[np.ndarray] = []
        for _ in range(self.config.draft_length):
            last = ctx[-1] if ctx else 0
            logits = np.asarray(draft_fn(last, ctx), dtype=np.float32)
            p = self._softmax(logits / temp)
            tok = int(self._rng.choice(len(p), p=p))
            draft_tokens.append(tok)
            draft_probs.append(p)
            ctx.append(tok)

        # ── Verify phase ──────────────────────────────────────────────────────
        accepted: List[int] = []
        verify_ctx = list(context_ids)
        for i, (dt, dp) in enumerate(zip(draft_tokens, draft_probs)):
            last = verify_ctx[-1] if verify_ctx else 0
            t_logits = np.asarray(target_fn(last, verify_ctx), dtype=np.float32)
            tp = self._softmax(t_logits / temp)

            # Acceptance probability: min(1, p_target / p_draft)
            accept_prob = float(min(1.0, tp[dt] / (dp[dt] + 1e-9)))
            u = float(self._rng.uniform(0.0, 1.0))
            if u < accept_prob:
                accepted.append(dt)
                verify_ctx.append(dt)
            else:
                # Rejected: sample from residual distribution
                residual = np.maximum(tp - dp, 0.0)
                if residual.sum() > 1e-9:
                    residual /= residual.sum()
                    bonus = int(self._rng.choice(len(residual), p=residual))
                    accepted.append(bonus)
                break

        # If all drafts accepted, sample one bonus token from target
        if len(accepted) == self.config.draft_length:
            last = verify_ctx[-1] if verify_ctx else 0
            t_logits = np.asarray(target_fn(last, verify_ctx), dtype=np.float32)
            tp = self._softmax(t_logits / temp)
            bonus = int(self._rng.choice(len(tp), p=tp))
            accepted.append(bonus)

        n_acc = len(accepted)
        n_dft = self.config.draft_length
        self._n_steps += 1
        self._total_accepted += n_acc
        self._total_drafted += n_dft

        return KangarooDraftResult(
            accepted_tokens=accepted,
            n_accepted=n_acc,
            n_drafted=n_dft,
            acceptance_rate=n_acc / max(n_dft, 1),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-9)

    @property
    def mean_acceptance_rate(self) -> float:
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted

    def reset_stats(self) -> None:
        self._n_steps = 0
        self._total_accepted = 0
        self._total_drafted = 0

    def __repr__(self) -> str:
        return (
            f"KangarooSpec(n_draft_layers={self.config.n_draft_layers}, "
            f"draft_length={self.config.draft_length}, "
            f"mean_acceptance_rate={self.mean_acceptance_rate:.3f})"
        )
