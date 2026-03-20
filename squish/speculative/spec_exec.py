"""squish/speculative/spec_exec.py

SpecExecDrafter — Massively Parallel Speculative Decoding.

Reference
---------
Svirschevski et al. "SpecExec: Massively Parallel Speculative Decoding for
Interactive LLM Inference on Consumer Devices." arXiv:2405.00047, 2024.

Algorithm
---------
SpecExec constructs a **speculative token tree** by greedily expanding the
residual probability distribution of the draft model:

1. Start from the current context.
2. At each node in the tree, sample the top-N most probable next tokens from
   the draft model (using residual = draft_prob - already_assigned_mass).
3. Expand the tree breadth-first until the total token budget ``B`` is reached.
4. The target model verifies all B tree leaves in one parallel forward pass.
5. Accept tokens by walking the tree from root, accepting each token with
   probability min(1, p_target / p_draft).

Key properties
--------------
* NumPy-only simulation with functional draft_fn / target_fn interfaces.
* ``budget`` — total draft tokens per speculation step.
* ``beam_width`` — branching factor at each node.
* ``temperature`` — draft sampling temperature.
* ``seed`` — RNG seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SpecExecConfig",
    "SpecExecResult",
    "SpecExecDrafter",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SpecExecConfig:
    """Configuration for :class:`SpecExecDrafter`.

    Attributes:
        budget: Total draft tokens generated per step.
        beam_width: Branching factor (top-N tokens per node).
        max_depth: Maximum tree depth (prevents unbounded recursion).
        temperature: Sampling temperature for draft distribution.
        seed: RNG seed.
    """

    budget: int = 16
    beam_width: int = 4
    max_depth: int = 8
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError(f"budget must be ≥ 1; got {self.budget}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be ≥ 1; got {self.beam_width}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be ≥ 1; got {self.max_depth}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class SpecExecResult:
    """Result of one SpecExec speculation step.

    Attributes:
        accepted_tokens: List of accepted token ids from root to first rejection.
        n_accepted: Number of accepted tokens.
        n_drafted: Total draft tree size.
        acceptance_rate: n_accepted / max(n_drafted, 1).
    """

    accepted_tokens: List[int]
    n_accepted: int
    n_drafted: int
    acceptance_rate: float


# ── Internal tree node ────────────────────────────────────────────────────────


@dataclass
class _TreeNode:
    token_id: int
    parent_id: int  # -1 for root
    depth: int
    draft_prob: float
    context: List[int]
    node_id: int


# ── Core class ─────────────────────────────────────────────────────────────────


class SpecExecDrafter:
    """Budget-bounded speculative token tree drafter.

    Example::

        cfg    = SpecExecConfig(budget=8, beam_width=2)
        drafter = SpecExecDrafter(cfg)

        def draft_fn(token, context):
            return np.random.randn(50257)

        def target_fn(token, context):
            return np.random.randn(50257)

        result = drafter.step([1, 2, 3], draft_fn, target_fn)
    """

    def __init__(self, config: Optional[SpecExecConfig] = None) -> None:
        self.config = config or SpecExecConfig()
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
    ) -> SpecExecResult:
        """Run one SpecExec speculation step.

        Args:
            context_ids: Current token context.
            draft_fn: ``(last_token, context) -> logits (vocab_size,)`` for draft model.
            target_fn: ``(last_token, context) -> logits (vocab_size,)`` for target model.

        Returns:
            :class:`SpecExecResult` with accepted tokens and statistics.
        """
        cfg = self.config
        temp = cfg.temperature

        # ── Build token tree greedily ─────────────────────────────────────────
        nodes: List[_TreeNode] = []
        # Queue of (parent_node_id, context) for BFS
        queue: List[Tuple[int, List[int]]] = [(-1, list(context_ids))]
        node_id = 0
        total_drafted = 0

        while queue and total_drafted < cfg.budget:
            parent_id, ctx = queue.pop(0)
            parent_depth = nodes[parent_id].depth if parent_id >= 0 else 0
            if parent_depth >= cfg.max_depth:
                continue

            last = ctx[-1] if ctx else 0
            logits = np.asarray(draft_fn(last, ctx), dtype=np.float32)
            probs = self._softmax(logits / temp)
            top_k = min(cfg.beam_width, len(probs))
            top_idx = np.argsort(-probs)[:top_k]

            for tok in top_idx:
                if total_drafted >= cfg.budget:
                    break
                tok = int(tok)
                p = float(probs[tok])
                n = _TreeNode(
                    token_id=tok,
                    parent_id=parent_id,
                    depth=parent_depth + 1,
                    draft_prob=p,
                    context=ctx + [tok],
                    node_id=node_id,
                )
                nodes.append(n)
                queue.append((node_id, ctx + [tok]))
                node_id += 1
                total_drafted += 1

        # ── Verify: walk tree from root, accept/reject ─────────────────────────
        accepted: List[int] = []
        verify_ctx = list(context_ids)

        # Collect root-level nodes (depth=1, parent=-1)
        current_level = [n for n in nodes if n.parent_id == -1]

        while current_level:
            n = current_level[0]
            last = verify_ctx[-1] if verify_ctx else 0
            t_logits = np.asarray(target_fn(last, verify_ctx), dtype=np.float32)
            tp = self._softmax(t_logits / temp)
            dp = n.draft_prob

            accept_p = float(min(1.0, tp[n.token_id] / (dp + 1e-9)))
            u = float(self._rng.uniform(0.0, 1.0))
            if u < accept_p:
                accepted.append(n.token_id)
                verify_ctx.append(n.token_id)
                # Move to children of accepted node
                current_level = [c for c in nodes if c.parent_id == n.node_id]
            else:
                # Resample from residual
                residual = np.maximum(tp - self._uniform_draft(len(tp), dp), 0.0)
                if residual.sum() > 1e-9:
                    residual /= residual.sum()
                    bonus = int(self._rng.choice(len(residual), p=residual))
                    accepted.append(bonus)
                break

        # If entire path accepted, sample one bonus token from target
        if not any(n.depth > len(accepted) for n in nodes if n.parent_id >= 0):
            if verify_ctx:
                last = verify_ctx[-1]
                t_logits = np.asarray(target_fn(last, verify_ctx), dtype=np.float32)
                tp = self._softmax(t_logits / temp)
                accepted.append(int(self._rng.choice(len(tp), p=tp)))

        n_acc = len(accepted)
        self._n_steps += 1
        self._total_accepted += n_acc
        self._total_drafted += total_drafted

        return SpecExecResult(
            accepted_tokens=accepted,
            n_accepted=n_acc,
            n_drafted=total_drafted,
            acceptance_rate=n_acc / max(total_drafted, 1),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-9)

    @staticmethod
    def _uniform_draft(vocab_size: int, p: float) -> np.ndarray:
        """Approximate draft distribution as uniform over vocab for residual."""
        arr = np.full(vocab_size, p / vocab_size, dtype=np.float32)
        return arr

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
            f"SpecExecDrafter(budget={self.config.budget}, "
            f"beam_width={self.config.beam_width}, "
            f"mean_acceptance_rate={self.mean_acceptance_rate:.3f})"
        )
