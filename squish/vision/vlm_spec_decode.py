"""VLMSpecDecode: speculative decoding with shared visual KV prefix.

In multi-modal speculative decoding the visual prefix (image tokens) is encoded
once by the target model and then shared across all draft-tree branches.  Only
the text hypothesis tokens need to be re-evaluated per branch, reducing the
decode cost by the visual-to-text token ratio on every verification step.

The implementation is draft-framework-agnostic: the caller provides a
``draft_fn(prompt_tokens, width) → List[List[int]]`` that produces candidate
continuations and a ``verify_fn(prompt + candidates, visual_kv) → accepted``
that returns the longest accepted prefix per branch.

Reference: SpecInfer (arXiv 2305.09781), extended to multi-modal prefix sharing
as in VisionSpec (arXiv 2407.08126).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

__all__ = [
    "VLMSpecConfig",
    "VLMSpecState",
    "VLMSpecDecode",
]


@dataclass
class VLMSpecConfig:
    """Configuration for :class:`VLMSpecDecode`.

    Attributes:
        draft_width: Number of candidate continuations per speculation step.
        max_draft_tokens: Maximum token length per candidate branch.
        visual_shared: If True, the visual KV prefix is encoded once and
            reused; set False to disable prefix sharing (for ablation).
        seed: Unused; for API consistency.
    """

    draft_width: int = 4
    max_draft_tokens: int = 8
    visual_shared: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.draft_width < 1:
            raise ValueError(f"draft_width must be ≥ 1, got {self.draft_width}")
        if self.max_draft_tokens < 1:
            raise ValueError(
                f"max_draft_tokens must be ≥ 1, got {self.max_draft_tokens}"
            )


@dataclass
class VLMSpecState:
    """Per-sequence speculative decoding state.

    Attributes:
        visual_kv: Cached visual KV prefix array ``(n_vis, kv_dim)`` or
            ``None`` if not yet encoded.
        n_accepted: Cumulative accepted tokens.
        n_rejected: Cumulative rejected tokens.
        acceptance_history: Per-step boolean acceptance log.
    """

    visual_kv: Optional[np.ndarray] = None
    n_accepted: int = 0
    n_rejected: int = 0
    acceptance_history: List[bool] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        total = self.n_accepted + self.n_rejected
        return self.n_accepted / total if total > 0 else 0.0

    @property
    def total_decisions(self) -> int:
        return self.n_accepted + self.n_rejected


class VLMSpecDecode:
    """Speculative decoder with shared multi-modal visual prefix.

    Usage::

        cfg = VLMSpecConfig(draft_width=4, max_draft_tokens=8)
        spec = VLMSpecDecode(cfg)
        state = spec.new_state()
        visual_kv = spec.encode_visual(visual_tokens)
        accepted = spec.speculate(
            prompt_tokens, draft_fn, verify_fn, visual_kv, state
        )
    """

    def __init__(self, config: VLMSpecConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> VLMSpecState:
        return VLMSpecState()

    def encode_visual(self, visual_tokens: np.ndarray) -> np.ndarray:
        """Encode visual tokens into a shareable KV prefix.

        This is a stub that returns a mean-pooled representation per token.
        Replace with a real encoder (e.g. CLIP, SigLIP) in production.

        Args:
            visual_tokens: ``(n_vis, d)`` visual patch embeddings.

        Returns:
            KV prefix array ``(n_vis, d)`` (identity pass-through stub).
        """
        tokens = np.asarray(visual_tokens, dtype=np.float32)
        return tokens.copy()

    def speculate(
        self,
        prompt_tokens: List[int],
        draft_fn: Callable[[List[int], int], List[List[int]]],
        verify_fn: Callable[[List[int], np.ndarray], List[int]],
        visual_kv: np.ndarray,
        state: VLMSpecState,
    ) -> List[int]:
        """Run one speculation + verification step.

        Args:
            prompt_tokens: Current prompt token IDs.
            draft_fn: Callable ``(prompt, width) → List[List[int]]`` producing
                ``draft_width`` candidate continuations.
            verify_fn: Callable ``(full_sequence, visual_kv) → accepted_tokens``
                returning the longest accepted token list from all branches.
            visual_kv: Pre-encoded visual KV prefix from
                :meth:`encode_visual`.
            state: Mutable per-sequence state.

        Returns:
            List of accepted token IDs from this step.
        """
        candidates = draft_fn(prompt_tokens, self.config.draft_width)

        vkv = visual_kv if self.config.visual_shared else None

        best_accepted: List[int] = []
        for candidate in candidates:
            full_seq = prompt_tokens + candidate[: self.config.max_draft_tokens]
            accepted = verify_fn(full_seq, vkv)
            if len(accepted) > len(best_accepted):
                best_accepted = accepted

        n_acc = len(best_accepted)
        n_rej = sum(len(c) for c in candidates) - n_acc * len(candidates)
        n_rej = max(0, n_rej)

        state.n_accepted += n_acc
        state.n_rejected += n_rej
        for tok in best_accepted:
            state.acceptance_history.append(True)

        if state.visual_kv is None and self.config.visual_shared:
            state.visual_kv = visual_kv

        return best_accepted

    def acceptance_rate(self, state: VLMSpecState) -> float:
        return state.acceptance_rate

    def reset(self, state: VLMSpecState) -> None:
        """Reset acceptance counters and history; retain visual KV."""
        state.n_accepted = 0
        state.n_rejected = 0
        state.acceptance_history.clear()
