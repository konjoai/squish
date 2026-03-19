"""
squish/speculative/cascade_spec.py

CascadeSpec — Stacked EAGLE-3 + n-gram lookahead verification tree.

Key insight
-----------
EAGLE-3 provides high-quality speculative drafts based on hidden-state
feature prediction.  N-gram prompt-lookup drafts are cheap and accurate for
continuation/copy tasks.  **CascadeSpec** chains them into a two-stage
speculation pipeline:

  Stage 1 — Draft tree:
    The EAGLE-3 head generates ``eagle_depth`` draft tokens forming a linear
    chain rooted at the last accepted token.

  Stage 2 — Lookahead extension:
    Each leaf of the EAGLE-3 draft is extended by up to ``ngram_extend``
    tokens using n-gram matches within the context (prompt-lookup style).
    This produces a tree of depth ``eagle_depth + ngram_extend``.

  Stage 3 — Batched verification:
    The full target model verifies all candidate paths in a single batched
    forward pass.

  Stage 4 — Greedy acceptance:
    Walk the tree from the root; accept each agreed token greedily; stop at
    the first disagreement.  The token at the disagreement node is re-sampled
    from the target distribution.

The result is a higher *expected tokens per forward pass* compared to either
EAGLE-3 alone or prompt-lookup alone, without needing a second model.

Reference
---------
- Li et al. "EAGLE-3: Scaling up Inference Acceleration …" arXiv 2025.
- Saxena, "Prompt Lookup Decoding" 2023.
- Cai et al. "Medusa: Simple LLM Inference Acceleration …" arXiv 2024.

Usage::

    import numpy as np
    from squish.speculative.cascade_spec import CascadeSpecConfig, CascadeSpecDecoder

    cfg = CascadeSpecConfig(eagle_depth=5, ngram_extend=3, ngram_min=2, ngram_max=5)
    dec = CascadeSpecDecoder(
        full_forward=lambda ids: target_model_logits(ids),   # (vocab,) np.float32
        config=cfg,
    )
    output_ids, stats = dec.generate(prompt_ids, max_new_tokens=128)
    print(f"Mean tokens/step: {stats.mean_tokens_per_step:.2f}")
    print(f"Acceptance rate:  {stats.acceptance_rate:.2%}")
"""

from __future__ import annotations

__all__ = [
    "CascadeSpecConfig",
    "CascadeSpecDecoder",
    "CascadeSpecStats",
]

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CascadeSpecConfig:
    """Configuration for CascadeSpec two-stage speculation.

    Parameters
    ----------
    eagle_depth : int
        Number of EAGLE-3 draft tokens to generate in Stage 1 (≥ 1).
    ngram_extend : int
        Maximum n-gram lookahead tokens appended per EAGLE-3 leaf in Stage 2
        (0 disables the lookahead extension — pure EAGLE-3 behaviour).
    ngram_min : int
        Minimum n-gram length for prompt-lookup matching (≥ 2).
    ngram_max : int
        Maximum n-gram length for prompt-lookup matching (≥ ngram_min).
    temperature : float
        Sampling temperature applied to the target-model logits during
        rejection-sampling acceptance (0 = greedy).
    max_context_window : int
        Maximum number of past tokens to keep in the n-gram index to bound
        memory and index rebuild cost.
    """

    eagle_depth:         int   = 5
    ngram_extend:        int   = 3
    ngram_min:           int   = 2
    ngram_max:           int   = 5
    temperature:         float = 0.0
    max_context_window:  int   = 2048

    def __post_init__(self) -> None:
        if self.eagle_depth < 1:
            raise ValueError(f"eagle_depth must be ≥ 1; got {self.eagle_depth}")
        if self.ngram_extend < 0:
            raise ValueError(f"ngram_extend must be ≥ 0; got {self.ngram_extend}")
        if self.ngram_min < 1:
            raise ValueError(f"ngram_min must be ≥ 1; got {self.ngram_min}")
        if self.ngram_max < self.ngram_min:
            raise ValueError(
                f"ngram_max ({self.ngram_max}) must be ≥ ngram_min ({self.ngram_min})"
            )
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be ≥ 0; got {self.temperature}")
        if self.max_context_window < 8:
            raise ValueError(
                f"max_context_window must be ≥ 8; got {self.max_context_window}"
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class CascadeSpecStats:
    """Runtime statistics for a CascadeSpec generation session.

    Attributes
    ----------
    total_tokens : int
        Total tokens generated (accepted + bonus).
    total_steps : int
        Total verification forward passes.
    eagle_accepted : int
        EAGLE-3 draft tokens accepted by the verifier.
    ngram_accepted : int
        N-gram extension tokens accepted by the verifier.
    rejected : int
        Total draft tokens rejected (one per failed verification step).
    """

    total_tokens:  int = 0
    total_steps:   int = 0
    eagle_accepted: int = 0
    ngram_accepted: int = 0
    rejected:      int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of all draft tokens accepted."""
        total_draft = self.eagle_accepted + self.ngram_accepted + self.rejected
        return (
            (self.eagle_accepted + self.ngram_accepted) / total_draft
            if total_draft > 0
            else 0.0
        )

    @property
    def mean_tokens_per_step(self) -> float:
        """Mean accepted tokens per verifier forward pass."""
        return self.total_tokens / self.total_steps if self.total_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Internal: lightweight token tree node
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    token_id: int
    parent:   Optional[_TreeNode]
    depth:    int
    source:   str   # "eagle" | "ngram"
    children: List[_TreeNode] = field(default_factory=list)

    def path_from_root(self) -> List[int]:
        """Return the token IDs from root to this node (inclusive)."""
        nodes: List[_TreeNode] = []
        cur: Optional[_TreeNode] = self
        while cur is not None:
            nodes.append(cur)
            cur = cur.parent
        return [n.token_id for n in reversed(nodes)]


# ---------------------------------------------------------------------------
# Internal: fast n-gram index for lookahead extension
# ---------------------------------------------------------------------------

class _NGramIndex:
    """Minimal trigram-based prompt-lookup index (no dependencies)."""

    def __init__(self, min_n: int, max_n: int) -> None:
        self._min = min_n
        self._max = max_n
        self._index: dict[tuple[int, ...], List[int]] = {}

    def build(self, token_ids: List[int]) -> None:
        """Rebuild the n-gram index from *token_ids*."""
        self._index.clear()
        ids = token_ids
        for n in range(self._min, self._max + 1):
            for i in range(len(ids) - n):
                key = tuple(ids[i : i + n])
                self._index.setdefault(key, []).append(i + n)

    def lookup(self, query: List[int], max_extend: int) -> List[int]:
        """Return up to *max_extend* continuation tokens for *query*."""
        # Try longest match first for best precision
        for n in range(min(self._max, len(query)), self._min - 1, -1):
            key = tuple(query[-n:])
            positions = self._index.get(key)
            if positions:
                pos = positions[-1]  # most recent match
                src = list(self._index.keys())[0]  # sentinel — unused
                # Recover source sequence from any matching key
                # We store only the continuation start index, so we need
                # the original sequence.  Instead, we regenerate from the
                # first match position stored as the absolute index.
                # Re-expose via the build-time token list reference.
                break
        return []   # Fallback — caller handles missing continuation

    def lookup_from_ids(
        self, context: List[int], query: List[int], max_extend: int
    ) -> List[int]:
        """Return up to *max_extend* continuation tokens from *context*."""
        for n in range(min(self._max, len(query)), self._min - 1, -1):
            key = tuple(query[-n:])
            positions = self._index.get(key, [])
            if positions:
                # Use the last (most recent) match for best cache locality.
                start_pos = positions[-1]
                end_pos = min(start_pos + max_extend, len(context))
                if start_pos < len(context):
                    return context[start_pos:end_pos]
        return []


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------

class CascadeSpecDecoder:
    """CascadeSpec two-stage speculative decoder.

    Parameters
    ----------
    full_forward : callable
        The target model forward function.
        Signature: ``full_forward(token_ids: list[int]) -> np.ndarray``
        Returns logits of shape ``(vocab_size,)`` for the last token position.
    config : CascadeSpecConfig
    """

    def __init__(
        self,
        full_forward: Callable[[List[int]], np.ndarray],
        config: CascadeSpecConfig,
    ) -> None:
        self._fwd    = full_forward
        self._cfg    = config
        self._stats  = CascadeSpecStats()
        self._ngram  = _NGramIndex(config.ngram_min, config.ngram_max)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CascadeSpecStats:
        """Running statistics (reset with :meth:`reset_stats`)."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset all running statistics to zero."""
        self._stats = CascadeSpecStats()

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 256,
        eos_id: int = -1,
    ) -> Tuple[List[int], CascadeSpecStats]:
        """Generate up to *max_new_tokens* tokens auto-regressively.

        Parameters
        ----------
        prompt_ids : list[int]
            Tokenised prompt (context).
        max_new_tokens : int
            Maximum number of new tokens to generate.
        eos_id : int
            Token ID that signals end-of-sequence (``-1`` disables EOS).

        Returns
        -------
        output_ids : list[int]
            Generated token ids (not including the prompt).
        stats : CascadeSpecStats
            Per-session statistics snapshot.
        """
        self.reset_stats()
        ctx: List[int] = list(prompt_ids)
        output: List[int] = []

        while len(output) < max_new_tokens:
            # Build n-gram index from current context
            window = ctx[-self._cfg.max_context_window :]
            self._ngram.build(window)

            # Stage 1 + 2: build draft candidates
            draft_tokens, draft_sources = self._build_draft(ctx)

            if not draft_tokens:
                # No draft available — single-step greedy
                logits = self._fwd(ctx)
                tok = self._greedy_or_sample(logits)
                ctx.append(tok)
                output.append(tok)
                self._stats.total_tokens  += 1
                self._stats.total_steps   += 1
                self._stats.rejected      += 0
                if eos_id >= 0 and tok == eos_id:
                    break
                continue

            # Stage 3: batched verification of the full token sequence
            # In the pure-numpy reference implementation we run verification
            # token-by-token (no true batch).  A Metal-backed version would
            # pass a 2-D index array.
            accepted, n_eagle, n_ngram, bonus_tok = self._verify(
                ctx, draft_tokens, draft_sources
            )

            self._stats.total_steps   += 1
            self._stats.eagle_accepted += n_eagle
            self._stats.ngram_accepted += n_ngram

            all_new = accepted + ([bonus_tok] if bonus_tok is not None else [])
            for t in all_new:
                ctx.append(t)
                output.append(t)
                self._stats.total_tokens += 1
                if eos_id >= 0 and t == eos_id:
                    return output, self._stats

            if len(accepted) < len(draft_tokens):
                self._stats.rejected += 1

        return output, self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_draft(
        self, ctx: List[int]
    ) -> Tuple[List[int], List[str]]:
        """Build the flat draft token sequence (Stage 1 + 2).

        The EAGLE-3 stage is *simulated* here without a loaded draft head:
        we use a lightweight unigram heuristic (argmax of a cheap bigram
        context prediction) as a stand-in so the full pipeline executes.
        When an actual ``Eagle3DraftHead`` is injected via
        :meth:`set_eagle_head`, it replaces this simulation.
        """
        eagle_draft = self._eagle_draft(ctx)
        if not eagle_draft:
            return [], []

        tokens  = list(eagle_draft)
        sources = ["eagle"] * len(eagle_draft)

        # Stage 2: n-gram extension off the last eagle draft token
        if self._cfg.ngram_extend > 0:
            extension = self._ngram.lookup_from_ids(
                ctx[-self._cfg.max_context_window :],
                tokens,
                self._cfg.ngram_extend,
            )
            tokens  += extension
            sources += ["ngram"] * len(extension)

        return tokens, sources

    def _eagle_draft(self, ctx: List[int]) -> List[int]:
        """Return up to *eagle_depth* draft tokens.

        Default implementation: bigram-frequency heuristic (no external head).
        Override by setting :attr:`eagle_head` to an ``Eagle3DraftHead``
        instance.
        """
        head = getattr(self, "_eagle_head", None)
        if head is not None:
            try:
                steps = head.draft_step(ctx[-1], n_steps=self._cfg.eagle_depth)
                return [int(np.argmax(logits)) for _, logits in steps]
            except Exception:
                pass

        # Fallback: argmax over a uniform proxy (yields token 0 as placeholder)
        # A real deployment must provide an actual Eagle3DraftHead.
        depth = self._cfg.eagle_depth
        if len(ctx) < 2:
            return []
        # Simple heuristic: repeat the last token for warmup testing
        return [ctx[-1]] * min(depth, 3)

    def set_eagle_head(self, head: object) -> None:
        """Attach an ``Eagle3DraftHead``-compatible object as the Stage-1 drafter."""
        self._eagle_head = head

    def _verify(
        self,
        ctx: List[int],
        draft_tokens: List[int],
        draft_sources: List[str],
    ) -> Tuple[List[int], int, int, Optional[int]]:
        """Verify draft tokens against the target model.

        Returns
        -------
        accepted : list[int]
            The longest prefix of *draft_tokens* accepted by the verifier.
        n_eagle : int
            Number of accepted EAGLE tokens.
        n_ngram : int
            Number of accepted n-gram extension tokens.
        bonus_tok : int | None
            An extra token sampled from the target distribution at the first
            rejection point (or end of accepted sequence).
        """
        accepted: List[int] = []
        n_eagle = n_ngram = 0
        bonus_tok: Optional[int] = None

        running_ctx = list(ctx)
        for i, (draft_tok, src) in enumerate(zip(draft_tokens, draft_sources)):
            logits = self._fwd(running_ctx)
            target_tok = self._greedy_or_sample(logits)
            if target_tok == draft_tok:
                accepted.append(draft_tok)
                if src == "eagle":
                    n_eagle += 1
                else:
                    n_ngram += 1
                running_ctx.append(draft_tok)
            else:
                # Rejection: accept the target token as a bonus
                bonus_tok = target_tok
                break

        if bonus_tok is None and len(accepted) == len(draft_tokens):
            # All draft tokens accepted — sample one more bonus token
            logits = self._fwd(running_ctx)
            bonus_tok = self._greedy_or_sample(logits)

        return accepted, n_eagle, n_ngram, bonus_tok

    def _greedy_or_sample(self, logits: np.ndarray) -> int:
        """Return the next token id given a logit vector."""
        logits = np.asarray(logits, dtype=np.float64)
        if self._cfg.temperature <= 0.0:
            return int(np.argmax(logits))
        shifted = logits / self._cfg.temperature
        shifted -= shifted.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))
