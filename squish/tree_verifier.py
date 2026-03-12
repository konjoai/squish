"""TreeVerifier — Batched tree-parallel speculative verification.

In speculative decoding, a draft model generates a token tree (multiple
candidates branching from the same prefix).  The target model verifies all
branches in one batched forward pass, accepting the longest valid prefix of
each branch via rejection sampling.

Reference:
    Chen et al., "Accelerating Large Language Model Decoding with Speculative
    Sampling", arXiv 2302.01318, 2023.  https://arxiv.org/abs/2302.01318

Usage::

    from squish.tree_verifier import TreeVerifier, VerifyConfig, TokenTree
    import numpy as np

    cfg      = VerifyConfig(n_draft_tokens=4, n_branches=3, temperature=1.0)
    verifier = TreeVerifier(cfg)

    vocab    = 32000
    tree     = TokenTree(
        tokens=np.random.randint(0, vocab, (3, 4), dtype=np.int32),
        draft_logits=np.random.randn(3, 4, vocab).astype(np.float32),
    )
    target_logits = np.random.randn(3, 4, vocab).astype(np.float32)
    result = verifier.verify(tree, target_logits)
    print(f"accepted={result.n_accepted}, rate={result.acceptance_rate:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "VerifyConfig",
    "TokenTree",
    "VerifyResult",
    "TreeVerifier",
    "VerifyStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VerifyConfig:
    """Configuration for tree-parallel speculative verification.

    Attributes:
        n_draft_tokens: Number of draft tokens generated per branch.
        n_branches:     Number of candidate branches in the token tree.
        temperature:    Sampling temperature applied to logits during
                        rejection-sampling acceptance checks.
    """

    n_draft_tokens: int = 4
    n_branches: int = 3
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.n_draft_tokens < 1:
            raise ValueError(
                f"n_draft_tokens must be >= 1; got {self.n_draft_tokens}"
            )
        if self.n_branches < 1:
            raise ValueError(f"n_branches must be >= 1; got {self.n_branches}")
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0; got {self.temperature}"
            )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenTree:
    """A tree of candidate draft tokens and their associated logits.

    Attributes:
        tokens:        Draft token IDs, shape ``(n_branches, n_draft_tokens)``,
                       int32.
        draft_logits:  Raw draft model logits, shape
                       ``(n_branches, n_draft_tokens, vocab_size)``, float32.
    """

    tokens: np.ndarray
    draft_logits: np.ndarray

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ValueError(
                f"tokens must be 2-D (n_branches, n_draft_tokens); "
                f"got shape {self.tokens.shape}"
            )
        if self.draft_logits.ndim != 3:
            raise ValueError(
                f"draft_logits must be 3-D (n_branches, n_draft_tokens, vocab); "
                f"got shape {self.draft_logits.shape}"
            )
        if self.tokens.shape != self.draft_logits.shape[:2]:
            raise ValueError(
                f"tokens shape {self.tokens.shape} is inconsistent with "
                f"draft_logits leading dims {self.draft_logits.shape[:2]}"
            )


@dataclass
class VerifyResult:
    """Outcome of a single tree-verification pass.

    Attributes:
        accepted_tokens: Accepted token IDs, shape ``(n_accepted,)``, int32.
        n_accepted:      Number of tokens accepted in this step.
        acceptance_rate: Fraction of draft tokens accepted (best branch /
                         n_draft_tokens).
    """

    accepted_tokens: np.ndarray
    n_accepted: int
    acceptance_rate: float


@dataclass
class VerifyStats:
    """Cumulative statistics collected by a :class:`TreeVerifier` instance.

    Attributes:
        total_verifications: Number of :meth:`~TreeVerifier.verify` calls made.
        total_accepted:      Total draft tokens accepted across all calls.
        total_draft:         Total draft tokens presented across all calls.
    """

    total_verifications: int = 0
    total_accepted: int = 0
    total_draft: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Cumulative per-token acceptance rate across all verification calls.

        Returns 0.0 when no tokens have been presented yet.
        """
        if self.total_draft == 0:
            return 0.0
        return self.total_accepted / self.total_draft


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    Args:
        logits:      1-D float array of shape ``(vocab_size,)``.
        temperature: Positive temperature scalar.

    Returns:
        Probability vector, float32, summing to 1.
    """
    scaled = logits.astype(np.float64) / temperature
    shifted = scaled - np.max(scaled)
    exp_v = np.exp(shifted)
    return (exp_v / exp_v.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# TreeVerifier
# ---------------------------------------------------------------------------


class TreeVerifier:
    """Batched tree-parallel speculative verifier using rejection sampling.

    For each branch in the token tree the verifier applies standard speculative
    rejection sampling: a draft token ``t`` at position ``i`` is accepted with
    probability ``min(1, p_target(t) / p_draft(t))``.  On rejection, a
    correction token is sampled from the residual distribution
    ``max(0, p_target - p_draft) / Z``.  The branch with the longest
    accepted prefix is returned.

    Args:
        config: :class:`VerifyConfig` instance controlling draft count,
                branch count, and sampling temperature.
    """

    def __init__(self, config: VerifyConfig) -> None:
        self._config = config
        self._stats = VerifyStats()
        self._rng = np.random.default_rng()

    def verify(
        self,
        tree: TokenTree,
        target_logits: np.ndarray,
    ) -> VerifyResult:
        """Verify a draft token tree against target model logits.

        Rejection-sampling acceptance is applied token by token within each
        branch.  The branch producing the longest accepted prefix is returned.
        If multiple branches tie, the first (lowest index) branch wins.

        Args:
            tree:          :class:`TokenTree` produced by the draft model.
            target_logits: Target model logits, shape
                           ``(n_branches, n_draft_tokens, vocab_size)``,
                           float32.  Must match ``tree.draft_logits.shape``.

        Returns:
            :class:`VerifyResult` with the accepted token sequence, count,
            and per-step acceptance rate.

        Raises:
            ValueError: if ``target_logits`` shape does not match
                        ``tree.draft_logits.shape``.
        """
        if target_logits.shape != tree.draft_logits.shape:
            raise ValueError(
                f"target_logits shape {target_logits.shape} must match "
                f"draft_logits shape {tree.draft_logits.shape}"
            )

        cfg = self._config
        n_branches, n_draft = tree.tokens.shape
        best_accepted: np.ndarray = np.empty(0, dtype=np.int32)
        best_count: int = 0

        for b in range(n_branches):
            accepted: list[int] = []
            for i in range(n_draft):
                token = int(tree.tokens[b, i])
                p_draft = _softmax(tree.draft_logits[b, i], cfg.temperature)
                p_target = _softmax(target_logits[b, i], cfg.temperature)

                p_d_t = float(p_draft[token])
                p_t_t = float(p_target[token])

                accept_prob = min(1.0, p_t_t / (p_d_t + 1e-30))
                if self._rng.random() < accept_prob:
                    accepted.append(token)
                else:
                    # Sample a correction token from the residual distribution.
                    residual = np.maximum(0.0, p_target - p_draft)
                    residual_sum = float(residual.sum())
                    if residual_sum > 1e-30:
                        residual /= residual_sum
                        correction = int(
                            self._rng.choice(len(residual), p=residual)
                        )
                        accepted.append(correction)
                    break

            if len(accepted) > best_count:
                best_count = len(accepted)
                best_accepted = np.array(accepted, dtype=np.int32)

        total_draft = n_branches * n_draft
        rate = best_count / max(n_draft, 1)

        self._stats.total_verifications += 1
        self._stats.total_accepted += best_count
        self._stats.total_draft += total_draft

        return VerifyResult(
            accepted_tokens=best_accepted,
            n_accepted=best_count,
            acceptance_rate=rate,
        )

    @property
    def stats(self) -> VerifyStats:
        """Cumulative verification statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset all cumulative statistics to zero."""
        self._stats = VerifyStats()
