"""squish/kernels/rs_tree_verify.py — Rust-backed tree speculative decoding verifier.

Wraps ``squish_quant_rs.tree_verify_softmax_f32`` with a NumPy fallback.

Tree-parallel speculative decoding verifies B draft branches in parallel,
applying token-level rejection sampling at each draft position.  The
longest accepted sequence across all branches is returned.
Rayon parallelises branch verification.

Reference: Miao et al., "SpecInfer: Accelerating Large Language Model
Serving with Tree-based Speculative Inference and Verification," ASPLOS
2024; Chen et al., "Sequoia: Scalable, Robust, and Hardware-Efficient
Speculative Decoding," NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "TreeVerifyConfig",
    "RustTreeVerify",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "tree_verify_softmax_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits / max(temperature, 1e-6)
    x = x - x.max()
    probs = np.exp(x)
    return probs / (probs.sum() + 1e-9)


def _numpy_tree_verify(
    draft_tokens: np.ndarray,
    draft_logits: np.ndarray,
    target_logits: np.ndarray,
    temperature: float,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """Reference rejection-sampling tree-speculative verification."""
    rng = np.random.default_rng(seed)
    n_branches, n_draft = draft_tokens.shape
    vocab = draft_logits.shape[2]
    best: list[int] = []
    for b in range(n_branches):
        accepted: list[int] = []
        for i in range(n_draft):
            token = int(draft_tokens[b, i])
            d_probs = _softmax(draft_logits[b, i], temperature)
            t_probs = _softmax(target_logits[b, i], temperature)
            p_d = float(d_probs[min(token, vocab - 1)])
            p_t = float(t_probs[min(token, vocab - 1)])
            accept_prob = min(1.0, p_t / (p_d + 1e-30))
            if rng.random() < accept_prob:
                accepted.append(token)
            else:
                residual = np.clip(t_probs - d_probs, 0.0, None)
                r_sum = residual.sum() + 1e-9
                correction = int(rng.choice(vocab, p=residual / r_sum))
                accepted.append(correction)
                break
        if len(accepted) > len(best):
            best = accepted
    return np.array(best, dtype=np.int32), len(best)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class TreeVerifyConfig:
    """Configuration for :class:`RustTreeVerify`.

    Attributes:
        temperature: Softmax temperature for draft/target distributions.
    """

    temperature: float = 1.0


class RustTreeVerify:
    """Rust-accelerated tree-parallel speculative decoding verifier.

    Verifies B draft branches in parallel using rejection sampling.
    Returns the longest accepted token sequence across all branches plus
    its length.  Rayon parallelises over branches.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[TreeVerifyConfig] = None) -> None:
        self._cfg = config or TreeVerifyConfig()

    def verify(
        self,
        draft_tokens: np.ndarray,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """Verify draft branches and return the best accepted sequence.

        Args:
            draft_tokens:  Draft token ids ``(B, n_draft)`` int32.
            draft_logits:  Draft logit tensors ``(B, n_draft, vocab)`` float32.
            target_logits: Target logit tensors ``(B, n_draft, vocab)`` float32.
            temperature:   Softmax temperature (None → use config).
            seed:          RNG seed for reproducible sampling.

        Returns:
            Tuple of
            - ``accepted`` — best accepted token sequence ``(n,)`` int32
            - ``best_len`` — length of accepted sequence

        Raises:
            ValueError: If draft_tokens, draft_logits, target_logits shapes
                are inconsistent.
        """
        dt = np.ascontiguousarray(draft_tokens, dtype=np.int32)
        dl = np.ascontiguousarray(draft_logits, dtype=np.float32)
        tl = np.ascontiguousarray(target_logits, dtype=np.float32)
        if dt.shape[0] != dl.shape[0] or dt.shape[0] != tl.shape[0]:
            raise ValueError("B dimension mismatch across draft_tokens / draft_logits / target_logits")
        if dt.shape[1] != dl.shape[1] or dt.shape[1] != tl.shape[1]:
            raise ValueError("n_draft dimension mismatch")
        if dl.shape[2] != tl.shape[2]:
            raise ValueError(
                f"vocab mismatch: draft={dl.shape[2]}, target={tl.shape[2]}"
            )
        temp = float(temperature) if temperature is not None else self._cfg.temperature
        if _HAS_RUST:
            tok, bl = _sq.tree_verify_softmax_f32(dt, dl, tl, temp, int(seed))
            return np.asarray(tok, dtype=np.int32), int(bl)
        return _numpy_tree_verify(dt, dl, tl, temp, int(seed))

    def acceptance_rate(
        self,
        draft_tokens: np.ndarray,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
        temperature: Optional[float] = None,
        n_trials: int = 100,
    ) -> float:
        """Estimate mean acceptance rate over n_trials Monte-Carlo runs."""
        lengths = []
        for seed in range(n_trials):
            _, best_len = self.verify(
                draft_tokens, draft_logits, target_logits, temperature, seed=seed
            )
            lengths.append(best_len)
        n_draft = draft_tokens.shape[1]
        return float(np.mean(lengths)) / max(n_draft, 1)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
