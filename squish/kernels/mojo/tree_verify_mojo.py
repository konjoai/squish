"""squish/kernels/mojo/tree_verify_mojo.py — Mojo-backed tree-speculative verifier.

Wraps the ``tree_verify`` Mojo kernel via MojoBridge with a NumPy fallback.
Verifies B draft branches in parallel via rejection sampling, returning
the longest accepted token sequence across all branches.

Reference: Miao et al., "SpecInfer: Accelerating Large Language Model
Serving with Tree-based Speculative Inference and Verification," ASPLOS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "TreeVerifyMojoConfig",
    "MojoTreeVerify",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("tree_verify")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits / max(temperature, 1e-6) - logits.max() / max(temperature, 1e-6)
    p = np.exp(x)
    return p / (p.sum() + 1e-9)


def _numpy_verify(
    draft_tokens: np.ndarray,
    draft_logits: np.ndarray,
    target_logits: np.ndarray,
    temperature: float,
    seed: int,
) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n_branches, n_draft = draft_tokens.shape
    vocab = draft_logits.shape[2]
    best: list[int] = []
    for b in range(n_branches):
        accepted: list[int] = []
        for i in range(n_draft):
            token = int(draft_tokens[b, i])
            dp = _softmax(draft_logits[b, i], temperature)
            tp = _softmax(target_logits[b, i], temperature)
            accept = min(1.0, float(tp[min(token, vocab - 1)]) / (float(dp[min(token, vocab - 1)]) + 1e-30))
            if rng.random() < accept:
                accepted.append(token)
            else:
                residual = np.clip(tp - dp, 0.0, None)
                s = residual.sum() + 1e-9
                accepted.append(int(rng.choice(vocab, p=residual / s)))
                break
        if len(accepted) > len(best):
            best = accepted
    return np.array(best, dtype=np.int32), len(best)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class TreeVerifyMojoConfig:
    """Configuration for :class:`MojoTreeVerify`.

    Attributes:
        temperature: Softmax temperature for draft/target distributions.
    """

    temperature: float = 1.0


class MojoTreeVerify:
    """Mojo-backed tree-parallel speculative decoding verifier.

    ``parallelize`` over branches; sequential rejection sampling inside
    each branch.  Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[TreeVerifyMojoConfig] = None) -> None:
        self._cfg = config or TreeVerifyMojoConfig()

    def verify(
        self,
        draft_tokens: np.ndarray,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """Verify draft branches via rejection sampling.

        Args:
            draft_tokens:  ``(B, n_draft)`` int32 draft token ids.
            draft_logits:  ``(B, n_draft, vocab)`` float32 draft logits.
            target_logits: ``(B, n_draft, vocab)`` float32 target logits.
            temperature:   Softmax temperature (None → config).
            seed:          RNG seed.

        Returns:
            ``(accepted (n,) int32, best_len int)``.
        """
        dt = np.ascontiguousarray(draft_tokens, dtype=np.int32)
        dl = np.ascontiguousarray(draft_logits, dtype=np.float32)
        tl = np.ascontiguousarray(target_logits, dtype=np.float32)
        if dt.shape[0] != dl.shape[0] or dt.shape[0] != tl.shape[0]:
            raise ValueError(
                f"Batch dimension mismatch: draft_tokens {dt.shape[0]}, "
                f"draft_logits {dl.shape[0]}, target_logits {tl.shape[0]}"
            )
        if dl.shape[2] != tl.shape[2]:
            raise ValueError(
                f"Vocab dimension mismatch: draft_logits {dl.shape[2]}, "
                f"target_logits {tl.shape[2]}"
            )
        temp = float(temperature) if temperature is not None else self._cfg.temperature
        if _kernel is not None:
            n_branches, n_draft = dt.shape
            vocab = dl.shape[2]
            out_tokens = np.zeros(n_draft, dtype=np.int32)
            best_len_buf = np.zeros(1, dtype=np.int32)
            _kernel(
                dt.ctypes.data, dl.ctypes.data, tl.ctypes.data,
                out_tokens.ctypes.data, best_len_buf.ctypes.data,
                n_branches, n_draft, vocab, temp, int(seed),
            )
            best_len = int(best_len_buf[0])
            return out_tokens[:best_len], best_len
        return _numpy_verify(dt, dl, tl, temp, int(seed))

    def acceptance_rate(
        self,
        draft_tokens: np.ndarray,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
        temperature: Optional[float] = None,
        n_trials: int = 100,
    ) -> float:
        """Estimate average acceptance rate over multiple trials.

        Runs :meth:`verify` ``n_trials`` times with different seeds and
        returns the mean fraction of draft tokens accepted.

        Args:
            draft_tokens:  ``(B, n_draft)`` int32.
            draft_logits:  ``(B, n_draft, vocab)`` float32.
            target_logits: ``(B, n_draft, vocab)`` float32.
            temperature:   Softmax temperature (None → config).
            n_trials:      Number of Monte-Carlo trials.

        Returns:
            Scalar float in ``[0, 1]``.
        """
        n_draft = draft_tokens.shape[1]
        total = 0
        for seed in range(n_trials):
            _, bl = self.verify(
                draft_tokens, draft_logits, target_logits,
                temperature=temperature, seed=seed,
            )
            total += bl
        return total / (n_trials * max(n_draft, 1))

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
