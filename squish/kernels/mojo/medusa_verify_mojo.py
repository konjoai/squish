"""squish/kernels/mojo/medusa_verify_mojo.py — Mojo-backed Medusa tree verify.

Wraps the ``medusa_verify`` Mojo kernel via MojoBridge with a NumPy fallback.

Reference: Cai et al., "Medusa: Simple LLM Inference Acceleration Framework
with Multiple Decoding Heads." ICML 2024 (arXiv 2401.10774).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "MedusaVerifyConfig",
    "MojoMedusaVerify",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("medusa_verify")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_verify(
    draft_tokens: np.ndarray,   # (n_heads,) int32 token ids
    draft_probs: np.ndarray,    # (n_heads,) float32 draft probabilities
    target_probs: np.ndarray,   # (n_heads, vocab_size) float32
    accept_threshold: float,
) -> Tuple[List[int], int]:
    """Accept/reject draft tokens against target distribution."""
    accepted: List[int] = []
    for i in range(len(draft_tokens)):
        tok = int(draft_tokens[i])
        p_draft = float(draft_probs[i])
        p_target = float(target_probs[i, tok])
        # standard speculative acceptance: accept with prob min(1, p_target/p_draft)
        ratio = p_target / max(p_draft, 1e-9)
        if ratio >= accept_threshold:
            accepted.append(tok)
        else:
            break
    return accepted, len(accepted)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class MedusaVerifyConfig:
    """Configuration for :class:`MojoMedusaVerify`.

    Attributes:
        n_heads:          Number of Medusa draft heads.
        vocab_size:       Vocabulary size.
        accept_threshold: Minimum acceptance ratio (default 0.0 → always accept
                          if p_target > 0).
    """

    n_heads: int = 2
    vocab_size: int = 32000
    accept_threshold: float = 0.0


class MojoMedusaVerify:
    """Mojo-backed Medusa tree verification pass.

    Parallelises the independent head acceptance checks across the draft
    tree.  Falls back to a sequential NumPy loop when the Mojo runtime is
    absent.
    """

    def __init__(self, config: Optional[MedusaVerifyConfig] = None) -> None:
        self._cfg = config or MedusaVerifyConfig()

    def verify(
        self,
        draft_tokens: np.ndarray,
        draft_probs: np.ndarray,
        target_probs: np.ndarray,
        accept_threshold: Optional[float] = None,
    ) -> Tuple[List[int], int]:
        """Run the speculative acceptance check.

        Args:
            draft_tokens:     Draft token IDs ``(n_heads,)`` int32.
            target_probs:     Target model logits/probs
                              ``(n_heads, vocab_size)`` float32.
            draft_probs:      Draft head probabilities ``(n_heads,)`` float32.
            accept_threshold: Override minimum acceptance ratio.

        Returns:
            ``(accepted_tokens: List[int], n_accepted: int)``
        """
        dt = np.ascontiguousarray(draft_tokens, dtype=np.int32).ravel()
        dp = np.ascontiguousarray(draft_probs, dtype=np.float32).ravel()
        tp = np.ascontiguousarray(target_probs, dtype=np.float32)
        thr = float(accept_threshold) if accept_threshold is not None else self._cfg.accept_threshold
        if _kernel is not None:
            n = int(dt.shape[0])
            accept_flags = np.zeros(n, dtype=np.int32)
            _kernel(
                dt.ctypes.data, dp.ctypes.data, tp.ctypes.data,
                accept_flags.ctypes.data, n, int(tp.shape[1]), thr,
            )
            accepted = [int(dt[i]) for i in range(n) if accept_flags[i] == 1]
            return accepted, len(accepted)
        return _numpy_verify(dt, dp, tp, thr)

    def n_heads(self) -> int:
        return self._cfg.n_heads

    def vocab_size(self) -> int:
        return self._cfg.vocab_size

    def accept_threshold(self) -> float:
        return self._cfg.accept_threshold

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
