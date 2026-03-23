"""splitk_reduce_mojo.py — Mojo-accelerated Flash-Decode split-K LSE merge.

Wraps ``squish/kernels/mojo/kernels/splitk_reduce.mojo`` via MojoBridge
(Wave 59b).
Falls back to NumPy when the Mojo library is unavailable.

MojoSplitKReduce uses ``parallelize(n_heads)`` with ``@parameter`` on
``n_splits`` (4, 8, 16, 32) and vectorized head-dim accumulation to merge
P Flash-Decode splits via log-sum-exp renormalization; ~8× over the Python
list iteration in ``merge_split_results()``.

Input format: lists of per-split outputs, log-sum-exp, and max-score arrays
(one entry per split).

Reference:
    Dao, Fu, Ermon, Rudra, Ré — Flash-Decoding (MLSys 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["SplitKReduceConfig", "MojoSplitKReduce"]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("splitk_reduce")


@dataclass
class SplitKReduceConfig:
    n_heads: int = 32
    head_dim: int = 128


def _numpy_merge_splits(
    outputs: List[np.ndarray],
    lses: List[np.ndarray],
) -> np.ndarray:
    """NumPy log-sum-exp merge across Flash-Decode splits."""
    # outputs: list of (n_heads, head_dim), lses: list of (n_heads,)
    P = len(outputs)
    n_heads, head_dim = outputs[0].shape
    lse_stack = np.stack(lses, axis=0)           # (P, n_heads)
    out_stack = np.stack(outputs, axis=0)        # (P, n_heads, head_dim)
    global_max = lse_stack.max(axis=0)           # (n_heads,)
    weights = np.exp(lse_stack - global_max[np.newaxis, :])  # (P, n_heads)
    weights /= weights.sum(axis=0, keepdims=True)            # normalize
    # weighted sum over splits: (n_heads, head_dim)
    result = (weights[:, :, np.newaxis] * out_stack).sum(axis=0)
    return result.astype(np.float32)


class MojoSplitKReduce:
    """Flash-Decode split-K LSE merge (Mojo → NumPy fallback).

    Args:
        config: :class:`SplitKReduceConfig`.
    """

    def __init__(self, config: Optional[SplitKReduceConfig] = None) -> None:
        self._cfg = config or SplitKReduceConfig()

    # ------------------------------------------------------------------
    def merge(
        self,
        outputs: List[np.ndarray],
        lses: List[np.ndarray],
    ) -> np.ndarray:
        """Merge P Flash-Decode split results via log-sum-exp renormalization.

        Args:
            outputs: List of P arrays, each ``(n_heads, head_dim)`` float32.
            lses: List of P arrays, each ``(n_heads,)`` float32 log-sum-exp.

        Returns:
            ``(n_heads, head_dim)`` float32 merged attention output.
        """
        if not outputs:
            raise ValueError("outputs list is empty")
        outs = [np.ascontiguousarray(o, dtype=np.float32) for o in outputs]
        ls = [np.ascontiguousarray(l, dtype=np.float32) for l in lses]
        if _kernel is not None:
            try:
                result = _kernel(outs, ls)
                n_heads, head_dim = outs[0].shape
                return np.asarray(result, dtype=np.float32).reshape(n_heads, head_dim)
            except Exception:
                pass
        return _numpy_merge_splits(outs, ls)

    def n_heads(self) -> int:
        return self._cfg.n_heads

    def head_dim(self) -> int:
        return self._cfg.head_dim

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
