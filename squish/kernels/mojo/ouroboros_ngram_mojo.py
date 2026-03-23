"""squish/kernels/mojo/ouroboros_ngram_mojo.py — Mojo-backed Ouroboros n-gram.

Wraps ``ouroboros_ngram_build`` and ``ouroboros_lookahead`` Mojo kernels via
MojoBridge with NumPy fallbacks.  Provides online n-gram table construction
from verified draft tokens and parallel depth-position draft sampling.

Reference: Yang et al., "Ouroboros: Speculative Decoding with Large Model
Enhanced Drafting," arXiv 2402.13720, 2024.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "OuroborosNgramMojoConfig",
    "MojoOuroborosNgram",
]

_bridge = MojoBridge()
_build_kernel = _bridge.load_kernel("ouroboros_ngram_build")
_lookahead_kernel = _bridge.load_kernel("ouroboros_lookahead")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_build(token_ids: np.ndarray, order: int, max_entries: int) -> np.ndarray:
    toks = token_ids.tolist()
    ctx_len = order - 1
    table: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(toks) - order + 1):
        ctx = tuple(toks[i: i + ctx_len])
        table[ctx][toks[i + ctx_len]] += 1
    rows = []
    for ctx, next_map in table.items():
        for next_tok, cnt in next_map.items():
            rows.append(list(ctx) + [next_tok, cnt])
    rows.sort(key=lambda r: r[-1], reverse=True)
    if max_entries > 0:
        rows = rows[:max_entries]
    if not rows:
        return np.zeros((0, order + 1), dtype=np.int32)
    return np.array(rows, dtype=np.int32)


def _numpy_lookahead(logits: np.ndarray, temperature: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth, vocab = logits.shape
    temp = max(temperature, 1e-6)
    tokens = np.empty(depth, dtype=np.int32)
    for i in range(depth):
        row = logits[i].astype(np.float64)
        row = row - row.max()
        row /= temp
        probs = np.exp(row)
        probs /= probs.sum() + 1e-9
        tokens[i] = rng.choice(vocab, p=probs)
    return tokens


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class OuroborosNgramMojoConfig:
    """Configuration for :class:`MojoOuroborosNgram`.

    Attributes:
        order:       N-gram order.
        max_entries: Max rows in frequency table.
        depth:       Lookahead chain length.
        temperature: Sampling temperature.
    """

    order: int = 4
    max_entries: int = 65536
    depth: int = 8
    temperature: float = 1.0


class MojoOuroborosNgram:
    """Mojo-backed Ouroboros n-gram table and lookahead sampling.

    Uses ``parallelize`` for shard-parallel n-gram ingestion and
    parallel depth-position sampling.  Falls back to NumPy when
    the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[OuroborosNgramMojoConfig] = None) -> None:
        self._cfg = config or OuroborosNgramMojoConfig()

    def build(
        self,
        token_ids: np.ndarray,
        order: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> np.ndarray:
        """Build n-gram frequency table.

        Args:
            token_ids:   ``(T,)`` int32 token sequence.
            order:       N-gram order (overrides config).
            max_entries: Max rows returned (overrides config).

        Returns:
            ``(n_unique, order+1)`` int32 rows sorted by frequency desc.
        """
        toks = np.ascontiguousarray(token_ids, dtype=np.int32).ravel()
        ord_ = int(order) if order is not None else self._cfg.order
        me = int(max_entries) if max_entries is not None else self._cfg.max_entries
        if _build_kernel is not None:
            n_rows = len(toks)  # upper bound
            out = np.zeros((n_rows, ord_ + 1), dtype=np.int32)
            actual = _build_kernel(toks.ctypes.data, out.ctypes.data, len(toks), ord_, me)
            return out[:actual]
        return _numpy_build(toks, ord_, me)

    def lookahead(
        self,
        logits: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample a draft chain from depth-position logit distributions.

        Args:
            logits:      ``(depth, vocab)`` float32.
            temperature: Sampling temperature (overrides config).
            seed:        Random seed.

        Returns:
            ``(depth,)`` int32 draft tokens.

        Raises:
            ValueError: If ``logits`` is not 2-D.
        """
        lg = np.ascontiguousarray(logits, dtype=np.float32)
        if lg.ndim != 2:
            raise ValueError(f"logits must be 2-D (depth, vocab), got {lg.shape}")
        depth, vocab = lg.shape
        temp = float(temperature) if temperature is not None else self._cfg.temperature
        if _lookahead_kernel is not None:
            out = np.empty(depth, dtype=np.int32)
            _lookahead_kernel(lg.ctypes.data, out.ctypes.data, depth, vocab, temp, seed)
            return out
        return _numpy_lookahead(lg, temp, seed)

    def backend(self) -> str:
        return "mojo" if (_build_kernel is not None and _lookahead_kernel is not None) else "numpy"
