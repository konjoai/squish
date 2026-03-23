"""squish/kernels/rs_ouroboros_ngram.py — Rust-backed Ouroboros n-gram table and lookahead.

Wraps ``squish_quant_rs.ouroboros_ngram_build`` and
``squish_quant_rs.ouroboros_lookahead_f32`` with NumPy fallbacks.

Ouroboros speculative decoding maintains an online n-gram frequency table
built from verified draft tokens.  At every acceptance step, the table is
updated; draft candidates are then proposed from the most likely next token
per context.  The lookahead chain samples ``depth`` tokens in parallel using
temperature-scaled next-token distributions.

Reference: Yang et al., "Ouroboros: Speculative Decoding with Large Model
Enhanced Drafting," arXiv 2402.13720, 2024.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "OuroborosNgramConfig",
    "RustOuroborosNgram",
]

try:
    import squish_quant as _sq
    _HAS_RUST = (
        hasattr(_sq, "ouroboros_ngram_build")
        and hasattr(_sq, "ouroboros_lookahead_f32")
    )
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_ngram_build(
    token_ids: np.ndarray,
    order: int,
    max_entries: int,
) -> np.ndarray:
    """Build n-gram frequency table via Python dicts.

    Args:
        token_ids:   ``(T,)`` int32 token sequence.
        order:       N-gram order (context = order-1).
        max_entries: Maximum rows to return.

    Returns:
        ``(n_unique, order+1)`` int32 — each row is
        ``[ctx_0, ..., ctx_{order-2}, next, count]``, sorted by count desc.
    """
    toks = token_ids.tolist()
    ctx_len = order - 1
    table: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(toks) - order + 1):
        ctx = tuple(toks[i: i + ctx_len])
        next_tok = toks[i + ctx_len]
        table[ctx][next_tok] += 1
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


def _numpy_lookahead(
    logits: np.ndarray,
    temperature: float,
    seed: int,
) -> np.ndarray:
    """Sample one token per depth step.

    Args:
        logits:      ``(depth, vocab)`` float32.
        temperature: Sampling temperature.
        seed:        Random seed.

    Returns:
        ``(depth,)`` int32 draft tokens.
    """
    rng = np.random.default_rng(seed)
    depth, vocab = logits.shape
    temp = max(temperature, 1e-6)
    tokens = np.empty(depth, dtype=np.int32)
    for i in range(depth):
        row = logits[i].astype(np.float64)
        row -= row.max()
        row /= temp
        probs = np.exp(row)
        probs /= probs.sum() + 1e-9
        tokens[i] = rng.choice(vocab, p=probs)
    return tokens


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class OuroborosNgramConfig:
    """Configuration for :class:`RustOuroborosNgram`.

    Attributes:
        order:       N-gram order.
        max_entries: Max rows retained in the frequency table.
        depth:       Lookahead draft chain length.
        temperature: Sampling temperature for lookahead.
    """

    order: int = 4
    max_entries: int = 65536
    depth: int = 8
    temperature: float = 1.0


class RustOuroborosNgram:
    """Rust-accelerated Ouroboros n-gram table construction and lookahead sampling.

    Provides two operations:
    * :meth:`build`     — construct frequency table from verified tokens.
    * :meth:`lookahead` — sample a draft chain of ``depth`` tokens.

    ``build`` parallelises over independent hash shards via Rayon.
    ``lookahead`` parallelises per-depth logit sampling.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[OuroborosNgramConfig] = None) -> None:
        self._cfg = config or OuroborosNgramConfig()

    def build(
        self,
        token_ids: np.ndarray,
        order: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> np.ndarray:
        """Build n-gram frequency table from a verified token sequence.

        Args:
            token_ids:   ``(T,)`` int32 verified tokens.
            order:       N-gram order (overrides config).
            max_entries: Max rows returned (overrides config).

        Returns:
            ``(n_unique, order+1)`` int32 rows:
            ``[ctx_0, …, ctx_{order-2}, next_tok, count]``, sorted desc.
        """
        toks = np.ascontiguousarray(token_ids, dtype=np.int32).ravel()
        ord_ = int(order) if order is not None else self._cfg.order
        me = int(max_entries) if max_entries is not None else self._cfg.max_entries
        if _HAS_RUST:
            return np.asarray(_sq.ouroboros_ngram_build(toks, ord_, me), dtype=np.int32)
        return _numpy_ngram_build(toks, ord_, me)

    def lookahead(
        self,
        logits: np.ndarray,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample a draft chain from per-depth logit distributions.

        Args:
            logits:      ``(depth, vocab)`` float32 draft logits.
            temperature: Sampling temperature (overrides config).
            seed:        Random seed.

        Returns:
            ``(depth,)`` int32 sampled draft tokens.

        Raises:
            ValueError: If ``logits`` is not 2-D.
        """
        lg = np.ascontiguousarray(logits, dtype=np.float32)
        if lg.ndim != 2:
            raise ValueError(f"logits must be 2-D (depth, vocab), got {lg.shape}")
        temp = float(temperature) if temperature is not None else self._cfg.temperature
        if _HAS_RUST:
            return np.asarray(
                _sq.ouroboros_lookahead_f32(lg, temp, int(seed)), dtype=np.int32
            )
        return _numpy_lookahead(lg, temp, seed)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
