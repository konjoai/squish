"""
squish/kv/block_sparse_kv.py

BlockSparseKVManager — Block-Level Sparse KV Cache for Efficient Attention.

Based on:
  "Block-Sparse Transformers" — Gray et al., OpenAI (NeurIPS 2019)
  Adapted for inference-time KV cache sparse attention:

  "Native Sparse Attention: Hardware-Aligned and Natively Trainable
   Sparse Attention" — DeepSeek AI — arXiv:2502.11089 (ICLR 2025)
   Key result: 4–8× attention FLOP reduction on long sequences with
   block-granularity sparsity masks.

  "MagicPIG: LSH Sampling for Efficient LLM Generation"
  arXiv:2410.16179 (2024)

Background
----------
Standard KV-cache attention attends over the **full** history of keys and
values, with O(S·d) memory and O(S·d) FLOPs per decode step.  As S grows
(e.g. 32K context), this dominates latency.

Block-sparse attention partitions the sequence into non-overlapping blocks
of size ``block_size``.  At each decode step, only the **top-k** blocks
(by attention score) are used, reducing the effective sequence length from
S to k · block_size.

Score function options:
  - ``"max_attn"``   : maximum raw QK dot-product within a block.
  - ``"mean_attn"``  : mean QK dot-product within a block.
  - ``"norm_attn"``  : max |QK| norm.

For causal autoregressive decoding, the current (rightmost) block is
always included regardless of score to ensure recent context is retained.

Classes
-------
``BlockSparseConfig``       — block size, top-k, score function
``SparseBlock``             — one selected block (start, end, score)
``BlockSparseStats``        — call counts, tokens saved, utilization
``BlockSparseKVManager``    — forward + prune API

Usage::

    from squish.kv.block_sparse_kv import BlockSparseConfig, BlockSparseKVManager

    cfg = BlockSparseConfig(block_size=32, top_k_blocks=8)
    manager = BlockSparseKVManager(cfg)

    # key:   (seq_len, n_heads, head_dim)
    # value: (seq_len, n_heads, head_dim)
    # query: (1, n_heads, head_dim)  — single decode step
    pruned_k, pruned_v, block_mask = manager.prune(key, value, query)
    output = manager.compute_attention(query, pruned_k, pruned_v)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

__all__ = [
    "BlockSparseConfig",
    "SparseBlock",
    "BlockSparseStats",
    "BlockSparseKVManager",
]

_SCORE_FNS = {"max_attn", "mean_attn", "norm_attn"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BlockSparseConfig:
    """Configuration for block-sparse KV attention.

    Attributes:
        block_size:   Number of tokens per block.  Must be >= 1.
        top_k_blocks: Number of highest-scored blocks to retain.  Must be >= 1.
        score_fn:     Block scoring function.  One of ``"max_attn"``,
                      ``"mean_attn"``, ``"norm_attn"``.
        always_last:  Always include the most recent block (regardless of
                      score) to preserve recent context.  Default: True.
        causal:       Apply causal masking in attention.  Default: True.
    """

    block_size: int = 32
    top_k_blocks: int = 8
    score_fn: Literal["max_attn", "mean_attn", "norm_attn"] = "max_attn"
    always_last: bool = True
    causal: bool = True

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.top_k_blocks < 1:
            raise ValueError(f"top_k_blocks must be >= 1, got {self.top_k_blocks}")
        if self.score_fn not in _SCORE_FNS:
            raise ValueError(
                f"score_fn must be one of {sorted(_SCORE_FNS)}, got '{self.score_fn}'"
            )


# ---------------------------------------------------------------------------
# Block descriptor
# ---------------------------------------------------------------------------


@dataclass
class SparseBlock:
    """A selected KV cache block.

    Attributes:
        start: Start token index (inclusive).
        end:   End token index (exclusive).
        score: Selection score (higher = more important).
    """

    start: int
    end: int
    score: float

    @property
    def size(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"SparseBlock([{self.start}:{self.end}], score={self.score:.4f})"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class BlockSparseStats:
    """Runtime statistics for BlockSparseKVManager.

    Attributes:
        total_forward_calls:  Number of ``compute_attention()`` calls.
        total_input_tokens:   Sum of seq_len of key tensors.
        total_pruned_tokens:  Sum of pruned-away tokens.
        total_blocks_selected: Sum of selected blocks across all calls.
    """

    total_forward_calls: int = 0
    total_input_tokens: int = 0
    total_pruned_tokens: int = 0
    total_blocks_selected: int = 0

    @property
    def mean_tokens_saved(self) -> float:
        if self.total_forward_calls == 0:
            return 0.0
        return self.total_pruned_tokens / self.total_forward_calls

    @property
    def mean_sparsity(self) -> float:
        if self.total_input_tokens == 0:
            return 0.0
        return self.total_pruned_tokens / self.total_input_tokens

    def __repr__(self) -> str:
        return (
            f"BlockSparseStats("
            f"calls={self.total_forward_calls}, "
            f"mean_saved={self.mean_tokens_saved:.0f}, "
            f"mean_sparsity={self.mean_sparsity:.2%})"
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _block_score(
    q: np.ndarray,
    k_block: np.ndarray,
    score_fn: str,
    scale: float,
) -> float:
    """Compute the importance score for a single key block.

    Parameters
    ----------
    q:       (n_heads, d) query (single decode step).
    k_block: (block_len, n_heads, d) key block.
    score_fn: scoring strategy.
    scale:    QK scale factor.

    Returns
    -------
    float score.
    """
    # Compute QK dot products: sum over heads
    # q: (H, d) → (H, 1, d)
    # k_block: (T, H, d) → (T, H, d)
    q_exp = q[None, :, :]  # (1, H, d)
    dots = (k_block * q_exp).sum(axis=-1) * scale  # (T, H)

    if score_fn == "max_attn":
        return float(dots.max())
    elif score_fn == "mean_attn":
        return float(dots.mean())
    else:  # norm_attn
        return float(np.abs(dots).max())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class BlockSparseKVManager:
    """Block-sparse KV attention: select top-k blocks, prune the rest.

    Parameters
    ----------
    config:
        Block-sparse configuration.
    """

    def __init__(self, config: Optional[BlockSparseConfig] = None) -> None:
        self._cfg = config or BlockSparseConfig()
        self.stats = BlockSparseStats()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_blocks(
        self, key: np.ndarray, query: np.ndarray
    ) -> List[SparseBlock]:
        """Score all key blocks by relevance to the query.

        Parameters
        ----------
        key:   (seq_k, n_heads, head_dim) float32
        query: (1, n_heads, head_dim) or (n_heads, head_dim) float32

        Returns
        -------
        List of SparseBlock sorted by descending score.
        """
        cfg = self._cfg
        key = np.asarray(key, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 3:
            q = q[0]  # (n_heads, head_dim)

        d = q.shape[-1]
        scale = 1.0 / max(math.sqrt(d), 1e-6)
        seq_k = key.shape[0]
        bs = cfg.block_size

        blocks: List[SparseBlock] = []
        for start in range(0, seq_k, bs):
            end = min(start + bs, seq_k)
            k_block = key[start:end]  # (block_len, H, d)
            score = _block_score(q, k_block, cfg.score_fn, scale)
            blocks.append(SparseBlock(start=start, end=end, score=score))

        blocks.sort(key=lambda b: b.score, reverse=True)
        return blocks

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def prune(
        self,
        key: np.ndarray,
        value: np.ndarray,
        query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select top-k blocks and return pruned KV tensors.

        Parameters
        ----------
        key:   (seq_k, n_heads, head_dim)
        value: (seq_k, n_heads, head_dim)
        query: (1 or seq_q, n_heads, head_dim)

        Returns
        -------
        pruned_key:   (selected_tokens, n_heads, head_dim)
        pruned_value: (selected_tokens, n_heads, head_dim)
        block_mask:   (seq_k,) bool array — True for selected tokens.
        """
        cfg = self._cfg
        key = np.asarray(key, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        seq_k = key.shape[0]

        # Use last query token for scoring
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 3:
            q_last = q[-1]  # (n_heads, d)
        else:
            q_last = q

        blocks = self.score_blocks(key, q_last)
        n_blocks = len(blocks)
        k = min(cfg.top_k_blocks, n_blocks)

        selected = blocks[:k]

        # Always include the last block
        if cfg.always_last and n_blocks > 0:
            last_start = (seq_k // cfg.block_size) * cfg.block_size
            last_block = SparseBlock(
                start=last_start,
                end=seq_k,
                score=float("inf"),
            )
            selected_starts = {b.start for b in selected}
            if last_start not in selected_starts:
                selected.append(last_block)

        # Build mask
        mask = np.zeros(seq_k, dtype=bool)
        for b in selected:
            mask[b.start : b.end] = True

        pruned_key = key[mask]
        pruned_value = value[mask]

        self.stats.total_forward_calls += 1
        self.stats.total_input_tokens += seq_k
        self.stats.total_pruned_tokens += seq_k - int(mask.sum())
        self.stats.total_blocks_selected += len(selected)

        return pruned_key, pruned_value, mask

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def compute_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
    ) -> np.ndarray:
        """Standard scaled dot-product attention on (already-pruned) KV.

        Parameters
        ----------
        query: (seq_q, n_heads, head_dim)
        key:   (seq_k, n_heads, head_dim)
        value: (seq_k, n_heads, head_dim)

        Returns
        -------
        output: (seq_q, n_heads, head_dim)
        """
        q = np.asarray(query, dtype=np.float32)
        k = np.asarray(key, dtype=np.float32)
        v = np.asarray(value, dtype=np.float32)

        if q.ndim == 2:
            q = q[:, None, :]
            k = k[:, None, :]
            v = v[:, None, :]
            squeeze = True
        else:
            squeeze = False

        seq_q, n_heads, d = q.shape
        scale = 1.0 / max(math.sqrt(d), 1e-6)
        out = np.zeros_like(q)

        for h in range(n_heads):
            qh = q[:, h, :]  # (sq, d)
            kh = k[:, h, :]  # (sk, d)
            vh = v[:, h, :]  # (sk, d)
            scores = (qh @ kh.T) * scale  # (sq, sk)
            if self._cfg.causal and seq_q > 1:
                qi = np.arange(seq_q)[:, None]
                ki = np.arange(k.shape[0])[None, :]
                scores = np.where(ki > qi, -1e9, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            w = np.exp(scores)
            w /= w.sum(axis=-1, keepdims=True)
            out[:, h, :] = w @ vh

        if squeeze:
            out = out[:, 0, :]
        return out

    def __repr__(self) -> str:
        return (
            f"BlockSparseKVManager("
            f"block_size={self._cfg.block_size}, "
            f"top_k={self._cfg.top_k_blocks}, "
            f"score_fn={self._cfg.score_fn!r}, "
            f"{self.stats})"
        )
