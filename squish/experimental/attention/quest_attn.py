"""
squish/attention/quest_attn.py

Quest: Query-Aware Sparse KV Attention.

Reference
---------
Tang et al. "Quest: Query-Aware Sparsity for Efficient Long-Context LLM
Inference." ICML 2024.

Algorithm
---------
For each decoding step the full KV cache is partitioned into fixed-size pages.
A cheap per-head similarity score is computed between the current query and
a *single representative* per page (e.g. the mean or max across the page).
Only the top-K highest-scoring pages are loaded for exact attention; the
remaining pages are skipped entirely.  This gives O(budget_pages × page_size)
attention instead of O(context_length) while preserving accuracy on most
long-context tasks.

Key properties
--------------
* Configurable budget_ratio ∈ (0, 1] — fraction of pages kept.
* Configurable page_size (default 16 tokens per page).
* page_score_fn — "mean" (fast), "max" (slightly better recall), "first".
* Exact fallback for short contexts (< min_length tokens).
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


@dataclass
class QuestConfig:
    """Configuration for QuestAttention."""

    budget_ratio: float = 0.25
    """Fraction of KV pages to keep in the sparse budget."""

    page_size: int = 16
    """Number of KV tokens per page."""

    page_score_fn: Literal["mean", "max", "first"] = "mean"
    """How to derive a page representative from its constituent K vectors."""

    min_length: int = 64
    """Contexts shorter than this skip sparsification (exact attention)."""

    head_dim: int = 64
    """Expected head dimension; used for score normalisation."""

    def __post_init__(self) -> None:
        if not 0.0 < self.budget_ratio <= 1.0:
            raise ValueError("budget_ratio must be in (0, 1]")
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")
        if self.min_length < 0:
            raise ValueError("min_length must be >= 0")


@dataclass
class QuestStats:
    """Runtime statistics counters for QuestAttention."""

    attn_calls: int = 0
    sparse_calls: int = 0
    exact_calls: int = 0
    total_pages_skipped: int = 0
    total_pages_kept: int = 0

    @property
    def sparsity(self) -> float:
        total = self.total_pages_skipped + self.total_pages_kept
        if total == 0:
            return 0.0
        return self.total_pages_skipped / total


class QuestAttention:
    """Query-aware sparse attention over a paged KV cache.

    Usage
    -----
    ::

        qa = QuestAttention()
        output = qa.attend(query, key_cache, value_cache)
    """

    def __init__(self, config: Optional[QuestConfig] = None) -> None:
        self.config = config or QuestConfig()
        self.stats = QuestStats()

    # ------------------------------------------------------------------
    # Page representative computation
    # ------------------------------------------------------------------

    def _page_reps(self, keys: np.ndarray) -> np.ndarray:
        """Compute one representative vector per page.

        Parameters
        ----------
        keys:
            Shape ``(seq_len, head_dim)``.

        Returns
        -------
        reps:
            Shape ``(n_pages, head_dim)``.
        """
        ps = self.config.page_size
        seq_len, hd = keys.shape
        n_pages = math.ceil(seq_len / ps)
        reps = np.zeros((n_pages, hd), dtype=np.float32)
        fn = self.config.page_score_fn
        for i in range(n_pages):
            page = keys[i * ps : (i + 1) * ps]  # (≤ps, hd)
            if fn == "mean":
                reps[i] = page.mean(axis=0)
            elif fn == "max":
                reps[i] = page.max(axis=0)
            else:  # "first"
                reps[i] = page[0]
        return reps

    # ------------------------------------------------------------------
    # Core attention computation
    # ------------------------------------------------------------------

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def attend(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Compute sparse query-aware attention.

        Parameters
        ----------
        query:
            Shape ``(head_dim,)`` — single query vector for one head.
        keys:
            Shape ``(seq_len, head_dim)``.
        values:
            Shape ``(seq_len, head_dim)``.

        Returns
        -------
        output:
            Shape ``(head_dim,)`` — attention-weighted value sum.
        """
        self.stats.attn_calls += 1
        seq_len = keys.shape[0]
        cfg = self.config
        scale = 1.0 / math.sqrt(cfg.head_dim)

        # Exact path for short contexts
        if seq_len <= cfg.min_length:
            self.stats.exact_calls += 1
            scores = (keys @ query) * scale          # (seq_len,)
            weights = self._softmax(scores)
            return weights @ values                  # (head_dim,)

        self.stats.sparse_calls += 1
        ps = cfg.page_size
        n_pages = math.ceil(seq_len / ps)
        budget_pages = max(1, round(n_pages * cfg.budget_ratio))

        # Score each page
        reps = self._page_reps(keys)                 # (n_pages, hd)
        page_scores = (reps @ query) * scale         # (n_pages,)

        # Select top-K pages
        top_idx = np.argpartition(page_scores, -budget_pages)[-budget_pages:]
        top_idx = np.sort(top_idx)

        self.stats.total_pages_kept += budget_pages
        self.stats.total_pages_skipped += n_pages - budget_pages

        # Gather selected rows
        sel_rows: List[np.ndarray] = []
        sel_vals: List[np.ndarray] = []
        for pi in top_idx:
            sel_rows.append(keys[pi * ps : (pi + 1) * ps])
            sel_vals.append(values[pi * ps : (pi + 1) * ps])

        k_sel = np.concatenate(sel_rows, axis=0)  # (budget_tokens, hd)
        v_sel = np.concatenate(sel_vals, axis=0)
        scores = (k_sel @ query) * scale
        weights = self._softmax(scores)
        return weights @ v_sel

    def reset_stats(self) -> None:
        """Reset runtime counters."""
        self.stats = QuestStats()
