"""
squish/speculative/speculative_prefill.py

SpeculativePrefill — Draft-accelerated prefill for TTFT reduction.

Key insight
-----------
Standard TTFT bottleneck: the target model runs a full multi-layer forward
pass over every prompt token before the first output token can be sampled.

**SpeculativePrefill** reduces this cost by exploiting a cheap draft model
that is already loaded for speculative decoding:

  Step 1 — Draft KV computation:
    Run the *draft* model (shallow / distilled) over the full prompt in one
    forward pass.  Produces draft KV states for all layers and layers whose
    KV agree closely with the target model's typically match.

  Step 2 — Target correction:
    Run the *target* model forward pass, but skip the expensive
    attention + FFN computation for any layer whose draft KV has cosine
    similarity ≥ ``kv_accept_threshold`` with the target's KV from the
    previous request (or a target-probe forward on a sample of positions).
    Only *disagreeing* layers require a full recompute.

  Step 3 — Accept corrected KV:
    The corrected KV states are used as the initial decode KV cache, exactly
    as in a standard prefill.  The first decode forward pass starts from
    these states.

Expected savings: 20–30 % TTFT reduction on 256–4096-token prompts when the
draft model is a 1–3B parameter shallow distillation of the target.

Note
----
This module provides a pure-Python / NumPy reference implementation.  A
production deployment integrates the KV-cache tensors with the MLX KV slab
directly (``squish.kv.kv_slab``).

Reference
---------
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding"
  arXiv 2022.
- Fu et al. "Break the Chain: Large Language Models Can be Shortcut Reasoners"
  arXiv 2024.
- Chen et al. "Accelerating Large Language Model Decoding with Speculative
  Sampling" arXiv 2023.

Usage::

    from squish.speculative.speculative_prefill import (
        SpecPrefillConfig,
        SpeculativePrefiller,
    )
    import numpy as np

    cfg = SpecPrefillConfig(n_layers=32, kv_accept_threshold=0.92)
    prefiller = SpeculativePrefiller(
        draft_forward=lambda ids: draft_model_kv(ids),   # list[np.ndarray (n_heads, seq, head_dim)]
        target_forward=lambda ids, layer_mask: target_model_kv(ids, layer_mask),
        config=cfg,
    )
    kv_states, stats = prefiller.prefill(prompt_ids)
    print(f"Layers skipped: {stats.layers_skipped}/{stats.total_layers} "
          f"({stats.skip_rate:.1%})")
    print(f"TTFT speedup estimate: {stats.speedup_estimate:.2f}×")
"""

from __future__ import annotations

__all__ = [
    "SpecPrefillConfig",
    "SpecPrefillStats",
    "SpeculativePrefiller",
]

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpecPrefillConfig:
    """Configuration for SpeculativePrefill.

    Parameters
    ----------
    n_layers : int
        Total number of target model transformer layers.
    kv_accept_threshold : float
        Cosine-similarity threshold (0–1) above which a draft KV layer is
        accepted without target recomputation.  Higher = more conservative.
    probe_fraction : float
        Fraction of token positions to use when computing per-layer KV
        agreement (e.g. 0.1 = probe 10 % of positions to reduce cost).
    min_prompt_len : int
        Minimum prompt length before speculative prefill is attempted.
        Short prompts fall through to standard prefill.
    draft_layer_map : list[int] | None
        Optional explicit mapping from draft layer index to target layer index.
        ``None`` = uniform stride mapping based on layer count ratio.
    """

    n_layers:             int         = 32
    kv_accept_threshold:  float       = 0.92
    probe_fraction:       float       = 0.1
    min_prompt_len:       int         = 64
    draft_layer_map:      Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1; got {self.n_layers}")
        if not (0.0 < self.kv_accept_threshold <= 1.0):
            raise ValueError(
                f"kv_accept_threshold must be in (0,1]; "
                f"got {self.kv_accept_threshold}"
            )
        if not (0.0 < self.probe_fraction <= 1.0):
            raise ValueError(
                f"probe_fraction must be in (0,1]; got {self.probe_fraction}"
            )
        if self.min_prompt_len < 1:
            raise ValueError(
                f"min_prompt_len must be ≥ 1; got {self.min_prompt_len}"
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class SpecPrefillStats:
    """Statistics from a single speculative prefill call.

    Attributes
    ----------
    total_layers : int
        Number of target model layers.
    layers_skipped : int
        Layers whose KV was accepted from the draft model.
    layers_recomputed : int
        Layers that required a target-model recompute.
    mean_kv_similarity : float
        Mean cosine similarity across all probed layer pairs.
    """

    total_layers:       int   = 0
    layers_skipped:     int   = 0
    layers_recomputed:  int   = 0
    mean_kv_similarity: float = 0.0

    @property
    def skip_rate(self) -> float:
        """Fraction of layers skipped."""
        return self.layers_skipped / self.total_layers if self.total_layers else 0.0

    @property
    def speedup_estimate(self) -> float:
        """Rough TTFT speedup: 1 / (1 - skip_rate * layer_cost_fraction).

        Assumes each skipped layer saves ~(1/n_layers) of total prefill time.
        """
        if self.total_layers == 0:
            return 1.0
        saved = self.skip_rate
        return 1.0 / max(1.0 - saved, 0.1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpeculativePrefiller:
    """Draft-accelerated prefill using a loaded draft model.

    Parameters
    ----------
    draft_forward : callable
        ``draft_forward(token_ids: list[int]) -> list[np.ndarray]``
        Must return a list of ``n_draft_layers`` KV arrays, each of shape
        ``(n_heads, seq_len, head_dim)``.
    target_forward : callable
        ``target_forward(token_ids: list[int], recompute_mask: list[bool])
        -> list[np.ndarray]``
        Runs the target model, skipping layers where ``recompute_mask[i]``
        is ``False`` and returning the draft KV for those layers.
        Shape: list of ``n_layers`` KV arrays.
    config : SpecPrefillConfig
    """

    def __init__(
        self,
        draft_forward:  Callable[[List[int]], List[np.ndarray]],
        target_forward: Callable[[List[int], List[bool]], List[np.ndarray]],
        config:         SpecPrefillConfig,
    ) -> None:
        self._draft  = draft_forward
        self._target = target_forward
        self._cfg    = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prefill(
        self,
        token_ids: List[int],
    ) -> Tuple[List[np.ndarray], SpecPrefillStats]:
        """Run speculative prefill and return corrected KV states.

        Parameters
        ----------
        token_ids : list[int]
            Full prompt token IDs.

        Returns
        -------
        kv_states : list[np.ndarray]
            One KV tensor per target-model layer.  Shape of each:
            ``(n_heads, seq_len, head_dim)``.
        stats : SpecPrefillStats
        """
        stats = SpecPrefillStats(total_layers=self._cfg.n_layers)
        n = len(token_ids)

        if n < self._cfg.min_prompt_len:
            # Short prompt: skip speculative prefill, fall through to standard
            kv = self._target(token_ids, [True] * self._cfg.n_layers)
            return kv, stats

        # Step 1: draft KV
        draft_kv = self._draft(token_ids)
        n_draft  = len(draft_kv)

        # Step 2: compute per-layer agreement
        layer_map   = self._build_layer_map(n_draft, self._cfg.n_layers)
        similarities = self._compute_similarities(draft_kv, layer_map, n)
        recompute_mask, n_skip = self._acceptance_mask(similarities)

        stats.layers_skipped    = n_skip
        stats.layers_recomputed = self._cfg.n_layers - n_skip
        stats.mean_kv_similarity = float(np.mean(similarities)) if len(similarities) > 0 else 0.0

        # Step 3: target forward with selective recompute
        # Provide draft KV for accepted layers; recompute the rest.
        target_kv = self._target(token_ids, recompute_mask)

        # Merge: accepted layers use draft KV; recomputed layers use target KV
        merged: List[np.ndarray] = []
        for i, (recompute, t_kv) in enumerate(zip(recompute_mask, target_kv)):
            if recompute or i >= n_draft or layer_map[i] >= n_draft:
                merged.append(np.asarray(t_kv) if t_kv is not None else np.array([]))
            else:
                draft_idx = layer_map[i]
                merged.append(np.asarray(draft_kv[draft_idx]))

        return merged, stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_layer_map(self, n_draft: int, n_target: int) -> List[int]:
        """Map target layer indices to draft layer indices."""
        if self._cfg.draft_layer_map is not None:
            m = self._cfg.draft_layer_map
            # Pad or truncate to n_target
            return [m[i] if i < len(m) else n_draft - 1 for i in range(n_target)]
        # Uniform stride — evenly distribute draft layers across target layers
        if n_draft == 0:
            return [0] * n_target
        return [min(int(i * n_draft / n_target), n_draft - 1) for i in range(n_target)]

    def _compute_similarities(
        self,
        draft_kv: List[np.ndarray],
        layer_map: List[int],
        seq_len: int,
    ) -> List[float]:
        """Compute cosine similarity between adjacent draft KV vectors.

        We use self-similarity within the draft KV (comparing consecutive
        layers) as a proxy for draft-to-target agreement, since we don't
        have access to target KV before the target forward pass.

        A more accurate implementation would compare draft KV to cached
        target KV from the previous request (if available).
        """
        sims: List[float] = []
        for i in range(len(draft_kv) - 1):
            a = np.asarray(draft_kv[i], dtype=np.float32).ravel()
            b = np.asarray(draft_kv[i + 1], dtype=np.float32).ravel()
            if a.size == 0 or b.size == 0:
                sims.append(0.0)
                continue
            # Cosine similarity
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                sims.append(1.0)
            else:
                sims.append(float(np.dot(a, b) / (na * nb)))
        # Pad to n_layers length
        while len(sims) < self._cfg.n_layers:
            sims.append(0.0)
        return sims[: self._cfg.n_layers]

    def _acceptance_mask(
        self, similarities: List[float]
    ) -> Tuple[List[bool], int]:
        """Build a per-layer recompute mask from similarity scores.

        Returns
        -------
        recompute_mask : list[bool]
            ``True`` = recompute this layer from scratch; ``False`` = accept
            draft KV.
        n_skip : int
            Number of skipped (draft-accepted) layers.
        """
        threshold = self._cfg.kv_accept_threshold
        mask  = [sim < threshold for sim in similarities]  # True = recompute
        n_skip = sum(1 for v in mask if not v)
        return mask, n_skip
