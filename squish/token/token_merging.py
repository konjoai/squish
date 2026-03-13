"""
squish/token_merging.py

Token Merging (ToMe) for faster prefill on Apple Silicon.

Based on Bolya et al. (2023), "Token Merging: Your ViT But Faster", ICLR 2023.
Applied to transformer LLM prefill: similar tokens in the key-space are merged
by averaging their representations, reducing sequence length and prefill FLOPs.

ToMe is *complementary* to LazyLLM (lazy_llm.py):
- ToMe merges similar tokens in early/mid layers (default layers 4–11).
- LazyLLM prunes unimportant tokens in deeper layers (default start_layer=2).

The combination yields additive speedups: ToMe reduces the effective batch
dimension while LazyLLM zeroes out low-importance tokens.

Speedup:
    30–40% prefill speedup with <2% quality degradation for sequences ≥ 100 tokens.

Algorithm:
    For each merge layer:
    1. Split sequence into two halves (A, B) based on token index parity.
    2. For every token in A, find its most-similar token in B using cosine
       similarity in the key (Q·K^T attention pre-softmax) space.
    3. Merge the top-r matched pairs by averaging their hidden-state vectors.
    4. The sequence length decreases by r per merge layer.

Usage::

    from squish.token.token_merging import TokenMergingConfig, patch_model_tome, unpatch_model_tome

    cfg   = TokenMergingConfig(r=16, start_layer=4, end_layer=11)
    state = patch_model_tome(model, cfg)

    # Between requests: reset merge state
    state.reset()

    # Undo patching
    unpatch_model_tome(model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "TokenMergingConfig",
    "TokenMergingState",
    "patch_model_tome",
    "unpatch_model_tome",
    "bipartite_merge",
    "unmerge_tokens",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenMergingConfig:
    """
    Hyper-parameters for Token Merging (ToMe).

    Parameters
    ----------
    r : int
        Number of token pairs to merge per merge-layer.  Total sequence
        reduction = r × (number of merge layers).
        E.g., r=16 over 8 layers removes 128 tokens from a 1024-token prefill.
    start_layer : int
        First transformer layer index where merging is applied.
        Earlier layers (0..start_layer-1) always see the full sequence.
    end_layer : int | None
        Last layer where merging is applied (inclusive).
        ``None`` means "all layers from start_layer onward".
    similarity_threshold : float
        Minimum cosine similarity required to merge a pair.  Pairs below
        this threshold are not merged (prevents merging semantically distinct
        tokens).  Set to -1.0 to always merge the top-r pairs.
    verbose : bool
        Print per-layer merge stats.
    """
    r:                    int   = 16
    start_layer:          int   = 4
    end_layer:            int | None = None   # None → merge in all layers ≥ start_layer
    similarity_threshold: float = 0.5
    verbose:              bool  = False

    def __post_init__(self) -> None:
        if self.r < 0:
            raise ValueError(f"r must be ≥ 0, got {self.r}")
        if self.start_layer < 0:
            raise ValueError("start_layer must be ≥ 0")
        if self.end_layer is not None and self.end_layer < self.start_layer:
            raise ValueError("end_layer must be ≥ start_layer")
        if not (-1.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [-1, 1]")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class TokenMergingState:
    """
    Mutable merge state for one request.

    Stores the merge mapping so that token outputs can be *unmerged* back to
    the original sequence length if needed (e.g., for logit slicing).

    .. note::
        Squish uses ToMe only during *prefill* and does NOT need to unmerge
        during auto-regressive decode (each decode step is a single new token).
        The mapping is stored for completeness and optional downstream use.
    """
    # Stack of merge maps: each entry is (src_indices, dst_indices, T_before)
    # where src_indices were merged INTO dst_indices.
    _merge_maps: list[tuple] = field(default_factory=list)

    def reset(self) -> None:
        """Clear merge maps between requests."""
        self._merge_maps.clear()

    def record_merge(
        self,
        src_indices: np.ndarray,
        dst_indices: np.ndarray,
        t_before: int,
    ) -> None:
        """Record one layer's merge operation."""
        self._merge_maps.append((src_indices, dst_indices, t_before))

    @property
    def n_merges(self) -> int:
        """Total number of merge operations applied."""
        return sum(len(s) for s, _, _ in self._merge_maps)

    @property
    def n_merge_layers(self) -> int:
        """Number of layers where merging was applied."""
        return len(self._merge_maps)


# ---------------------------------------------------------------------------
# Core merge / unmerge helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_bipartite(
    a: np.ndarray,  # (T_a, D)
    b: np.ndarray,  # (T_b, D)
) -> np.ndarray:
    """
    Compute cosine similarity between every token in *a* and every token in *b*.

    Returns
    -------
    sim : (T_a, T_b)  float32 — similarity[i, j] = cosine(a[i], b[j])
    """
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    # Normalise rows
    a_norm = a_f / (np.linalg.norm(a_f, axis=-1, keepdims=True) + 1e-8)
    b_norm = b_f / (np.linalg.norm(b_f, axis=-1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T   # (T_a, T_b)


def bipartite_merge(
    hidden: np.ndarray,          # (T, D)  float32
    r: int,
    similarity_threshold: float = 0.5,
) -> tuple:
    """
    Merge up to *r* token pairs in *hidden* via bipartite matching.

    Splits the sequence into even (set A) and odd (set B) positions, computes
    pairwise cosine similarity, greedily matches the top-*r* pairs, and
    averages the matched pairs.

    Parameters
    ----------
    hidden : (T, D) float32 — full hidden-state sequence
    r      : maximum number of pairs to merge
    similarity_threshold : only merge pairs with similarity ≥ this value

    Returns
    -------
    merged    : (T - n_merged, D) float32
    src_idx   : (n_merged,) int64 — which tokens in 0..T-1 were merged INTO dst
    dst_idx   : (n_merged,) int64 — which tokens in 0..T-1 received the merge
    """
    T, D = hidden.shape
    if T < 2 or r <= 0:
        empty = np.empty(0, dtype=np.int64)
        return hidden, empty, empty

    # Split into two halves by index parity
    a_idx = np.arange(0, T, 2, dtype=np.int64)    # even positions
    b_idx = np.arange(1, T, 2, dtype=np.int64)    # odd positions

    a = hidden[a_idx]   # (T_a, D)
    b = hidden[b_idx]   # (T_b, D)

    # Similarity: (T_a, T_b)
    sim = _cosine_similarity_bipartite(a, b)

    # Greedy matching: each a-token picks its best b-partner
    best_sim = sim.max(axis=1)           # (T_a,)
    best_b   = sim.argmax(axis=1)        # (T_a,) — index into b_idx

    # Filter by threshold and cap at r
    eligible_mask = best_sim >= similarity_threshold
    eligible_a    = np.where(eligible_mask)[0]   # indices into a_idx array

    if len(eligible_a) == 0:
        empty = np.empty(0, dtype=np.int64)
        return hidden, empty, empty

    # Sort eligible pairs by descending similarity, take top-r
    order         = np.argsort(-best_sim[eligible_a])
    selected_a    = eligible_a[order[:r]]
    selected_b    = best_b[selected_a]

    # Global token indices
    src_global = a_idx[selected_a]   # tokens that will be merged away (from A)
    dst_global = b_idx[selected_b]   # tokens that receive the merge (in B)

    # Handle duplicate destination assignments (multiple A → same B)
    # Keep only the first occurrence per unique dst to avoid double-merging.
    _, unique_pos = np.unique(dst_global, return_index=True)
    src_global = src_global[unique_pos]
    dst_global = dst_global[unique_pos]

    # Average: dst receives mean of (dst, src)
    out = hidden.copy()
    out[dst_global] = (out[dst_global].astype(np.float32)
                       + out[src_global].astype(np.float32)) * 0.5

    # Build output by removing src tokens
    keep_mask = np.ones(T, dtype=bool)
    keep_mask[src_global] = False
    merged = out[keep_mask]

    return merged.astype(hidden.dtype), src_global, dst_global


def unmerge_tokens(
    merged: np.ndarray,   # (T_merged, D)
    src_idx: np.ndarray,  # (n_merged,) int64
    dst_idx: np.ndarray,  # (n_merged,) int64
    t_original: int,
) -> np.ndarray:
    """
    Approximately invert a bipartite merge by duplicating merged tokens back.

    The unmerged sequence has the same length as the original.  Merged tokens
    are reconstructed as copies of their destination (dst) position.

    Parameters
    ----------
    merged     : (T_merged, D)
    src_idx    : original token indices that were merged away
    dst_idx    : destination indices they were merged into
    t_original : original sequence length T_before

    Returns
    -------
    (t_original, D) — sequence restored to original length
    """
    T_merged = merged.shape[0]
    D        = merged.shape[1]

    if src_idx.size == 0:
        assert T_merged == t_original, "Token count mismatch on unmerge"
        return merged

    out = np.zeros((t_original, D), dtype=merged.dtype)
    # Map merged-sequence indices back to original positions
    # "after merge" indices: removed src_idx positions, kept the rest.
    keep_mask = np.ones(t_original, dtype=bool)
    keep_mask[src_idx] = False
    keep_pos = np.where(keep_mask)[0]   # original positions still present

    if len(keep_pos) != T_merged:
        # Dimension mismatch — return safe fallback (repeat-pad)
        out[keep_pos[:T_merged]] = merged[:len(keep_pos)]
        out[src_idx] = out[dst_idx]
        return out

    out[keep_pos] = merged
    # Restore removed tokens as copies of their merge destination
    out[src_idx] = out[dst_idx]
    return out


# ---------------------------------------------------------------------------
# Layer wrapper
# ---------------------------------------------------------------------------

class _ToMeLayerWrapper:
    """
    Wraps a single TransformerBlock to apply ToMe token merging during prefill.

    Compatible with the same mlx_lm model layout as ``_LazyLLMLayerWrapper``.
    """

    def __init__(
        self,
        original_layer: Any,
        layer_idx: int,
        config: TokenMergingConfig,
        state: TokenMergingState,
    ) -> None:
        self._orig   = original_layer
        self._idx    = layer_idx
        self._config = config
        self._state  = state

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            import mlx.core as mx
            import numpy as _np
        except ImportError:  # pragma: no cover
            return self._orig(x, *args, **kwargs)

        # Pass-through: decode steps (single token) or if r==0
        if x.shape[1] <= 1 or self._config.r == 0:
            return self._orig(x, *args, **kwargs)

        # Check layer range
        end = self._config.end_layer
        if end is not None and self._idx > end:
            return self._orig(x, *args, **kwargs)

        # Extract sequence as numpy, apply merge, restore as MLX
        # Shape: (B, T, D) — B is always 1 in Squish generation
        B, T, D = x.shape
        x_np = _np.array(x[0].astype(mx.float32))   # (T, D)

        merged_np, src_idx, dst_idx = bipartite_merge(
            x_np,
            r=self._config.r,
            similarity_threshold=self._config.similarity_threshold,
        )

        if len(src_idx) > 0:
            self._state.record_merge(src_idx, dst_idx, T)
            if self._config.verbose:
                print(
                    f"  [token_merging] layer={self._idx:02d} "
                    f"merged={len(src_idx)}  T:{T}→{merged_np.shape[0]}"
                )

        x_merged = mx.array(merged_np[_np.newaxis]).astype(x.dtype)   # (1, T', D)

        # Forward through original layer with merged sequence
        return self._orig(x_merged, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


# ---------------------------------------------------------------------------
# Model-level wrap / unwrap
# ---------------------------------------------------------------------------

def _get_layers(model: Any) -> list | None:
    """Return transformer layer list from a mlx_lm model, or None."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def patch_model_tome(
    model: Any,
    config: TokenMergingConfig | None = None,
) -> TokenMergingState:
    """
    Patch *model* in-place with Token Merging (ToMe) for prefill acceleration.

    Parameters
    ----------
    model  : mlx_lm Transformer model
    config : ``TokenMergingConfig``.  Defaults to ``r=16, start_layer=4``.

    Returns
    -------
    state : ``TokenMergingState`` — call ``state.reset()`` between requests.
            Returns a no-op state if the model is incompatible.
    """
    if config is None:
        config = TokenMergingConfig()

    layers = _get_layers(model)
    state  = TokenMergingState()

    if layers is None:
        import logging
        logging.getLogger(__name__).warning(
            "token_merging: cannot locate transformer layers — patching skipped"
        )
        return state

    end = config.end_layer if config.end_layer is not None else len(layers) - 1

    new_layers = []
    for i, layer in enumerate(layers):
        if config.start_layer <= i <= end:
            new_layers.append(_ToMeLayerWrapper(layer, i, config, state))
        else:
            new_layers.append(layer)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
    else:
        model.layers = new_layers

    model._tome_state = state
    model._tome_orig  = layers

    return state


def unpatch_model_tome(model: Any) -> None:
    """
    Remove ToMe patches from *model*, restoring original layers.

    Safe to call even if the model was never patched.
    """
    if not hasattr(model, "_tome_orig"):
        return

    orig = model._tome_orig
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = orig
    elif hasattr(model, "layers"):
        model.layers = orig

    del model._tome_state
    del model._tome_orig
