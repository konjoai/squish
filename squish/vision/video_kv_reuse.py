"""VideoKVReuse: temporal KV block reuse for efficient multi-frame inference.

Consecutive video frames share overlapping visual patches.  Regions whose
patch embeddings change less than a cosine-similarity threshold between
frame *t-1* and frame *t* can reuse the corresponding KV vectors without
recomputation.  This reduces the visual-token FLOP budget for video sequences
by the empirically measured static-region ratio (often 50–80% for slow-panning
footage at 4–8 fps).

Design follows the analysis in VideoLLM-online (arXiv 2406.11816) and extends
the static-region caching of DeltaLLM (arXiv 2406.12434).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "VideoKVReuseConfig",
    "VideoKVReuseState",
    "VideoKVReuse",
]


@dataclass
class VideoKVReuseConfig:
    """Configuration for :class:`VideoKVReuse`.

    Attributes:
        change_threshold: Cosine-distance threshold. A patch is considered
            *changed* when ``1 - cos_sim`` exceeds this value.  Values below
            this threshold reuse the previous KV.
        token_dim: Head dimension of the KV vectors.
        max_tokens: Maximum number of patches per frame.
        seed: Unused; for API consistency.
    """

    change_threshold: float = 0.15
    token_dim: int = 128
    max_tokens: int = 1024
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.change_threshold < 1.0):
            raise ValueError(
                f"change_threshold must be in (0, 1), got {self.change_threshold}"
            )
        if self.token_dim < 1:
            raise ValueError(f"token_dim must be ≥ 1, got {self.token_dim}")


@dataclass
class VideoKVReuseState:
    """Per-sequence temporal reuse state.

    Attributes:
        prev_patches: Patch embeddings from the previous frame ``(n, d)`` or
            ``None`` for the very first frame.
        prev_k: Previous frame key cache ``(n, token_dim)`` or ``None``.
        prev_v: Previous frame value cache ``(n, token_dim)`` or ``None``.
        n_reused: Cumulative count of reused patch KV pairs.
        n_recomputed: Cumulative count of recomputed patch KV pairs.
        n_frames: Number of frames processed so far.
    """

    prev_patches: Optional[np.ndarray] = None
    prev_k: Optional[np.ndarray] = None
    prev_v: Optional[np.ndarray] = None
    n_reused: int = 0
    n_recomputed: int = 0
    n_frames: int = 0

    @property
    def reuse_ratio(self) -> float:
        total = self.n_reused + self.n_recomputed
        return self.n_reused / total if total > 0 else 0.0

    @property
    def total_patches_processed(self) -> int:
        return self.n_reused + self.n_recomputed


class VideoKVReuse:
    """Reuse unchanged-region KV vectors across video frames.

    The caller supplies a ``kv_fn`` that, given patch embeddings ``(n, d)``,
    returns ``(k, v)`` each of shape ``(n, token_dim)``.  The implementation
    calls ``kv_fn`` only for changed patches; unchanged patches receive the
    cached KV from the previous frame.

    Example::

        cfg = VideoKVReuseConfig(change_threshold=0.15, token_dim=64)
        reuser = VideoKVReuse(cfg)
        state = reuser.new_state()
        for frame_patches in video_frames:
            k, v = reuser.process_frame(frame_patches, my_kv_fn, state)
    """

    def __init__(self, config: VideoKVReuseConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> VideoKVReuseState:
        return VideoKVReuseState()

    def process_frame(
        self,
        patches: np.ndarray,
        kv_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        state: VideoKVReuseState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute KV for a new frame, reusing unchanged regions.

        Args:
            patches: Patch embeddings ``(n, d)`` for the current frame.
            kv_fn: Callable mapping ``patches(m, d) → (k(m, token_dim),
                v(m, token_dim))`` for a subset of patches.
            state: Mutable per-sequence state.

        Returns:
            ``(k, v)`` each of shape ``(n, token_dim)``.
        """
        patches = np.asarray(patches, dtype=np.float32)
        n = patches.shape[0]

        # First frame: compute everything.
        if state.prev_patches is None or state.prev_patches.shape[0] != n:
            k, v = kv_fn(patches)
            k = np.asarray(k, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32)
            state.n_recomputed += n
            state.prev_patches = patches.copy()
            state.prev_k = k.copy()
            state.prev_v = v.copy()
            state.n_frames += 1
            return k, v

        change_mask = self._change_mask(state.prev_patches, patches)
        n_changed = int(change_mask.sum())
        n_unchanged = n - n_changed

        k_out = np.zeros((n, self.config.token_dim), dtype=np.float32)
        v_out = np.zeros((n, self.config.token_dim), dtype=np.float32)

        # Reuse unchanged
        if n_unchanged > 0:
            k_out[~change_mask] = state.prev_k[~change_mask]
            v_out[~change_mask] = state.prev_v[~change_mask]

        # Recompute changed
        if n_changed > 0:
            k_new, v_new = kv_fn(patches[change_mask])
            k_out[change_mask] = np.asarray(k_new, dtype=np.float32)
            v_out[change_mask] = np.asarray(v_new, dtype=np.float32)

        state.n_reused += n_unchanged
        state.n_recomputed += n_changed
        state.prev_patches = patches.copy()
        state.prev_k = k_out.copy()
        state.prev_v = v_out.copy()
        state.n_frames += 1
        return k_out, v_out

    def reuse_ratio(self, state: VideoKVReuseState) -> float:
        return state.reuse_ratio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_sim_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-row cosine similarity between corresponding rows of a and b."""
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        na = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
        nb = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
        return (a / na * (b / nb)).sum(axis=1)  # (n,)

    def _change_mask(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> np.ndarray:
        """Boolean mask; True where cosine distance exceeds threshold."""
        cos_sim = self._cosine_sim_matrix(prev, curr)
        cos_dist = 1.0 - cos_sim
        return cos_dist > self.config.change_threshold
