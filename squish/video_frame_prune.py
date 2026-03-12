"""VideoFramePrune — Temporal and spatial pruning of video-language model tokens.

Video-language models encode each frame as a grid of patch tokens (e.g. 14×14
= 196 patches at ViT-B/16 resolution).  Two complementary pruning strategies
are provided:

Temporal pruning
    Consecutive frames whose mean-pooled embeddings are highly similar
    (cosine similarity >= ``similarity_threshold``) are considered redundant.
    Only the *first* frame in a run of near-duplicate frames is kept; subsequent
    similar frames are dropped.  This alone yields 60–80% frame reduction on
    typical video streams where most frames change slowly.

Spatial pruning
    Within each kept frame, low-saliency patch tokens are discarded.  Patch
    saliency is defined as the L2 norm of the patch feature vector.  The top
    ``(1 - spatial_prune_ratio)`` fraction of patches are retained, returning
    their *original* index positions in ascending order (preserving position
    embedding semantics).

Typical usage::

    from squish.video_frame_prune import FrameConfig, VideoFramePruner
    import numpy as np

    cfg    = FrameConfig(n_frames=32, tokens_per_frame=196, embed_dim=256)
    pruner = VideoFramePruner(cfg)

    # frame_embeddings: mean-pooled embedding per frame, shape (32, 256)
    frame_embeddings = np.random.randn(32, 256).astype(np.float32)
    kept_frame_idx   = pruner.prune_temporal(frame_embeddings)

    # patch_features: patch token features for one frame, shape (196, 256)
    patch_features = np.random.randn(196, 256).astype(np.float32)
    kept_patch_idx = pruner.prune_spatial(patch_features)

    print(pruner.stats)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FrameConfig",
    "VideoFramePruner",
    "FrameStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FrameConfig:
    """Configuration for :class:`VideoFramePruner`.

    Parameters
    ----------
    n_frames : int
        Expected number of input frames per video clip.
    tokens_per_frame : int
        Number of spatial patch tokens per frame (e.g. 196 for a 14×14 grid).
    similarity_threshold : float
        Cosine-similarity value at or above which a frame is deemed temporally
        redundant with the preceding kept frame and is dropped.
        Must be in ``(0, 1]``.
    spatial_prune_ratio : float
        Fraction of patch tokens to discard per kept frame based on L2-norm
        saliency.  Must be in ``[0, 1)``.  A value of ``0.0`` keeps all
        patches; ``0.5`` keeps the top-50% most salient patches.
    embed_dim : int
        Embedding dimensionality for frame and patch feature vectors.
    """

    n_frames: int = 32
    tokens_per_frame: int = 196
    similarity_threshold: float = 0.92
    spatial_prune_ratio: float = 0.3
    embed_dim: int = 256

    def __post_init__(self) -> None:
        if self.n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        if self.tokens_per_frame < 1:
            raise ValueError("tokens_per_frame must be >= 1")
        if not (0.0 < self.similarity_threshold <= 1.0):
            raise ValueError(
                "similarity_threshold must be in (0, 1]; "
                f"got {self.similarity_threshold}"
            )
        if not (0.0 <= self.spatial_prune_ratio < 1.0):
            raise ValueError(
                "spatial_prune_ratio must be in [0, 1); "
                f"got {self.spatial_prune_ratio}"
            )
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class FrameStats:
    """Aggregate statistics for :class:`VideoFramePruner`.

    Attributes
    ----------
    total_temporal_prune_calls : int
        Number of :meth:`~VideoFramePruner.prune_temporal` calls.
    total_frames_in : int
        Total frames received across all temporal prune calls.
    total_frames_kept : int
        Total frames retained across all temporal prune calls.
    total_spatial_prune_calls : int
        Number of :meth:`~VideoFramePruner.prune_spatial` calls.
    total_patches_kept : int
        Total patch indices returned across all spatial prune calls.
    """

    total_temporal_prune_calls: int = 0
    total_frames_in: int = 0
    total_frames_kept: int = 0
    total_spatial_prune_calls: int = 0
    total_patches_kept: int = 0

    @property
    def temporal_keep_rate(self) -> float:
        """Fraction of input frames retained by temporal pruning."""
        if self.total_frames_in == 0:
            return 0.0
        return self.total_frames_kept / self.total_frames_in


# ---------------------------------------------------------------------------
# VideoFramePruner
# ---------------------------------------------------------------------------


class VideoFramePruner:
    """Prune temporally redundant frames and low-saliency spatial patch tokens.

    Parameters
    ----------
    config : FrameConfig
        Pruner configuration.
    """

    def __init__(self, config: FrameConfig) -> None:
        self._cfg = config
        self._stats = FrameStats()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def prune_temporal(self, frame_embeddings: np.ndarray) -> np.ndarray:
        """Return the indices of non-redundant frames.

        Frame *i* is kept if ``i == 0`` or if its cosine similarity with the
        most recently kept frame is *strictly below* ``similarity_threshold``.

        Parameters
        ----------
        frame_embeddings : np.ndarray
            Mean-pooled frame embeddings, shape ``(n_frames, embed_dim)``.

        Returns
        -------
        np.ndarray
            1-D int64 array of kept frame indices in ascending order.

        Raises
        ------
        ValueError
            If *frame_embeddings* does not have shape ``(n_frames, embed_dim)``.
        """
        frame_embeddings = np.asarray(frame_embeddings, dtype=np.float32)
        if frame_embeddings.ndim != 2:
            raise ValueError(
                f"frame_embeddings must be 2-D (n_frames, embed_dim); "
                f"got ndim={frame_embeddings.ndim}"
            )
        n_frames, embed_dim = frame_embeddings.shape
        if embed_dim != self._cfg.embed_dim:
            raise ValueError(
                f"frame_embeddings embed_dim must be {self._cfg.embed_dim}; "
                f"got {embed_dim}"
            )

        if n_frames == 0:
            self._stats.total_temporal_prune_calls += 1
            return np.empty(0, dtype=np.int64)

        # L2-normalise all frames for fast cosine similarity computation.
        norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
        normed = frame_embeddings / (norms + 1e-10)

        kept: list[int] = [0]
        last_kept_normed = normed[0]

        for i in range(1, n_frames):
            cos_sim = float(np.dot(last_kept_normed, normed[i]))
            if cos_sim < self._cfg.similarity_threshold:
                kept.append(i)
                last_kept_normed = normed[i]

        result = np.array(kept, dtype=np.int64)
        self._stats.total_temporal_prune_calls += 1
        self._stats.total_frames_in += n_frames
        self._stats.total_frames_kept += len(result)
        return result

    def prune_spatial(self, patch_features: np.ndarray) -> np.ndarray:
        """Return the indices of the most salient patch tokens in a single frame.

        Saliency is measured as the L2 norm of each patch's feature vector.
        The top ``ceil(n_patches * (1 - spatial_prune_ratio))`` patches are
        returned, sorted in ascending index order to preserve position-embedding
        semantics downstream.

        Parameters
        ----------
        patch_features : np.ndarray
            Patch feature matrix, shape ``(n_patches, embed_dim)``.

        Returns
        -------
        np.ndarray
            1-D int64 array of kept patch indices in ascending order.

        Raises
        ------
        ValueError
            If *patch_features* does not have 2 dimensions or the wrong
            ``embed_dim``.
        """
        patch_features = np.asarray(patch_features, dtype=np.float32)
        if patch_features.ndim != 2:
            raise ValueError(
                f"patch_features must be 2-D (n_patches, embed_dim); "
                f"got ndim={patch_features.ndim}"
            )
        n_patches, embed_dim = patch_features.shape
        if embed_dim != self._cfg.embed_dim:
            raise ValueError(
                f"patch_features embed_dim must be {self._cfg.embed_dim}; "
                f"got {embed_dim}"
            )

        n_keep = max(
            1,
            int(np.ceil(n_patches * (1.0 - self._cfg.spatial_prune_ratio))),
        )
        n_keep = min(n_keep, n_patches)

        if n_keep == n_patches:
            result = np.arange(n_patches, dtype=np.int64)
        else:
            norms = np.linalg.norm(patch_features, axis=1)  # (n_patches,)
            # Partial sort: argpartition for efficiency on large patch grids.
            top_idx = np.argpartition(-norms, n_keep - 1)[:n_keep]
            result = np.sort(top_idx).astype(np.int64)

        self._stats.total_spatial_prune_calls += 1
        self._stats.total_patches_kept += len(result)
        return result

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> FrameStats:
        """Current aggregate statistics."""
        return self._stats

    def __repr__(self) -> str:
        return (
            f"VideoFramePruner(n_frames={self._cfg.n_frames}, "
            f"tokens_per_frame={self._cfg.tokens_per_frame}, "
            f"threshold={self._cfg.similarity_threshold}, "
            f"spatial_prune_ratio={self._cfg.spatial_prune_ratio})"
        )
