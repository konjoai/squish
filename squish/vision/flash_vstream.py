"""FlashVStream: 3-tier memory architecture for streaming video comprehension.

Zhang et al. (arXiv 2406.08085, ACL 2024) decompose the KV cache for video
into three memory tiers: (1) *spatial* memory — full KV for the current frame,
maximum fidelity; (2) *temporal* memory — selected past-frame KV condensed by a
saliency-based eviction policy; (3) *sensory* memory — a configurable-length
sliding window of the most recent frames retained verbatim.

The three-tier design lets Squish process 60-minute+ video streams in 16 GB by
evicting low-saliency frames while preserving the temporal context needed for
episodic QA.

Reference: Zhang et al., "Flash-VStream: Memory-Based Real-Time Understanding
for Long Video Streams", arXiv 2406.08085, ACL 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "FlashVStreamConfig",
    "FrameEntry",
    "FlashVStreamState",
    "FlashVStream",
]


@dataclass
class FlashVStreamConfig:
    """Configuration for :class:`FlashVStream`.

    Attributes:
        sensory_window: Number of recent frames to keep verbatim in sensory memory.
        temporal_capacity: Maximum number of frames in temporal memory.
        saliency_low_threshold: Frames with saliency below this are eligible for
            eviction once temporal memory exceeds capacity.
        token_dim: Embedding dimension of per-frame KV vectors.
        seed: RNG seed.
    """

    sensory_window: int = 8
    temporal_capacity: int = 32
    saliency_low_threshold: float = 0.2
    token_dim: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sensory_window < 1:
            raise ValueError(f"sensory_window must be ≥ 1, got {self.sensory_window}")
        if self.temporal_capacity < 1:
            raise ValueError(f"temporal_capacity must be ≥ 1, got {self.temporal_capacity}")
        if not (0.0 <= self.saliency_low_threshold <= 1.0):
            raise ValueError(
                f"saliency_low_threshold must be in [0, 1], got {self.saliency_low_threshold}"
            )


@dataclass
class FrameEntry:
    """One video frame stored in memory.

    Attributes:
        frame_idx: Index of this frame in the original video stream.
        kv: KV representation of shape ``(n_tokens, token_dim)``.
        saliency: Scalar saliency score used for eviction policy.
    """

    frame_idx: int
    kv: np.ndarray
    saliency: float = 0.5

    @property
    def n_tokens(self) -> int:
        return self.kv.shape[0]


@dataclass
class FlashVStreamState:
    """Per-stream mutable state.

    Attributes:
        spatial: Current-frame KV (or None if no frame ingested yet).
        temporal: Past-frame entries surviving saliency eviction.
        sensory: Fixed-size sliding window of the most recent frames.
        n_frames_seen: Total frames ingested.
        n_frames_evicted: Frames removed from temporal memory.
    """

    spatial: Optional[FrameEntry] = None
    temporal: List[FrameEntry] = field(default_factory=list)
    sensory: List[FrameEntry] = field(default_factory=list)
    n_frames_seen: int = 0
    n_frames_evicted: int = 0

    @property
    def total_tokens(self) -> int:
        sp = self.spatial.n_tokens if self.spatial else 0
        tp = sum(e.n_tokens for e in self.temporal)
        se = sum(e.n_tokens for e in self.sensory)
        return sp + tp + se


class FlashVStream:
    """Maintain a 3-tier video KV memory with saliency-based temporal eviction.

    Usage::

        cfg = FlashVStreamConfig(sensory_window=8, temporal_capacity=32)
        vstream = FlashVStream(cfg)
        state = vstream.new_state()
        for frame_kv, saliency in video_frames:
            vstream.ingest(frame_kv, saliency, state)
        k, v = vstream.get_kv(state)
    """

    def __init__(self, config: FlashVStreamConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> FlashVStreamState:
        """Return a fresh empty stream state."""
        return FlashVStreamState()

    def ingest(
        self,
        frame_kv: np.ndarray,
        saliency: float,
        state: FlashVStreamState,
    ) -> None:
        """Process one new video frame.

        Parameters
        ----------
        frame_kv:
            KV array of shape ``(n_tokens, token_dim)``.
        saliency:
            Scalar saliency score in [0, 1] for this frame.
        state:
            Per-stream mutable state.
        """
        frame_kv = np.asarray(frame_kv, dtype=np.float32)
        entry = FrameEntry(
            frame_idx=state.n_frames_seen,
            kv=frame_kv,
            saliency=float(saliency),
        )
        state.n_frames_seen += 1

        # Promote outgoing spatial → temporal (if one exists)
        if state.spatial is not None:
            state.temporal.append(state.spatial)
            self._maybe_evict(state)

        # Update sensory sliding window
        state.sensory.append(entry)
        if len(state.sensory) > self.config.sensory_window:
            # Oldest sensory frame is promoted to temporal if salient
            oldest = state.sensory.pop(0)
            state.temporal.append(oldest)
            self._maybe_evict(state)

        # Current frame is new spatial
        state.spatial = entry

    def get_kv(
        self, state: FlashVStreamState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return concatenated K and V arrays from all three tiers.

        Returns ``(k, v)`` each of shape ``(total_tokens, token_dim // 2)``.
        The convention here is that the last half of ``token_dim`` is V, the
        first half is K (matching the design in many flash-attention KV layouts).
        """
        all_entries: List[FrameEntry] = []
        all_entries.extend(state.temporal)
        all_entries.extend(state.sensory)
        if state.spatial is not None:
            all_entries.append(state.spatial)

        if not all_entries:
            empty = np.empty((0, self.config.token_dim // 2), dtype=np.float32)
            return empty, empty

        kv_cat = np.concatenate([e.kv for e in all_entries], axis=0)
        half = kv_cat.shape[1] // 2
        return kv_cat[:, :half], kv_cat[:, half:]

    def memory_stats(self, state: FlashVStreamState) -> dict:
        """Return summary statistics about memory usage."""
        return {
            "n_frames_seen": state.n_frames_seen,
            "n_frames_evicted": state.n_frames_evicted,
            "temporal_frames": len(state.temporal),
            "sensory_frames": len(state.sensory),
            "has_spatial": state.spatial is not None,
            "total_tokens": state.total_tokens,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_evict(self, state: FlashVStreamState) -> None:
        """Evict the lowest-saliency temporal entry when over capacity."""
        while len(state.temporal) > self.config.temporal_capacity:
            # Find lowest-saliency entry
            idx = int(np.argmin([e.saliency for e in state.temporal]))
            state.temporal.pop(idx)
            state.n_frames_evicted += 1
