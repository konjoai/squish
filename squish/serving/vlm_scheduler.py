"""VLMBatchScheduler: resolution-aware multi-modal request batching.

Grouping multi-modal requests by image resolution prevents the VRAM spike that
occurs when a low-cost text-only request is batched with a 4K-image request
that requires 4× the visual encoder capacity.

Requests are classified into four buckets: ``low`` (≤ low_res_threshold²
pixels), ``mid``, ``high``, and ``video``.  Within each bucket, requests are
ordered to maximise visual encoder overlap with the LLM prefill phase (longest
visual-token sequences first, so slow encoding paths shadow text tokenization).

Design rational:
    - LLaVA-Next authors document bucket-based resolution handling.
    - InternVL2 training splits resolutions into tiles; similar scheduling
      avoids CUDA OOM when bins are mixed.
    - The video bucket is separated because temporal attention requires a full
      sequence-of-frames KV — mixing with single-image requests prevents
      effective KV-cache reuse.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List

__all__ = [
    "VLMSchedulerConfig",
    "VLMRequest",
    "VLMBatch",
    "VLMBatchScheduler",
]

_TILE_SIZE: int = 336
_TOKENS_PER_TILE: int = 49  # 7×7 grid at 48-pixel patch stride


@dataclass
class VLMSchedulerConfig:
    """Configuration for :class:`VLMBatchScheduler`.

    Attributes:
        low_res_threshold: Max pixel dimension (width or height) for the
            ``low`` resolution bucket.
        high_res_threshold: Min pixel dimension to classify as ``high``.
        max_batch_size: Maximum number of requests per batch.
        video_fps_threshold: Requests with ``fps`` above this are forced into
            the ``video`` bucket regardless of resolution.
        seed: Unused; for API consistency.
    """

    low_res_threshold: int = 336
    high_res_threshold: int = 672
    max_batch_size: int = 8
    video_fps_threshold: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.low_res_threshold < 1:
            raise ValueError(
                f"low_res_threshold must be ≥ 1, got {self.low_res_threshold}"
            )
        if self.high_res_threshold <= self.low_res_threshold:
            raise ValueError(
                "high_res_threshold must be > low_res_threshold"
            )
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be ≥ 1, got {self.max_batch_size}"
            )


@dataclass
class VLMRequest:
    """A single multi-modal inference request.

    Attributes:
        prompt: The text portion of the request.
        image_height: Input image height in pixels.
        image_width: Input image width in pixels.
        is_video: True if request carries a video sequence.
        fps: Frame rate (used to distinguish video from single-frame).
        request_id: Unique identifier (auto-generated UUID4 if not provided).
    """

    prompt: str
    image_height: int
    image_width: int
    is_video: bool = False
    fps: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def max_dim(self) -> int:
        return max(self.image_height, self.image_width)


@dataclass
class VLMBatch:
    """A batch of requests sharing the same resolution bucket.

    Attributes:
        requests: Ordered list of requests.
        bucket: Resolution bucket name.
        estimated_visual_tokens: Sum of estimated visual token counts.
    """

    requests: List[VLMRequest]
    bucket: str
    estimated_visual_tokens: int

    @property
    def n_requests(self) -> int:
        return len(self.requests)

    @property
    def total_visual_tokens(self) -> int:
        return self.estimated_visual_tokens


class VLMBatchScheduler:
    """Classify and batch VLM requests for efficient scheduling.

    Example::

        cfg = VLMSchedulerConfig(max_batch_size=4)
        sched = VLMBatchScheduler(cfg)
        batches = sched.batch(requests)
        ordered = sched.schedule(requests)
    """

    def __init__(self, config: VLMSchedulerConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def classify(self, request: VLMRequest) -> str:
        """Return the resolution bucket for a single request.

        Buckets: ``"video"``, ``"low"``, ``"mid"``, or ``"high"``.
        """
        if request.is_video or request.fps >= self.config.video_fps_threshold:
            return "video"
        dim = request.max_dim
        if dim <= self.config.low_res_threshold:
            return "low"
        if dim >= self.config.high_res_threshold:
            return "high"
        return "mid"

    def batch(self, requests: List[VLMRequest]) -> List[VLMBatch]:
        """Group requests by bucket and split into max-batch-size chunks.

        Within each bucket, requests are sorted by descending visual token
        estimate (longest visual prefill first).
        """
        buckets: dict[str, List[VLMRequest]] = {
            "low": [],
            "mid": [],
            "high": [],
            "video": [],
        }
        for req in requests:
            buckets[self.classify(req)].append(req)

        batches: List[VLMBatch] = []
        bs = self.config.max_batch_size
        for bucket_name, reqs in buckets.items():
            if not reqs:
                continue
            reqs_sorted = sorted(
                reqs,
                key=lambda r: self.estimated_visual_tokens(
                    r.image_height, r.image_width
                ),
                reverse=True,
            )
            for start in range(0, len(reqs_sorted), bs):
                chunk = reqs_sorted[start : start + bs]
                est = sum(
                    self.estimated_visual_tokens(r.image_height, r.image_width)
                    for r in chunk
                )
                batches.append(
                    VLMBatch(
                        requests=chunk,
                        bucket=bucket_name,
                        estimated_visual_tokens=est,
                    )
                )
        return batches

    def schedule(self, requests: List[VLMRequest]) -> List[VLMRequest]:
        """Return a globally ordered list of requests.

        Order: high → mid → low → video (descending visual cost first so that
        the encoder latency is absorbed by concurrent prefill work for cheaper
        requests).
        """
        bucket_order = ["high", "mid", "low", "video"]
        grouped: dict[str, List[VLMRequest]] = {b: [] for b in bucket_order}
        for req in requests:
            grouped[self.classify(req)].append(req)

        ordered: List[VLMRequest] = []
        for bucket_name in bucket_order:
            reqs = grouped[bucket_name]
            reqs.sort(
                key=lambda r: self.estimated_visual_tokens(
                    r.image_height, r.image_width
                ),
                reverse=True,
            )
            ordered.extend(reqs)
        return ordered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimated_visual_tokens(height: int, width: int) -> int:
        """Estimate the number of visual tokens for a given image size.

        Based on the LLaVA-Next / InternVL2 tiling formula: each
        ``_TILE_SIZE × _TILE_SIZE`` region produces ``_TOKENS_PER_TILE``
        visual tokens.
        """
        n_tiles_h = max(1, round(height / _TILE_SIZE))
        n_tiles_w = max(1, round(width / _TILE_SIZE))
        return n_tiles_h * n_tiles_w * _TOKENS_PER_TILE
