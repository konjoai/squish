# [Experimental] This module is part of Squish v39+ (Wave 65).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""Stage-aware prefill/decode dispatch for TCA-TBE inference (Wave 65).

During transformer inference there are two fundamentally different
computation regimes:

Prefill (seq_len > 1)
    The model processes a prompt of multiple tokens simultaneously.
    Memory bandwidth predominates; a decoupled *decompress-then-GEMM*
    pipeline (zip_gemm) is preferred so the decompressor and the matrix
    multiply can overlap on the GPU.  Chunked prefill is supported to
    bound peak activation memory and improve GPU occupancy for long prompts.

Decode (seq_len == 1)
    The model generates one new token per step.  Latency is paramount;
    the fused *ZipGEMV* kernel (zip_gemv) is used to hide decompress
    latency inside the dot-product accumulation.

This module provides a thin dispatcher that:

1. Inspects the ``input_ids`` tensor shape to detect the inference stage.
2. Returns a :class:`DispatchDecision` that specifies the selected Metal
   kernel pipeline and chunking strategy.
3. Exposes a :meth:`StageDispatcher.dispatch_chunked` iterator for
   multi-chunk prefill, yielding one :class:`DispatchDecision` per chunk.

Usage::

    from squish.runtime.stage_dispatcher import StageDispatcher, InferenceStage

    dispatcher = StageDispatcher(tca_tbe_enabled=True, chunk_size=512)
    decision   = dispatcher.dispatch(input_ids)      # shape (batch, seq_len)

    if decision.stage == InferenceStage.DECODE:
        # use zip_gemv pipeline
        ...
    else:
        # iterate chunks for prefill
        for chunk_decision in dispatcher.dispatch_chunked(input_ids):
            ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

import numpy as np

__all__ = [
    "InferenceStage",
    "KernelPipeline",
    "DispatchDecision",
    "StageDispatcher",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class InferenceStage(str, Enum):
    """The two inference regimes distinguished by sequence length.

    Attributes:
        PREFILL: Multi-token prompt processing (``seq_len > 1``).
        DECODE:  Single-token generation step (``seq_len == 1``).
    """

    PREFILL = "prefill"
    DECODE  = "decode"


class KernelPipeline(str, Enum):
    """Metal kernel pipeline identifiers.

    Attributes:
        ZIP_GEMV: Fused ZipGEMV for the decode path.
        ZIP_GEMM: Decoupled ZipGEMM for the prefill path.
        NUMPY:    Pure NumPy fallback (no Metal, used in CI and testing).
    """

    ZIP_GEMV = "zip_gemv"
    ZIP_GEMM = "zip_gemm"
    NUMPY    = "numpy"


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispatchDecision:
    """Immutable record describing which kernel pipeline to use.

    Attributes:
        stage: :class:`InferenceStage` — PREFILL or DECODE.
        kernel_pipeline: :class:`KernelPipeline` value (or its string
            equivalent) identifying the kernel to invoke.
        seq_len: Full sequence length of ``input_ids`` used to make this
            decision.
        batch_size: Batch dimension of ``input_ids``.
        chunk_start: For chunked prefill, the inclusive start token index
            of this chunk (0-based).  Always 0 for decode.
        chunk_end: For chunked prefill, the exclusive end token index.
            Equal to ``seq_len`` for non-chunked decisions.
        chunk_idx: Zero-based chunk index (0 for the first / only chunk).
        is_chunked: True if this decision is one part of a multi-chunk
            prefill.
        tca_tbe_enabled: Whether TCA-TBE compression is active for this
            decision.  When False, ``kernel_pipeline`` is NUMPY.
    """

    stage: InferenceStage
    kernel_pipeline: str   # KernelPipeline value stored as str for serializability
    seq_len: int
    batch_size: int
    chunk_start: int = 0
    chunk_end: int = 0
    chunk_idx: int = 0
    is_chunked: bool = False
    tca_tbe_enabled: bool = True

    @property
    def chunk_seq_len(self) -> int:
        """Number of tokens in this dispatch's chunk."""
        return self.chunk_end - self.chunk_start

    def __post_init__(self) -> None:
        # chunk_end defaults: normalise 0 → seq_len on construction.
        if self.chunk_end == 0:
            # frozen dataclass: use object.__setattr__ for post-init normalisation
            object.__setattr__(self, "chunk_end", self.seq_len)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class StageDispatcher:
    """Prefill/decode stage detector and kernel pipeline selector.

    Parameters:
        tca_tbe_enabled: When False, always routes to the NumPy fallback
            pipeline regardless of stage.  Useful for testing and for
            hardware without Metal support.
        chunk_size: Default number of tokens per prefill chunk.  The last
            chunk may be smaller.
    """

    _DEFAULT_CHUNK_SIZE: int = 512

    def __init__(
        self,
        tca_tbe_enabled: bool = True,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._tca_tbe_enabled = tca_tbe_enabled
        self._chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tca_tbe_enabled(self) -> bool:
        """Whether TCA-TBE dispatch is active."""
        return self._tca_tbe_enabled

    @property
    def chunk_size(self) -> int:
        """Default prefill chunk size (tokens)."""
        return self._chunk_size

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_input_ids(input_ids: np.ndarray) -> None:
        """Raise ``ValueError`` if *input_ids* is not a valid 2-D integer array.

        Args:
            input_ids: Expected shape ``(batch_size, seq_len)``.

        Raises:
            ValueError: Shape not 2-D or unsupported dtype.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2-D (batch, seq_len), got shape {input_ids.shape}"
            )
        if not np.issubdtype(input_ids.dtype, np.integer):
            raise ValueError(
                f"input_ids dtype must be integer, got {input_ids.dtype}"
            )

    @staticmethod
    def detect_stage(input_ids: np.ndarray) -> InferenceStage:
        """Determine the inference stage from *input_ids* shape.

        Args:
            input_ids: Shape ``(batch_size, seq_len)``.

        Returns:
            :attr:`InferenceStage.DECODE` if ``seq_len == 1``, otherwise
            :attr:`InferenceStage.PREFILL`.

        Raises:
            ValueError: If *input_ids* is not 2-D.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2-D (batch, seq_len), got shape {input_ids.shape}"
            )
        seq_len = input_ids.shape[1]
        return InferenceStage.DECODE if seq_len == 1 else InferenceStage.PREFILL

    # ------------------------------------------------------------------
    # Primary dispatch
    # ------------------------------------------------------------------

    def _select_pipeline(self, stage: InferenceStage) -> str:
        if not self._tca_tbe_enabled:
            return KernelPipeline.NUMPY.value
        if stage == InferenceStage.DECODE:
            return KernelPipeline.ZIP_GEMV.value
        return KernelPipeline.ZIP_GEMM.value

    def dispatch(self, input_ids: np.ndarray) -> DispatchDecision:
        """Return a single :class:`DispatchDecision` for *input_ids*.

        For prefill requests longer than ``chunk_size``, this method still
        returns one decision covering the *full* sequence without chunking.
        Use :meth:`dispatch_chunked` to iterate over chunks.

        Args:
            input_ids: Shape ``(batch_size, seq_len)`` integer array.

        Returns:
            :class:`DispatchDecision` with ``is_chunked=False``.

        Raises:
            ValueError: If *input_ids* is not a valid 2-D integer array.
        """
        self.validate_input_ids(input_ids)
        batch_size, seq_len = input_ids.shape
        stage = self.detect_stage(input_ids)
        pipeline = self._select_pipeline(stage)
        return DispatchDecision(
            stage=stage,
            kernel_pipeline=pipeline,
            seq_len=seq_len,
            batch_size=batch_size,
            chunk_start=0,
            chunk_end=seq_len,
            chunk_idx=0,
            is_chunked=False,
            tca_tbe_enabled=self._tca_tbe_enabled,
        )

    # ------------------------------------------------------------------
    # Chunked prefill
    # ------------------------------------------------------------------

    def dispatch_chunked(
        self,
        input_ids: np.ndarray,
        chunk_size: Optional[int] = None,
    ) -> Iterator[DispatchDecision]:
        """Yield one :class:`DispatchDecision` per chunk of *input_ids*.

        For decode steps (``seq_len == 1``) a single decision is yielded
        with ``is_chunked=False``.  For prefill, the sequence is split into
        ceil(seq_len / chunk_size) chunks; each decision has ``is_chunked=True``
        (or False when there is exactly one chunk).

        Args:
            input_ids: Shape ``(batch_size, seq_len)`` integer array.
            chunk_size: Override the dispatcher's default chunk size for
                this call only.

        Yields:
            :class:`DispatchDecision` for each chunk.

        Raises:
            ValueError: If *input_ids* is invalid or *chunk_size* < 1.
        """
        self.validate_input_ids(input_ids)
        eff_chunk = chunk_size if chunk_size is not None else self._chunk_size
        if eff_chunk < 1:
            raise ValueError(f"chunk_size must be >= 1, got {eff_chunk}")

        batch_size, seq_len = input_ids.shape
        stage = self.detect_stage(input_ids)
        pipeline = self._select_pipeline(stage)

        if stage == InferenceStage.DECODE or seq_len <= eff_chunk:
            yield DispatchDecision(
                stage=stage,
                kernel_pipeline=pipeline,
                seq_len=seq_len,
                batch_size=batch_size,
                chunk_start=0,
                chunk_end=seq_len,
                chunk_idx=0,
                is_chunked=False,
                tca_tbe_enabled=self._tca_tbe_enabled,
            )
            return

        # Multi-chunk prefill.
        chunk_idx = 0
        for start in range(0, seq_len, eff_chunk):
            end = min(start + eff_chunk, seq_len)
            yield DispatchDecision(
                stage=stage,
                kernel_pipeline=pipeline,
                seq_len=seq_len,
                batch_size=batch_size,
                chunk_start=start,
                chunk_end=end,
                chunk_idx=chunk_idx,
                is_chunked=True,
                tca_tbe_enabled=self._tca_tbe_enabled,
            )
            chunk_idx += 1
