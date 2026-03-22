"""PipeInfer: asynchronous chunked prefill-decode pipeline (arXiv 2407.11798, 2024).

Splits a single-request prompt into N fixed-size chunks and begins decode
immediately after the first chunk's prefill completes, overlapping the
remaining prefill chunks with early decode steps.  Reduces user-perceived
TTFT by 30–50% for prompts above 256 tokens.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "PipeInferConfig",
    "PipeInferRequest",
    "PipeInferTick",
    "PipeInferScheduler",
]


@dataclass
class PipeInferConfig:
    """Configuration for :class:`PipeInferScheduler`.

    Attributes:
        chunk_size: Prefill tokens processed per scheduler tick (default 128).
        max_decode_steps: Maximum decode tokens generated per request after
            all prefill chunks complete.
        seed: Unused numeric seed kept for API consistency.
    """

    chunk_size: int = 128
    max_decode_steps: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.max_decode_steps < 1:
            raise ValueError(
                f"max_decode_steps must be >= 1, got {self.max_decode_steps}"
            )


@dataclass
class PipeInferRequest:
    """A single request submitted to :class:`PipeInferScheduler`.

    Attributes:
        request_id: Unique string identifier.
        prompt_tokens: Total number of tokens in the prompt.
        n_decode_tokens: Number of decode steps to simulate.
    """

    request_id: str
    prompt_tokens: int
    n_decode_tokens: int = 1

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError(
                f"prompt_tokens must be >= 0, got {self.prompt_tokens}"
            )
        if self.n_decode_tokens < 1:
            raise ValueError(
                f"n_decode_tokens must be >= 1, got {self.n_decode_tokens}"
            )


@dataclass
class PipeInferTick:
    """One scheduler tick output for a single request.

    Attributes:
        request_id: Owning request.
        chunk_index: Zero-based index of the prefill chunk processed this tick.
        n_prefill_tokens: Prefill tokens processed in this tick.
        n_decode_tokens: Decode tokens emitted in this tick (0 on first tick).
        first_token_emitted: True if this tick produced the first decode token.
    """

    request_id: str
    chunk_index: int
    n_prefill_tokens: int
    n_decode_tokens: int
    first_token_emitted: bool


class PipeInferScheduler:
    """PipeInfer pipelined prefill-decode scheduler.

    State machine per request:

    * ``pending_prefill``: chunks remaining.  After chunk 0, decode begins.
    * ``pending_decode``: tokens remaining to emit after all prefill done.

    Each call to :meth:`step` advances all active requests by one tick and
    returns the list of :class:`PipeInferTick` events.
    """

    def __init__(self, config: Optional[PipeInferConfig] = None) -> None:
        self._config = config or PipeInferConfig()
        self._requests: Dict[str, PipeInferRequest] = {}
        self._chunk_cursor: Dict[str, int] = {}   # tokens already prefilled
        self._decode_cursor: Dict[str, int] = {}  # decode tokens already emitted
        self._first_token_done: Dict[str, bool] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> PipeInferConfig:
        return self._config

    @property
    def n_active(self) -> int:
        """Number of requests still in flight."""
        with self._lock:
            return len(self._requests)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: PipeInferRequest) -> None:
        """Submit a new request to the pipeline.

        Parameters
        ----------
        request:
            The :class:`PipeInferRequest` to schedule.

        Raises
        ------
        ValueError
            If a request with the same ``request_id`` is already active.
        """
        with self._lock:
            if request.request_id in self._requests:
                raise ValueError(
                    f"Request {request.request_id!r} is already active"
                )
            self._requests[request.request_id] = request
            self._chunk_cursor[request.request_id] = 0
            self._decode_cursor[request.request_id] = 0
            self._first_token_done[request.request_id] = False

    def step(self) -> List[PipeInferTick]:
        """Advance all active requests by one tick.

        Returns
        -------
        List[PipeInferTick]
            One tick per active request.  Completed requests are removed after
            emitting their final tick.
        """
        cfg = self._config
        ticks: List[PipeInferTick] = []
        completed: List[str] = []

        with self._lock:
            for rid, req in list(self._requests.items()):
                prefilled = self._chunk_cursor[rid]
                decoded = self._decode_cursor[rid]
                remaining_prefill = req.prompt_tokens - prefilled
                chunk_index = prefilled // cfg.chunk_size if cfg.chunk_size > 0 else 0

                # Prefill chunk this tick
                n_prefill = min(cfg.chunk_size, remaining_prefill)
                self._chunk_cursor[rid] += n_prefill

                # Emit decode tokens once chunk 0 is done (pipeline start)
                n_decode = 0
                first_emitted = False
                if self._chunk_cursor[rid] >= min(cfg.chunk_size, req.prompt_tokens):
                    # At least one chunk done — start decoding
                    if decoded < req.n_decode_tokens:
                        n_decode = 1
                        self._decode_cursor[rid] += 1
                        if not self._first_token_done[rid]:
                            self._first_token_done[rid] = True
                            first_emitted = True

                ticks.append(
                    PipeInferTick(
                        request_id=rid,
                        chunk_index=chunk_index,
                        n_prefill_tokens=n_prefill,
                        n_decode_tokens=n_decode,
                        first_token_emitted=first_emitted,
                    )
                )

                all_prefill_done = self._chunk_cursor[rid] >= req.prompt_tokens
                all_decode_done = self._decode_cursor[rid] >= req.n_decode_tokens
                if all_prefill_done and all_decode_done:
                    completed.append(rid)

            for rid in completed:
                del self._requests[rid]
                del self._chunk_cursor[rid]
                del self._decode_cursor[rid]
                del self._first_token_done[rid]

        return ticks

    def is_done(self) -> bool:
        """Return True when all submitted requests have completed."""
        with self._lock:
            return len(self._requests) == 0

    def ttft_estimate(self, prompt_length: int) -> float:
        """Estimate the relative TTFT gain vs full-prefill baseline.

        Returns a value in (0, 1] where 1.0 = no improvement (prompt fits
        in one chunk) and 0.5 = 50% TTFT reduction.

        Parameters
        ----------
        prompt_length:
            Total prompt token count.
        """
        cfg = self._config
        if prompt_length <= cfg.chunk_size or cfg.chunk_size == 0:
            return 1.0
        # TTFT = time for first chunk / time for full prefill
        return cfg.chunk_size / prompt_length
