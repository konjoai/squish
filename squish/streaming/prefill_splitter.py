"""
squish/streaming/prefill_splitter.py

Adaptive prefill chunk splitter for minimum time-to-first-token (TTFT).

Prefill attention cost is O(seq²) — processing the entire prompt as one
operation maximises throughput but leaves the user waiting until all
prompt tokens are computed.  Splitting prefill into chunks allows yielding
the first token after only the *first chunk* completes.

This module implements an adaptive chunk-size selector that:

1. Maintains an EMA of measured prefill throughput (tokens / second).
2. After each chunk, solves for the chunk size that would hit exactly
   ``config.target_ttft_ms`` given current throughput.
3. Clamps the result to ``[min_chunk_size, max_chunk_size]``.

At calibration convergence the first chunk is sized to arrive exactly at
the TTFT target while subsequent chunks use the maximum size for throughput.

Based on Sarathi-Serve: "Taming Throughput-Latency Tradeoff in LLM Inference
with Sarathi-Serve" (Agrawal et al., NeurIPS 2024, arXiv:2403.02310) with
online adaptive sizing not present in the original paper.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, List, Optional, Tuple


@dataclass
class PrefillSplitterConfig:
    """Configuration for the adaptive prefill chunk splitter.

    Parameters
    ----------
    min_chunk_size:
        Minimum number of tokens per prefill chunk.
    max_chunk_size:
        Maximum tokens per chunk.  First chunk is bounded by current optimal
        size; subsequent chunks use this maximum for throughput.
    initial_chunk_size:
        Starting chunk size before any calibration measurements are made.
    throughput_floor_tps:
        Minimum decode throughput to maintain (tokens/sec).  If positive,
        the chunk size will never be grown beyond a size that would cause
        queued decode requests to starve.  Set to 0.0 to disable.
    alpha:
        EMA smoothing factor for throughput estimates (0 < alpha <= 1).
        Higher values track recent performance more aggressively.
    calibration_window:
        Circular buffer depth for storing raw (chunk_size, tps) measurements.
    target_ttft_ms:
        Target time-to-first-token in milliseconds used to compute the
        optimal first chunk size.
    """

    min_chunk_size: int = 64
    max_chunk_size: int = 2_048
    initial_chunk_size: int = 512
    throughput_floor_tps: float = 0.0
    alpha: float = 0.25
    calibration_window: int = 20
    target_ttft_ms: float = 200.0

    def __post_init__(self) -> None:
        if self.min_chunk_size < 1:
            raise ValueError("min_chunk_size must be >= 1")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.calibration_window < 1:
            raise ValueError("calibration_window must be >= 1")
        if self.target_ttft_ms <= 0:
            raise ValueError("target_ttft_ms must be positive")
        if self.throughput_floor_tps < 0:
            raise ValueError("throughput_floor_tps must be >= 0")


class PrefillSplitter:
    """Adaptive prefill chunk-size optimiser for minimum TTFT.

    Splits an incoming prompt token sequence into a first TTFT-optimised
    chunk and subsequent max-throughput chunks, then adapts the first chunk
    size based on measured per-chunk throughput via EMA.

    Usage
    -----
    ::

        splitter = PrefillSplitter(PrefillSplitterConfig(target_ttft_ms=150))
        for chunk in splitter.split(prompt_token_ids):
            t0 = time.perf_counter()
            run_prefill(chunk)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            splitter.record_chunk(len(chunk), elapsed_ms)
    """

    def __init__(self, config: Optional[PrefillSplitterConfig] = None) -> None:
        self.config = config or PrefillSplitterConfig()
        self._chunk_size: int = self.config.initial_chunk_size
        self._measurements: Deque[Tuple[int, float]] = deque(
            maxlen=self.config.calibration_window
        )
        self._ema_tps: float = 0.0
        self._n_measurements: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, token_ids: List[int]) -> Iterator[List[int]]:
        """Split a token sequence into prefill chunks.

        The first chunk is sized at ``current_chunk_size`` (TTFT-optimal).
        Subsequent chunks are sized at ``config.max_chunk_size`` for maximum
        throughput.

        Parameters
        ----------
        token_ids:
            Full prompt token sequence.

        Yields
        ------
        List[int]
            Successive token sub-sequences.
        """
        seq_len = len(token_ids)
        if seq_len == 0:
            return

        pos = 0
        first = True
        while pos < seq_len:
            size = self._chunk_size if first else self.config.max_chunk_size
            first = False
            end = min(pos + size, seq_len)
            yield token_ids[pos:end]
            pos = end

    def chunk_count(self, seq_len: int) -> int:
        """Estimate the number of chunks for a given sequence length.

        Parameters
        ----------
        seq_len:
            Length of the prompt in tokens.
        """
        if seq_len <= 0:
            return 0
        if self._chunk_size >= seq_len:
            return 1
        remainder = seq_len - self._chunk_size
        return 1 + max(
            1, (remainder + self.config.max_chunk_size - 1) // self.config.max_chunk_size
        )

    def record_chunk(self, chunk_size: int, elapsed_ms: float) -> None:
        """Record observed throughput for a processed prefill chunk.

        Call this once per chunk immediately after the prefill forward pass
        completes.  The EMA is updated and the chunk size is re-calibrated.

        Parameters
        ----------
        chunk_size:
            Number of tokens in the chunk just processed.
        elapsed_ms:
            Wall-clock time for the prefill forward pass in milliseconds.
        """
        if elapsed_ms <= 0 or chunk_size <= 0:
            return

        tps = (chunk_size / elapsed_ms) * 1_000.0
        self._measurements.append((chunk_size, tps))

        alpha = self.config.alpha
        if self._n_measurements == 0:
            self._ema_tps = tps
        else:
            self._ema_tps = alpha * tps + (1.0 - alpha) * self._ema_tps
        self._n_measurements += 1

        self._adapt()

    def set_chunk_size(self, size: int) -> None:
        """Manually override the current first-chunk size.

        Useful for benchmarking or externally driven tuning.
        """
        cfg = self.config
        self._chunk_size = max(cfg.min_chunk_size, min(cfg.max_chunk_size, size))

    def estimated_ttft_ms(self, chunk_size: Optional[int] = None) -> float:
        """Estimate TTFT for a given chunk size using current throughput.

        Parameters
        ----------
        chunk_size:
            Chunk size to estimate for.  Defaults to ``current_chunk_size``.

        Returns
        -------
        float
            Estimated TTFT in milliseconds.  Returns ``inf`` if no throughput
            estimate is available yet.
        """
        size = chunk_size if chunk_size is not None else self._chunk_size
        if self._ema_tps <= 0:
            return float("inf")
        return (size / self._ema_tps) * 1_000.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_chunk_size(self) -> int:
        """Active first-chunk size used by the next ``split`` call."""
        return self._chunk_size

    @property
    def estimated_tps(self) -> float:
        """EMA-smoothed prefill throughput estimate in tokens / second."""
        return self._ema_tps

    @property
    def n_measurements(self) -> int:
        """Total number of chunk measurements recorded."""
        return self._n_measurements

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adapt(self) -> None:
        """Adjust first-chunk size to target the configured TTFT."""
        cfg = self.config
        if self._ema_tps <= 0:
            return
        # Solve: target_ttft = chunk_size / ema_tps * 1000
        optimal = int(cfg.target_ttft_ms * self._ema_tps / 1_000.0)
        self._chunk_size = max(
            cfg.min_chunk_size, min(cfg.max_chunk_size, optimal)
        )
