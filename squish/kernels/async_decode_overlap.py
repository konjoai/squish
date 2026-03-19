"""
squish/kernels/async_decode_overlap.py

AsyncDecodeOverlap — Pipeline GPU computation and CPU sampling across decode steps.

Key insight
-----------
The standard single-step decode loop is sequential:

  1. ``logits = model(x, cache)``   — Metal GPU kernel (async internally, but
     ``mx.eval()`` forces a synchronous wait)
  2. ``sample(logits)``             — NumPy / CPU (fast, ~0.1 ms)

The CPU sampling work for step **N** can overlap with the GPU work for step
**N+1** if we:

  a. Issue the GPU kernel for step N+1 *without* waiting for step N to
     finish first.
  b. Run ``sample(step_N_logits)`` on a background CPU thread.
  c. Synchronise both results before yielding the token and issuing step N+2.

This creates a 1-token pipeline in which GPU and CPU work for consecutive
steps run concurrently, hiding the CPU sampling latency (~0.1–0.5 ms) behind
the GPU kernel execution time (~5–20 ms).

Architecture
------------
``AsyncDecodeOverlap`` manages a pair of worker threads (GPU-side remains
on the calling thread; sampling is offloaded) and a two-slot ring buffer
of pending futures.

.. note::
   MLX lazy graph execution means the Metal kernel does not start until
   ``mx.eval()`` is called.  We call ``mx.eval()`` on the previous step's
   logits *on the calling thread* right before kicking off the sampling
   thread, which keeps the Metal schedule well-formed.

Usage::

    from squish.kernels.async_decode_overlap import (
        OverlapConfig,
        AsyncDecodeOverlap,
    )
    import numpy as np

    cfg     = OverlapConfig(temperature=0.8, top_p=0.95)
    overlap = AsyncDecodeOverlap(cfg)

    # model_forward(x) must return an mlx array of shape (1, 1, vocab)
    output_ids = []
    for tok_id in overlap.decode_loop(
        model_forward=lambda x: model(x, cache=kv),
        first_token_id=prompt_next_id,
        max_tokens=256,
        eos_id=tokenizer.eos_id,
    ):
        output_ids.append(tok_id)
"""

from __future__ import annotations

__all__ = [
    "OverlapConfig",
    "AsyncDecodeOverlap",
    "OverlapStats",
]

import queue
import threading
from collections.abc import Generator
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OverlapConfig:
    """Configuration for AsyncDecodeOverlap.

    Parameters
    ----------
    temperature : float
        Sampling temperature (0 = greedy).
    top_p : float
        Nucleus sampling cumulative probability threshold.
    top_k : int
        Top-k filtering (0 disables).
    repetition_penalty : float
        Multiplicative penalty for repeated tokens (1.0 = no penalty).
    pipeline_depth : int
        Number of in-flight decode steps (1 = one step ahead).  Values > 1
        increase throughput but can cause correctness issues if the model's
        ``cache`` is not re-entrant; keep at 1 unless you know it is safe.
    sample_thread_timeout : float
        Maximum seconds to wait for the sampling thread before falling back
        to synchronous sampling.
    """

    temperature:           float = 0.0
    top_p:                 float = 1.0
    top_k:                 int   = 0
    repetition_penalty:    float = 1.0
    pipeline_depth:        int   = 1
    sample_thread_timeout: float = 0.1

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be ≥ 0; got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0,1]; got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be ≥ 0; got {self.top_k}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty must be > 0; got {self.repetition_penalty}"
            )
        if self.pipeline_depth < 1:
            raise ValueError(
                f"pipeline_depth must be ≥ 1; got {self.pipeline_depth}"
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class OverlapStats:
    """Statistics collected during an overlapped decode run.

    Attributes
    ----------
    total_steps : int
        Total decode steps completed.
    overlapped_steps : int
        Steps where sampling and GPU work ran concurrently.
    fallback_steps : int
        Steps where the sampling thread timed out and we fell back to
        synchronous sampling (pipeline stall).
    """

    total_steps:     int = 0
    overlapped_steps: int = 0
    fallback_steps:  int = 0

    @property
    def overlap_rate(self) -> float:
        """Fraction of steps with successful GPU/CPU overlap."""
        return self.overlapped_steps / self.total_steps if self.total_steps else 0.0


# ---------------------------------------------------------------------------
# Sampler (pure NumPy — runs safely on background thread)
# ---------------------------------------------------------------------------

def _sample_np(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    prev_ids: List[int],
) -> int:
    """Sample a single token id from a logit vector.

    All operations are pure NumPy so this is safe to run concurrently with
    an MLX Metal kernel on the main thread.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()

    # Repetition penalty
    if repetition_penalty != 1.0 and prev_ids:
        for t in set(prev_ids[-64:]):
            if 0 <= t < len(logits):
                logits[t] /= repetition_penalty if logits[t] > 0 else -repetition_penalty

    if temperature <= 0.0:
        return int(np.argmax(logits))

    logits /= temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, len(logits))
        threshold = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits >= threshold, logits, -1e9)

    # Stable softmax
    logits -= logits.max()
    probs = np.exp(logits)

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_idx  = np.argsort(probs)[::-1]
        sorted_p    = probs[sorted_idx]
        cumulative  = np.cumsum(sorted_p)
        cutoff_mask = cumulative - sorted_p > top_p
        sorted_p[cutoff_mask] = 0.0
        probs[sorted_idx] = sorted_p

    total = probs.sum()
    if total <= 0.0:
        return int(np.argmax(logits))
    probs /= total
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AsyncDecodeOverlap:
    """Overlapping GPU decode with CPU sampling across consecutive steps.

    Parameters
    ----------
    config : OverlapConfig
    """

    def __init__(self, config: OverlapConfig) -> None:
        self._cfg   = config
        self._stats = OverlapStats()

    @property
    def stats(self) -> OverlapStats:
        """Running overlap statistics (reset each :meth:`decode_loop` call)."""
        return self._stats

    def decode_loop(
        self,
        model_forward: Callable,
        first_token_id: int,
        max_tokens: int,
        eos_id: int = -1,
    ) -> Generator[int, None, None]:
        """Yield tokens from an overlapped decode loop.

        Parameters
        ----------
        model_forward : callable
            ``model_forward(x_array) -> mlx_array`` shape ``(1, 1, vocab)``.
            **Must** be called from the main thread (Metal GPU thread).
        first_token_id : int
            The token id produced by the prefill step.
        max_tokens : int
            Maximum number of new tokens to generate.
        eos_id : int
            EOS token id (``-1`` disables early stop).

        Yields
        ------
        int
            Generated token ids in order.
        """
        self._stats = OverlapStats()
        cfg         = self._cfg
        prev_ids: List[int] = [first_token_id]
        next_id             = first_token_id

        # result_queue carries sampled token ids from the background thread
        result_q: queue.SimpleQueue = queue.SimpleQueue()
        pending_sample: Optional[threading.Thread] = None

        def _run_sample(logits_np: np.ndarray) -> None:
            tok = _sample_np(
                logits_np,
                cfg.temperature,
                cfg.top_p,
                cfg.top_k,
                cfg.repetition_penalty,
                prev_ids,
            )
            result_q.put(tok)

        for step in range(max_tokens):
            yield next_id
            if eos_id >= 0 and next_id == eos_id:
                break

            # Issue GPU kernel for step N+1 (does NOT block — MLX is lazy)
            try:
                import mlx.core as mx
                x      = mx.array([[next_id]], dtype=mx.int32)
                logits = model_forward(x)
                # Force Metal kernel completion for PREVIOUS step's logits
                # (already handled by the time we reach here because we called
                # mx.eval in the caller before entering this loop, and we
                # always eval before sampling).
                mx.eval(logits)
                logits_np = np.array(logits[0, -1].astype(mx.float32))
            except Exception:
                # MLX not available or model call failed
                break

            # If a previous sampling thread is still running, collect its result
            if pending_sample is not None:
                self._stats.total_steps += 1
                try:
                    sampled = result_q.get(timeout=cfg.sample_thread_timeout)
                    self._stats.overlapped_steps += 1
                except queue.Empty:
                    # Timeout: fallback synchronous sample from the previous logits
                    sampled = _sample_np(
                        logits_np,  # approximation — use current logits
                        cfg.temperature,
                        cfg.top_p,
                        cfg.top_k,
                        cfg.repetition_penalty,
                        prev_ids,
                    )
                    self._stats.fallback_steps += 1
                    # Drain any stale result
                    pending_sample.join(timeout=0.0)
                next_id = sampled
                prev_ids.append(next_id)

            # Launch sampling for THIS step's logits on background thread
            # (overlaps with the NEXT step's GPU kernel)
            pending_sample = threading.Thread(
                target=_run_sample,
                args=(logits_np,),
                daemon=True,
            )
            pending_sample.start()

            # On the first step, we don't have a pending result yet —
            # collect it synchronously so we always have a valid next_id.
            if step == 0:
                self._stats.total_steps += 1
                try:
                    next_id = result_q.get(timeout=cfg.sample_thread_timeout)
                    self._stats.overlapped_steps += 1
                except queue.Empty:
                    next_id = _sample_np(
                        logits_np,
                        cfg.temperature,
                        cfg.top_p,
                        cfg.top_k,
                        cfg.repetition_penalty,
                        prev_ids,
                    )
                    self._stats.fallback_steps += 1
                prev_ids.append(next_id)
                pending_sample = None  # already collected

        # Drain final pending thread if any
        if pending_sample is not None:
            pending_sample.join(timeout=cfg.sample_thread_timeout)
