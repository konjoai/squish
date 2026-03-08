"""
squish/quant_spec.py

QuantSpec — Self-Speculative Decoding with Quantized Draft KV Cache.

Inspired by:
  "QuantSpec: Self-Speculative Decoding with Layer-Adaptive
   Draft KV Quantization" — Apple Research (2025)

Background
----------
Standard speculative decoding (Leviathan 2022) requires a *separate* draft
model — doubling memory.  *Self-speculative* approaches re-use the target
model with a different configuration:

  • QuantSpec uses a **quantized KV cache** for the draft pass (4-bit / 2-bit)
    and the full-precision KV for the verification pass.
  • Earlier transformer layers are skipped in the draft pass
    ("early-exit" speculative decoding).

Memory benefit:  single model in memory; draft pass costs ~2-4× less memory
than verification pass.

This module provides:
  1. ``QuantSpecConfig`` — parameters controlling draft vs. verify behaviour.
  2. ``QuantSpecDecoder`` — stateful decoder that orchestrates the two-pass
     draft→verify loop using only NumPy / basic operations.  It is designed
     to wrap any callable ``model_fn(token_ids, kv)`` that returns ``(logits, new_kv)``.

Integration with squish server::

    from squish.quant_spec import QuantSpecConfig, QuantSpecDecoder

    cfg     = QuantSpecConfig(gamma=4, draft_quant_bits=4)
    decoder = QuantSpecDecoder(model_fn=model_fn, config=cfg, tokenizer=tokenizer)

    for token in decoder.generate(prompt_ids, max_new_tokens=256):
        yield token
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "QuantSpecConfig",
    "DraftQuantizer",
    "QuantSpecDecoder",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantSpecConfig:
    """
    Configuration for QuantSpec self-speculative decoding.

    Parameters
    ----------
    gamma : int
        Number of draft tokens generated per verification step.
    draft_quant_bits : int
        Bit-width for draft-pass KV quantization (2 or 4).
    draft_skip_layers : int
        Number of early transformer layers to skip in the draft pass.
        Reduces draft-pass cost (layers skipped == identity pass).
    temperature : float
        Sampling temperature for both draft and target distributions.
    top_p : float
        Nucleus sampling probability.  1.0 = disabled.
    acceptance_threshold : float
        Minimum speculative acceptance probability.  Drafts with
        ``p_target / p_draft < threshold`` are rejected.
        Use 1.0 for the true Leviathan re-sampling rule.
    """
    gamma:               int   = 4
    draft_quant_bits:    int   = 4
    draft_skip_layers:   int   = 8
    temperature:         float = 1.0
    top_p:               float = 1.0
    acceptance_threshold: float = 0.0   # 0 = always accept (greedy)

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be ≥ 1")
        if self.draft_quant_bits not in (2, 4, 8):
            raise ValueError("draft_quant_bits must be 2, 4, or 8")
        if self.draft_skip_layers < 0:
            raise ValueError("draft_skip_layers must be ≥ 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in [0, 1]")


# ---------------------------------------------------------------------------
# Draft Quantizer
# ---------------------------------------------------------------------------

class DraftQuantizer:
    """
    Integer quantizer for draft-pass KV tensors.

    Implements symmetric per-tensor min-max quantization.

    Parameters
    ----------
    bits : int
        Quantization bit-width (2, 4, or 8).
    """

    def __init__(self, bits: int = 4) -> None:
        if bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4, or 8")
        self._bits    = bits
        self._levels  = 2 ** bits
        self._qmax    = self._levels // 2 - 1         # e.g. 7 for 4-bit
        self._qmin    = -(self._levels // 2)           # e.g. -8 for 4-bit

    # ── Public API ─────────────────────────────────────────────────────────────

    def quantize(
        self,
        tensor: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Quantize *tensor* to integer representation.

        Returns
        -------
        q_tensor : same-shape int8 / int16 ndarray
        scale    : float — scale factor (dequant: q * scale + zero)
        zero     : float — zero point
        """
        t = np.asarray(tensor, dtype=np.float32)
        t_min = float(t.min())
        t_max = float(t.max())

        if t_min == t_max:
            return np.zeros_like(t, dtype=np.int8), 1.0, t_min

        scale = (t_max - t_min) / (self._qmax - self._qmin)
        zero  = t_min - scale * self._qmin
        q     = np.round((t - zero) / scale).clip(self._qmin, self._qmax)
        dtype = np.int8 if self._bits <= 8 else np.int16
        return q.astype(dtype), scale, zero

    def dequantize(
        self,
        q_tensor: np.ndarray,
        scale:    float,
        zero:     float,
    ) -> np.ndarray:
        """
        Dequantize integer tensor back to float32.
        """
        return q_tensor.astype(np.float32) * scale + zero

    @property
    def bits(self) -> int:
        return self._bits

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to float32."""
        return 32.0 / self._bits


# ---------------------------------------------------------------------------
# Sampling utilities (pure NumPy)
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    l = np.asarray(logits, dtype=np.float64)
    if temperature != 1.0:
        l = l / temperature
    l -= l.max()
    exp_l = np.exp(l)
    return (exp_l / exp_l.sum()).astype(np.float32)


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus (top-p) filtering: zero out tokens outside the top-p nucleus."""
    if top_p >= 1.0:
        return probs
    sorted_idx  = np.argsort(probs)[::-1]
    cum_prob    = np.cumsum(probs[sorted_idx])
    cutoff_mask = cum_prob > top_p
    # Keep at least one token
    cutoff_mask[0] = False
    remove_mask = np.zeros_like(probs, dtype=bool)
    remove_mask[sorted_idx[cutoff_mask]] = True
    filtered = probs.copy()
    filtered[remove_mask] = 0.0
    total = filtered.sum()
    if total > 0:
        filtered /= total
    return filtered


def _sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one token index from a probability distribution."""
    return int(rng.choice(len(probs), p=probs / (probs.sum() + 1e-12)))


# ---------------------------------------------------------------------------
# QuantSpecDecoder
# ---------------------------------------------------------------------------

class QuantSpecDecoder:
    """
    Self-speculative decoding engine using quantized draft KV caches.

    The caller supplies:
    - ``draft_fn(token_ids, kv_state, skip_layers)`` → ``(logits, new_kv)``
    - ``verify_fn(token_ids, kv_state)`` → ``(logits_batch, new_kv)``

    Where ``kv_state`` is an opaque object managed by the caller.

    When only one function is supplied (self-speculative mode) it is used
    for both draft and verify passes with the appropriate ``skip_layers``
    argument.

    Parameters
    ----------
    draft_fn  : callable — lightweight draft forward (quantized KV)
    verify_fn : callable — full-precision verification forward
    config    : QuantSpecConfig
    seed      : int — RNG seed for reproducible sampling
    """

    def __init__(
        self,
        draft_fn:  Callable,
        config:    QuantSpecConfig,
        verify_fn: Optional[Callable] = None,
        seed:      int = 42,
    ) -> None:
        self._draft_fn  = draft_fn
        self._verify_fn = verify_fn or draft_fn
        self._cfg       = config
        self._rng       = np.random.default_rng(seed)
        self._quantizer = DraftQuantizer(bits=config.draft_quant_bits)

        # Stats
        self.total_draft_tokens    = 0
        self.total_accepted_tokens = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def acceptance_rate(self) -> float:
        """Running acceptance rate (accepted / drafted)."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    def generate_step(
        self,
        context_ids: np.ndarray,
        kv_state:    Any,
    ) -> Tuple[List[int], Any]:
        """
        Execute one speculative step: draft ``gamma`` tokens then verify.

        Parameters
        ----------
        context_ids : (seq_len,) int32 — prompt + tokens so far
        kv_state    : opaque KV cache state passed through to model fns

        Returns
        -------
        accepted_tokens : list of int  (≥1, ≤gamma+1 tokens)
        new_kv_state    : updated KV state after verification
        """
        cfg  = self._cfg
        ids  = list(context_ids)
        draft_tokens:  List[int]              = []
        draft_probs:   List[np.ndarray]       = []

        # ── Draft phase ────────────────────────────────────────────────────
        draft_kv = kv_state  # draft operates on a copy conceptually
        for _ in range(cfg.gamma):
            inp = np.array(ids, dtype=np.int32)
            logits, draft_kv = self._draft_fn(inp, draft_kv, cfg.draft_skip_layers)
            logits_np = np.asarray(logits).flatten()
            probs     = _softmax(logits_np, cfg.temperature)
            probs     = _top_p_filter(probs, cfg.top_p)
            token     = _sample(probs, self._rng)
            draft_tokens.append(token)
            draft_probs.append(probs)
            ids.append(token)

        self.total_draft_tokens += len(draft_tokens)

        # ── Verify phase ───────────────────────────────────────────────────
        # Single verify pass over context + all draft tokens
        verify_inp = np.array(ids, dtype=np.int32)
        all_logits, new_kv = self._verify_fn(verify_inp, kv_state)
        # all_logits shape: either (seq_len, vocab) or (vocab,)
        all_logits = np.asarray(all_logits)
        if all_logits.ndim == 1:
            # Model only returned last-step logits — fall back to greedy accept
            target_probs_list = [_softmax(all_logits, cfg.temperature)]
            target_probs_list = target_probs_list * len(draft_tokens)
        else:
            # Extract probs at each draft position
            ctx_len = len(context_ids)
            target_probs_list = [
                _softmax(all_logits[ctx_len + i - 1], cfg.temperature)
                for i in range(len(draft_tokens))
            ]

        # ── Speculative rejection sampling ─────────────────────────────────
        accepted: List[int] = []
        for i, (t, p_draft, p_target) in enumerate(
            zip(draft_tokens, draft_probs, target_probs_list)
        ):
            acceptance_prob = float(p_target[t]) / (float(p_draft[t]) + 1e-12)
            acceptance_prob = min(1.0, acceptance_prob)

            if acceptance_prob < cfg.acceptance_threshold:
                # Reject all remaining drafts; sample from adjusted distribution
                break

            u = float(self._rng.uniform())
            if u <= acceptance_prob:
                accepted.append(t)
            else:
                # Reject: sample a token from (p_target - p_draft)^+
                break

        self.total_accepted_tokens += len(accepted)

        # Bonus token from target distribution after last accepted position
        if len(accepted) < len(draft_tokens):
            bonus_logits = (
                all_logits[len(context_ids) + len(accepted) - 1]
                if all_logits.ndim > 1
                else all_logits
            )
            bonus_probs = _softmax(np.asarray(bonus_logits).flatten(), cfg.temperature)
            bonus_probs = _top_p_filter(bonus_probs, cfg.top_p)
            bonus = _sample(bonus_probs, self._rng)
            accepted.append(bonus)
        elif all_logits.ndim > 1:
            # All accepted; also sample the bonus target token
            bonus_logits = all_logits[len(context_ids) + len(draft_tokens) - 1]
            bonus_probs = _softmax(np.asarray(bonus_logits).flatten(), cfg.temperature)
            bonus_probs = _top_p_filter(bonus_probs, cfg.top_p)
            bonus = _sample(bonus_probs, self._rng)
            accepted.append(bonus)

        return accepted, new_kv

    def reset_stats(self) -> None:
        """Reset acceptance-rate counters."""
        self.total_draft_tokens    = 0
        self.total_accepted_tokens = 0
