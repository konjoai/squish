#!/usr/bin/env python3
"""
squish/speculative.py

Speculative decoding for Squish — use a small draft model to propose K tokens
per step, then verify all K+1 positions with the large target model in a single
forward pass.

Algorithm: Leviathan et al., 2022 — "Fast Inference from Transformers via
Speculative Decoding", Algorithm 1.
https://arxiv.org/abs/2211.17192

Expected speedup vs auto-regressive decoding:
    7B  target + 0.5B draft  → 1.8–2.5×   (typical chat distributions)
    14B target + 1.5B draft  → 2.0–3.0×

Usage (from server.py, or standalone):
    from squish.speculative import SpeculativeGenerator, load_draft_model

    draft = load_draft_model(
        model_dir="~/models/Qwen2.5-0.5B-Instruct-bf16",
        compressed_dir="~/models/squish_0.5b",
    )
    gen = SpeculativeGenerator(target_model, target_tokenizer, draft)
    for token_text, finish_reason in gen.stream(prompt, max_tokens=512):
        ...

Standalone test:
    python3 -m squish.speculative \
        --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \
        --compressed-dir ~/models/squish_7b \
        --draft-model ~/models/Qwen2.5-0.5B-Instruct-bf16 \
        --draft-compressed ~/models/squish_0.5b \
        --prompt "Explain quantum entanglement in one paragraph."
"""

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning knobs ─────────────────────────────────────────────────────────────

_DEFAULT_K          = 4      # draft tokens to speculate per step
_DEFAULT_TEMP       = 0.7
_DEFAULT_TOP_P      = 0.9
_MAX_SPEC_TOKENS    = 8      # cap K at this regardless of setting

# ── Sampling helpers ──────────────────────────────────────────────────────────

def _softmax_np(logits_row: np.ndarray, temp: float) -> np.ndarray:
    """Compute softmax(logits / temp) in float64 for numerical stability."""
    logits64 = logits_row.astype(np.float64)
    logits64 /= max(temp, 1e-8)
    logits64 -= logits64.max()
    probs = np.exp(logits64)
    probs /= probs.sum()
    return probs.astype(np.float32)


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Zero out tokens outside the nucleus (top-p) and renormalise."""
    if top_p >= 1.0:
        return probs
    idx    = np.argsort(-probs)
    cumsum = np.cumsum(probs[idx])
    cutoff = int((cumsum <= top_p).sum()) + 1
    mask   = np.zeros_like(probs)
    mask[idx[:max(1, cutoff)]] = 1.0
    filtered = probs * mask
    total    = filtered.sum()
    return filtered / total if total > 0 else probs


def _sample(probs: np.ndarray) -> int:
    """Multinomial sample — returns a single token id."""
    try:
        return int(np.random.choice(len(probs), p=probs))
    except ValueError:
        # Numerical issues — fall back to argmax
        return int(np.argmax(probs))


def _greedy(logits_row: np.ndarray) -> int:
    return int(np.argmax(logits_row))


def _get_logits(model, ids: list[int]) -> np.ndarray:
    """
    Single synchronous forward pass.

    Returns the *last* row of logits as a float32 numpy array
    (shape: vocab_size).  This is the next-token prediction.
    """
    x      = mx.array(ids, dtype=mx.int32)[None]   # (1, seq_len)
    out    = model(x)                               # (1, seq_len, vocab)
    last   = out[0, -1]                             # (vocab,)
    mx.eval(last)
    return np.array(last, dtype=np.float32)


def _get_all_logits(model, ids: list[int], n_positions: int) -> np.ndarray:
    """
    Verification pass: run the model and return the last ``n_positions`` rows.

    Shape: (n_positions, vocab_size).
    Used to verify K draft tokens + produce one bonus token simultaneously.
    """
    x   = mx.array(ids, dtype=mx.int32)[None]      # (1, seq_len)
    out = model(x)                                  # (1, seq_len, vocab)
    # We want positions [-n_positions:] − the verification slice.
    rows = out[0, -n_positions:]                    # (n_positions, vocab)
    mx.eval(rows)
    return np.array(rows, dtype=np.float32)         # (n_positions, vocab)


# ── Stateful (KV-cached) helpers ──────────────────────────────────────────────
# These replace the stateless helpers above when mlx_lm's KV cache is
# available.  The stateless path is kept as a fallback.

def _try_make_model_cache(model):
    """
    Create an mlx_lm KV prompt-cache for *model*.

    Tries the public API (mlx_lm >= 0.18) first, then an older internal path.
    Returns ``None`` when neither is available OR when the model uses
    ``RotatingKVCache`` (whose internal state cannot be safely truncated by
    simply setting ``.offset``).
    """
    cache = None
    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except Exception:
        pass
    if cache is None:
        try:
            import mlx_lm.utils as _u
            cache = _u.make_kv_caches(model)
        except Exception:
            pass
    if cache is None:
        return None
    # RotatingKVCache wraps around — offset truncation would corrupt state.
    try:
        for c in cache:
            if "rotating" in type(c).__name__.lower():
                return None
    except Exception:
        return None
    return cache


def _cache_offset(cache) -> int:
    """Current token offset of the first cache entry (0 if unavailable)."""
    try:
        return cache[0].offset
    except Exception:
        return 0


def _cache_set_offset(cache, offset: int) -> None:
    """
    Roll back all cache entries to *offset* tokens.

    mlx_lm's ``KVCache`` stores K/V arrays and tracks position via
    ``.offset``.  Setting it back to a prior value causes the next forward
    pass to overwrite the rejected suffix — correct and allocation-free.
    """
    if cache is None:
        return
    try:
        for c in cache:
            c.offset = offset
    except Exception:
        pass


def _prefill_cached(model, cache, ids: list[int]) -> np.ndarray:
    """
    Prefill *model*'s KV cache with *ids*.

    Returns last-row logits (shape: ``vocab_size``) as float32 numpy —
    the prediction for the very next token.
    """
    x   = mx.array(ids, dtype=mx.int32)[None]   # (1, seq_len)
    out = model(x, cache=cache)                  # (1, seq_len, vocab)
    last = out[0, -1]
    mx.eval(last)
    return np.array(last, dtype=np.float32)


def _decode_step_cached(model, cache, token_id: int) -> np.ndarray:
    """Single-token incremental decode.  Returns next-token logits (vocab_size,)."""
    x   = mx.array([[token_id]], dtype=mx.int32)  # (1, 1)
    out = model(x, cache=cache)                   # (1, 1, vocab)
    last = out[0, -1]
    mx.eval(last)
    return np.array(last, dtype=np.float32)


def _decode_multi_cached(model, cache, token_ids: list[int]) -> np.ndarray:
    """
    Multi-token incremental decode with KV cache.

    Returns ALL output logit rows — shape ``(len(token_ids), vocab_size)``.
    Row ``j`` is the prediction for what follows ``token_ids[j]``.
    """
    x   = mx.array(token_ids, dtype=mx.int32)[None]  # (1, T)
    out = model(x, cache=cache)                       # (1, T, vocab)
    rows = out[0]
    mx.eval(rows)
    return np.array(rows, dtype=np.float32)


# ── Draft model loader ────────────────────────────────────────────────────────

def load_draft_model(
    model_dir: str,
    compressed_dir: str = "",
    verbose: bool = False,
):
    """
    Load the small draft model through the same compressed_loader pipeline.

    Returns (model, tokenizer).  Vocabulary must be compatible with the target
    (same tokeniser family — e.g. both Qwen2.5, both Llama…).
    """
    from .compressed_loader import load_compressed_model

    model_dir_p = Path(model_dir).expanduser()
    comp_dir_p  = Path(compressed_dir).expanduser() if compressed_dir else \
                  Path(model_dir_p.parent / (model_dir_p.name + "-compressed"))

    model, tokenizer = load_compressed_model(
        model_dir  = str(model_dir_p),
        npz_path   = str(comp_dir_p),
        verbose    = verbose,
    )
    return model, tokenizer


# ── Speculative Generator ─────────────────────────────────────────────────────

class SpeculativeGenerator:
    """
    Speculative decoding wrapper.

    Wraps a (target_model, target_tokenizer) pair and an optional
    (draft_model, draft_tokenizer).  When the draft model is provided it uses
    the speculative algorithm; otherwise it falls back to standard greedy/sampling.

    Both models are assumed to share a common vocabulary (same tokeniser family).
    Token IDs from the draft tokeniser are used as draft candidates and verified
    against the target's distribution.
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model             = None,
        draft_tokenizer         = None,
        k: int                  = _DEFAULT_K,
    ):
        self._target  = target_model
        self._ttok    = target_tokenizer
        self._draft   = draft_model
        self._dtok    = draft_tokenizer or target_tokenizer
        self._k       = min(max(1, k), _MAX_SPEC_TOKENS)

        # Acceptance stats (reset per stream() call)
        self.accepted_total  = 0
        self.proposed_total  = 0
        self.steps           = 0

        # ── Stateful KV caches ────────────────────────────────────────────────
        # Created once at init; reset (offset → 0) at the start of each
        # stream() call.  Both are None when mlx_lm's cache API is unavailable
        # or when the model uses RotatingKVCache — the stateless path is used
        # transparently in those cases.
        self._target_cache = _try_make_model_cache(target_model)
        self._draft_cache  = (
            _try_make_model_cache(draft_model) if draft_model is not None else None
        )
        _use_stateful = (
            self._target_cache is not None
            and (draft_model is None or self._draft_cache is not None)
        )
        logger.debug(
            "speculative: %s KV caches",
            "stateful" if _use_stateful else "stateless (fallback)",
        )

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_total / self.proposed_total
                if self.proposed_total > 0 else 0.0)

    def _reset_stats(self) -> None:
        self.accepted_total = 0
        self.proposed_total = 0
        self.steps = 0

    def _reset_caches(self) -> None:
        """Roll both KV caches back to position 0 (start of new request)."""
        _cache_set_offset(self._target_cache, 0)
        _cache_set_offset(self._draft_cache, 0)

    # ── main streaming API ────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_tokens: int       = 512,
        temperature: float    = _DEFAULT_TEMP,
        top_p: float          = _DEFAULT_TOP_P,
        stop_ids: list[list[int]] | None = None,
        seed: int | None   = None,
    ) -> Iterator[tuple[str, str | None]]:
        """
        Yield (token_text, finish_reason_or_None) tuples.
        finish_reason is 'stop' or 'length' on the final token; None otherwise.

        Identical external interface to server.py: _generate_tokens().
        """
        self._reset_stats()
        if seed is not None:
            np.random.seed(seed)
            try:
                mx.random.seed(seed)
            except Exception:
                pass

        eos_id    = getattr(self._ttok, "eos_token_id", None) or 151645
        input_ids = list(self._ttok.encode(prompt))
        stop_ids  = stop_ids or []

        if self._draft is None:
            # No draft model — plain auto-regressive
            yield from self._plain_stream(
                input_ids, max_tokens, temperature, top_p, stop_ids, eos_id
            )
            return

        yield from self._speculative_stream(
            input_ids, max_tokens, temperature, top_p, stop_ids, eos_id
        )

    # ── speculative dispatch ──────────────────────────────────────────────────

    def _speculative_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Dispatch to stateful (fast) or stateless (fallback) path."""
        if self._draft_cache is not None and self._target_cache is not None:
            yield from self._stateful_spec_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id
            )
        else:
            yield from self._stateless_spec_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id
            )

    # ── stateful speculative inner loop ──────────────────────────────────────

    def _stateful_spec_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """
        Speculative decoding with incremental KV caches.

        Each cycle costs:
          • Draft:  k × O(1) forward passes  (1 token each, cache grows)
          • Target: 1 × O(k) forward pass    (k tokens, cache rolled back then refilled)

        vs. the stateless path where both models re-scan the full growing context
        on every step — O(context_len²) total vs O(context_len) here.

        Cache consistency rule
        ----------------------
        After every accepted sequence of n tokens, both caches sit at
        ``base + n``.  A single forward of the ''final'' token (correction or
        bonus) advances both to ``base + n + 1``, ready for the next cycle.
        """
        self._reset_caches()

        # Prefill both models; keep their last-row logits as the starting
        # prediction for the first draft token.
        d_last = _prefill_cached(self._draft,  self._draft_cache,  ids)
        t_last = _prefill_cached(self._target, self._target_cache, ids)

        generated  = 0
        stop_buf: list[int] = []

        while generated < max_tokens:
            base = _cache_offset(self._target_cache)

            # ── Step 1: draft proposes k tokens (stateful, 1 token/pass) ──────
            draft_ids  : list[int]        = []
            draft_probs: list[np.ndarray] = []
            cur_d_logits = d_last

            for _ in range(self._k):
                probs = _softmax_np(cur_d_logits, temperature)
                probs = _top_p_filter(probs, top_p)
                tok   = _sample(probs)
                draft_ids.append(tok)
                draft_probs.append(probs)
                if tok == eos_id:
                    break
                cur_d_logits = _decode_step_cached(
                    self._draft, self._draft_cache, tok)

            self.proposed_total += len(draft_ids)
            self.steps += 1

            # ── Step 2: target verifies (1 pass, k tokens) ────────────────────
            # Roll target cache back to before draft tokens.
            _cache_set_offset(self._target_cache, base)

            # Forward all draft tokens through target.
            # target_fwd[j] = prediction AFTER draft_ids[j]
            #               = verification logit for draft_ids[j+1]  (j < k-1)
            #               = bonus token logit                       (j == k-1)
            target_fwd = _decode_multi_cached(
                self._target, self._target_cache, draft_ids)

            # Prepend t_last (prediction for draft_ids[0], from the prior round)
            # to form the complete verification window.
            # target_rows[i] predicts draft_ids[i]:
            #   target_rows[0] = t_last          (no extra forward needed)
            #   target_rows[1] = target_fwd[0]
            #   ...
            #   target_rows[k] = target_fwd[k-1]  (bonus)
            target_rows = np.concatenate(
                [t_last[np.newaxis], target_fwd], axis=0)  # (k+1, vocab)

            # ── Step 3: sequential accept / reject ────────────────────────────
            new_tokens: list[int] = []
            accepted = 0
            for i, (d_tok, d_probs) in enumerate(
                    zip(draft_ids, draft_probs, strict=False)):
                t_probs  = _softmax_np(target_rows[i], temperature)
                t_probs  = _top_p_filter(t_probs, top_p)
                p_target = float(t_probs[d_tok])
                p_draft  = float(d_probs[d_tok])

                if np.random.random() < min(1.0, p_target / max(p_draft, 1e-12)):
                    new_tokens.append(d_tok)
                    accepted += 1
                else:
                    adjusted = np.maximum(0.0, t_probs - d_probs)
                    s = adjusted.sum()
                    if s > 0:
                        adjusted /= s
                        fallback = _sample(adjusted)
                    else:
                        fallback = _greedy(target_rows[i])
                    new_tokens.append(fallback)
                    break

            self.accepted_total += accepted

            # ── Step 4: bonus token (all k accepted) ──────────────────────────
            if accepted == len(draft_ids):
                bonus_probs = _softmax_np(target_rows[len(draft_ids)], temperature)
                bonus_probs = _top_p_filter(bonus_probs, top_p)
                new_tokens.append(_sample(bonus_probs))

            # ── Step 5: advance caches to end of accepted sequence ────────────
            # n_acc tokens before the "final" token (correction or bonus).
            n_acc     = len(new_tokens) - 1
            final_tok = new_tokens[-1]
            # Trim both caches to base + n_acc, then run final_tok once
            # so both sit at base + n_acc + 1 for the next cycle.
            _cache_set_offset(self._draft_cache,  base + n_acc)
            _cache_set_offset(self._target_cache, base + n_acc)
            d_last = _decode_step_cached(self._draft,  self._draft_cache,  final_tok)
            t_last = _decode_step_cached(self._target, self._target_cache, final_tok)

            # ── Step 6: yield accepted + final token ──────────────────────────
            for tok in new_tokens:
                if tok == eos_id:
                    yield self._tok_text(tok), "stop"
                    return
                tok_text = self._tok_text(tok)
                generated += 1
                stop_buf.append(tok)
                for seq in stop_ids:
                    if stop_buf[-len(seq):] == seq:
                        yield tok_text, "stop"
                        return
                if len(stop_buf) > 64:
                    stop_buf = stop_buf[-64:]
                if generated >= max_tokens:
                    yield tok_text, "length"
                    return
                yield tok_text, None

        logger.debug(
            "stateful spec: %d steps, %.1f%% acceptance, %d tokens",
            self.steps, self.acceptance_rate * 100, generated,
        )

    # ── stateless speculative inner loop (fallback) ───────────────────────────

    def _stateless_spec_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """
        Core speculative decoding loop.

        Each step:
        1. Draft model proposes K tokens autoregressively.
        2. Target model verifies all K positions in one forward pass.
        3. Tokens are accepted/rejected per the speculative sampling rule.
        4. At least one bonus token (from target) is always appended.

        This gives the same distribution as sampling from the target alone.
        """
        generated     = 0
        stop_buf: list[int] = []
        context       = list(ids)

        while generated < max_tokens:
            # ── Step 1: draft K tokens ────────────────────────────────────
            draft_ids     : list[int]        = []
            draft_probs   : list[np.ndarray] = []

            for _ in range(self._k):
                logits = _get_logits(self._draft, context + draft_ids)
                probs  = _softmax_np(logits, temperature)
                probs  = _top_p_filter(probs, top_p)
                tok    = _sample(probs)
                draft_ids.append(tok)
                draft_probs.append(probs)
                if tok == eos_id:
                    break

            self.proposed_total += len(draft_ids)
            self.steps += 1

            # ── Step 2: target model verifies all K positions + 1 ────────
            # Feed context + all draft tokens; read last K+1 logit rows.
            full_seq     = context + draft_ids
            n_verify     = len(draft_ids) + 1   # K draft + 1 bonus
            target_rows  = _get_all_logits(self._target, full_seq, n_verify)
            # target_rows[i] is the logit for position (context_len - 1 + i)
            # which predicts token draft_ids[i].

            # ── Step 3: sequential accept/reject ─────────────────────────
            new_tokens: list[int] = []
            accepted   = 0
            for i, (d_tok, d_probs) in enumerate(zip(draft_ids, draft_probs, strict=False)):
                t_probs = _softmax_np(target_rows[i], temperature)
                t_probs = _top_p_filter(t_probs, top_p)

                p_target = float(t_probs[d_tok])
                p_draft  = float(d_probs[d_tok])

                accept_prob = min(1.0, p_target / max(p_draft, 1e-12))
                if np.random.random() < accept_prob:
                    new_tokens.append(d_tok)
                    accepted += 1
                else:
                    # Rejection: sample from the adjusted distribution
                    adjusted = np.maximum(0.0, t_probs - d_probs)
                    s = adjusted.sum()
                    if s > 0:
                        adjusted /= s
                        fallback_tok = _sample(adjusted)
                    else:
                        fallback_tok = _greedy(target_rows[i])
                    new_tokens.append(fallback_tok)
                    break

            self.accepted_total += accepted

            # ── Step 4: bonus token from target ──────────────────────────
            if accepted == len(draft_ids):
                # All K accepted — greedily sample from the K+1 position
                bonus_probs = _softmax_np(target_rows[len(draft_ids)], temperature)
                bonus_probs = _top_p_filter(bonus_probs, top_p)
                new_tokens.append(_sample(bonus_probs))

            # ── Yield accepted tokens ─────────────────────────────────────
            for tok in new_tokens:
                if tok == eos_id:
                    yield self._tok_text(tok), "stop"
                    return

                tok_text = self._tok_text(tok)
                generated += 1
                stop_buf.append(tok)
                context.append(tok)

                # Check stop sequences
                for seq in stop_ids:
                    if stop_buf[-len(seq):] == seq:
                        yield tok_text, "stop"
                        return
                if len(stop_buf) > 64:
                    stop_buf = stop_buf[-64:]

                if generated >= max_tokens:
                    yield tok_text, "length"
                    return
                yield tok_text, None

        logger.debug(
            "spec: %d steps, %.1f%% acceptance, %d tokens",
            self.steps,
            self.acceptance_rate * 100,
            generated,
        )

    def _tok_text(self, tok_id: int) -> str:
        try:
            return self._ttok.decode([tok_id])
        except Exception:
            return ""

    # ── plain stream (no draft model) ────────────────────────────────────────

    def _plain_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Dispatch to stateful (KV-cached) or stateless auto-regressive path."""
        if self._target_cache is not None:
            yield from self._stateful_plain_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id)
        else:
            yield from self._stateless_plain_stream(
                ids, max_tokens, temperature, top_p, stop_ids, eos_id)

    def _stateful_plain_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Plain auto-regressive with stateful KV cache — O(1) per token."""
        _cache_set_offset(self._target_cache, 0)
        last_logits = _prefill_cached(self._target, self._target_cache, ids)
        stop_buf: list[int] = []
        generated = 0

        for _ in range(max_tokens):
            if temperature == 0.0:
                tok = _greedy(last_logits)
            else:
                probs = _softmax_np(last_logits, temperature)
                probs = _top_p_filter(probs, top_p)
                tok   = _sample(probs)

            if tok == eos_id:
                yield self._tok_text(tok), "stop"
                return

            tok_text = self._tok_text(tok)
            generated += 1
            stop_buf.append(tok)

            for seq in stop_ids:
                if stop_buf[-len(seq):] == seq:
                    yield tok_text, "stop"
                    return
            if len(stop_buf) > 64:
                stop_buf = stop_buf[-64:]

            if generated >= max_tokens:
                yield tok_text, "length"
                return
            yield tok_text, None
            last_logits = _decode_step_cached(self._target, self._target_cache, tok)

        yield "", "stop"

    def _stateless_plain_stream(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_ids: list[list[int]],
        eos_id: int,
    ) -> Iterator[tuple[str, str | None]]:
        """Plain auto-regressive sampling — stateless fallback (O(n²) in context)."""
        context    = list(ids)
        stop_buf   : list[int] = []
        generated  = 0

        for _ in range(max_tokens):
            logits = _get_logits(self._target, context)
            if temperature == 0.0:
                tok = _greedy(logits)
            else:
                probs = _softmax_np(logits, temperature)
                probs = _top_p_filter(probs, top_p)
                tok   = _sample(probs)

            if tok == eos_id:
                yield self._tok_text(tok), "stop"
                return

            tok_text = self._tok_text(tok)
            context.append(tok)
            generated += 1
            stop_buf.append(tok)

            for seq in stop_ids:
                if stop_buf[-len(seq):] == seq:
                    yield tok_text, "stop"
                    return
            if len(stop_buf) > 64:
                stop_buf = stop_buf[-64:]

            if generated >= max_tokens:
                yield tok_text, "length"
                return
            yield tok_text, None

        yield "", "stop"


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test speculative decoding")
    ap.add_argument("--model-dir",        required=True)
    ap.add_argument("--compressed-dir",   required=True)
    ap.add_argument("--draft-model",      default="")
    ap.add_argument("--draft-compressed", default="")
    ap.add_argument("--prompt",           default="What is the capital of France?")
    ap.add_argument("--max-tokens",       type=int, default=200)
    ap.add_argument("--k",                type=int, default=_DEFAULT_K)
    ap.add_argument("--temperature",      type=float, default=0.0)
    args = ap.parse_args()

    from .compressed_loader import load_compressed_model

    print("Loading target model …")
    t0 = time.perf_counter()
    target_model, target_tok = load_compressed_model(
        model_dir=args.model_dir, npz_path=args.compressed_dir, verbose=True,
    )
    print(f"Target loaded in {time.perf_counter() - t0:.1f}s\n")

    draft_model = draft_tok = None
    if args.draft_model:
        print("Loading draft model …")
        t0 = time.perf_counter()
        draft_model, draft_tok = load_draft_model(
            args.draft_model, args.draft_compressed
        )
        print(f"Draft loaded in {time.perf_counter() - t0:.1f}s\n")

    gen = SpeculativeGenerator(
        target_model, target_tok,
        draft_model=draft_model, draft_tokenizer=draft_tok,
        k=args.k,
    )

    print(f"Prompt: {args.prompt!r}\n")
    print("─" * 60)

    t0 = time.perf_counter()
    n_tokens = 0
    for tok_text, finish_reason in gen.stream(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ):
        print(tok_text, end="", flush=True)
        n_tokens += 1
        if finish_reason:
            break

    elapsed = time.perf_counter() - t0
    print(f"\n\n─ {n_tokens} tokens in {elapsed:.2f}s  "
          f"({n_tokens / elapsed:.1f} tok/s)")
    if draft_model is not None:
        print(f"  acceptance rate: {gen.acceptance_rate * 100:.1f}%  "
              f"({gen.steps} steps)")
