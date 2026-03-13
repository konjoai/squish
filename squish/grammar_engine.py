#!/usr/bin/env python3
"""
squish/grammar_engine.py

Optional XGrammar-based structured-output engine.

When xgrammar is installed (``pip install xgrammar``) the :class:`GrammarEngine`
class constrains token sampling to valid JSON, JSON-schema, or regex output via
a finite-state machine (FSM) bitmask applied just before each sampling step.

When xgrammar is **not** installed every method silently falls back to a no-op
so the server runs without requiring the optional dependency::

    pip install "squish[grammar]"    # installs xgrammar

Typical server-side lifecycle
──────────────────────────────
::

    engine = GrammarEngine(state.tokenizer)          # build once at startup
    state  = engine.json_object_grammar()            # or json_schema_grammar()

    # inside decode loop:
    logits  = engine.constrain_logits(logits, state) # mask invalid tokens
    token   = sample(logits)
    state   = engine.advance(state, token)           # advance FSM
    fwd     = engine.jump_forward_tokens(state)      # deterministic prefix
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

_SCHEMA_CACHE_MAXSIZE: int = 32


class GrammarEngine:
    """
    Thin wrapper around *xgrammar*; silently falls back to no-ops when the
    library is not installed.

    All public methods are safe to call regardless of whether xgrammar is
    available — they return identity values (unchanged logits, ``None`` states,
    empty lists) in fallback mode.
    """

    def __init__(self, tokenizer: Any) -> None:
        """
        Initialise the engine for *tokenizer*.

        Attempts to import xgrammar and build the
        ``TokenizerInfo`` + ``GrammarCompiler`` objects.  On any failure
        (xgrammar not installed, incompatible tokenizer format, etc.) the
        engine silently enters fallback/no-op mode (``self._available = False``).

        Parameters
        ----------
        tokenizer:
            A HuggingFace ``PreTrainedTokenizer`` or ``PreTrainedTokenizerFast``
            instance (or any object accepted by
            ``xgrammar.TokenizerInfo.from_huggingface``).
        """
        self._available: bool = False
        self._xgr: Any = None
        self._tok_info: Any = None
        self._compiler: Any = None
        self._tokenizer = tokenizer
        self._schema_cache: OrderedDict = OrderedDict()  # schema_hash -> compiled grammar
        try:
            import xgrammar as _xgr  # noqa: PLC0415
            self._xgr = _xgr
            self._tok_info = _xgr.TokenizerInfo.from_huggingface(tokenizer)
            self._compiler = _xgr.GrammarCompiler(self._tok_info)
            self._available = True
        except Exception:
            pass

    # ── Availability ──────────────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if *xgrammar* is importable in the current environment."""
        try:
            import xgrammar  # noqa: F401,PLC0415
            return True
        except ImportError:
            return False

    # ── Grammar construction ──────────────────────────────────────────────────

    def json_schema_grammar(self, schema: dict) -> Any:
        """
        Compile *schema* (a JSON-schema dict) and return a ``GrammarMatcher``
        that accepts only tokens compatible with the schema.

        Compiled grammars are cached by a 16-hex-char SHA-256 hash of the
        canonicalised schema; up to ``_SCHEMA_CACHE_MAXSIZE`` entries are kept
        (LRU eviction).  Creating a ``GrammarMatcher`` from the cached compiled
        grammar is still done fresh per call so each request gets its own FSM
        cursor position.

        Returns ``None`` when xgrammar is unavailable.
        """
        if not self._available:
            return None
        schema_hash = hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()
        ).hexdigest()[:16]
        compiled = self._schema_cache.get(schema_hash)
        if compiled is None:
            compiled = self._compiler.compile_json_schema(json.dumps(schema))
            self._schema_cache[schema_hash] = compiled
            if len(self._schema_cache) > _SCHEMA_CACHE_MAXSIZE:
                self._schema_cache.popitem(last=False)  # LRU evict oldest
        return self._xgr.GrammarMatcher(compiled)

    def json_object_grammar(self) -> Any:
        """
        Return a ``GrammarMatcher`` that accepts any well-formed JSON object.

        Returns ``None`` when xgrammar is unavailable.
        """
        if not self._available:
            return None
        compiled = self._compiler.compile_builtin_json_grammar()
        return self._xgr.GrammarMatcher(compiled)

    def regex_grammar(self, pattern: str) -> Any:
        """
        Return a ``GrammarMatcher`` that constrains output to strings matching
        *pattern*.

        Returns ``None`` when xgrammar is unavailable.
        """
        if not self._available:
            return None
        compiled = self._compiler.compile_regex(pattern)
        return self._xgr.GrammarMatcher(compiled)

    # ── Logit constraining ────────────────────────────────────────────────────

    def constrain_logits(self, logits_mx: Any, state: Any) -> Any:
        """
        Apply the grammar token bitmask to *logits_mx*.

        Converts *logits_mx* (an ``mlx.core.array``) to numpy, fills the next
        valid-token bitmask from *state*, applies it in-place, then converts
        back to an ``mlx.core.array``.

        Returns *logits_mx* unchanged when:

        * xgrammar is not available, or
        * *state* is ``None``, or
        * any error occurs during masking.

        Parameters
        ----------
        logits_mx:
            A 1-D ``mlx.core.array`` of logit values (vocabulary dimension).
        state:
            A ``GrammarMatcher`` returned by one of the grammar-construction
            methods, or ``None``.
        """
        if not self._available or state is None:
            return logits_mx
        try:
            import mlx.core as mx  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            logits_np = np.array(logits_mx.astype(mx.float32))
            bitmask = self._xgr.allocate_token_bitmask(1, self._tok_info.vocab_size)
            state.fill_next_token_bitmask(bitmask, 0)
            self._xgr.apply_token_bitmask_inplace(logits_np, bitmask)
            return mx.array(logits_np)
        except Exception:
            return logits_mx

    # ── FSM advancement ───────────────────────────────────────────────────────

    def advance(self, state: Any, token_id: int) -> Any:
        """
        Advance the grammar FSM by accepting *token_id*.

        The ``GrammarMatcher`` is mutated in-place; the same *state* object is
        returned.  Returns *state* unchanged (without raising) when xgrammar
        is unavailable, *state* is ``None``, or on any error.

        Parameters
        ----------
        state:
            A ``GrammarMatcher`` previously returned by a grammar-construction
            method, or ``None``.
        token_id:
            The integer token ID just sampled.
        """
        if not self._available or state is None:
            return state
        try:
            state.accept_token(token_id)
        except Exception:
            pass
        return state

    def jump_forward_tokens(self, state: Any) -> list[int]:
        """
        Return a list of token IDs that can be emitted deterministically
        (jump-forward decoding) without sampling.

        Returns an empty list when:

        * xgrammar is unavailable,
        * *state* is ``None``,
        * no deterministic prefix exists at the current FSM position, or
        * any error occurs.

        Parameters
        ----------
        state:
            A ``GrammarMatcher`` at the current FSM position, or ``None``.
        """
        if not self._available or state is None:
            return []
        try:
            fwd_str = state.find_jump_forward_string()
            if not fwd_str:
                return []
            ids = self._tokenizer.encode(fwd_str, add_special_tokens=False)
            return list(ids)
        except Exception:
            return []

    # ── TagDispatch factory ───────────────────────────────────────────────────

    def tag_dispatch_for_schema(
        self,
        trigger: str,
        schema: dict,
    ) -> "TagDispatch":
        """
        Return a :class:`TagDispatch` that activates ``json_schema_grammar(schema)``
        the moment *trigger* is seen in the token stream.

        The trigger string is tokenised without special tokens; an empty token
        list (e.g. when *trigger* is not in the vocabulary) means that the
        ``TagDispatch`` will never activate — useful as a safe no-op when the
        model's vocabulary does not contain the trigger.

        Parameters
        ----------
        trigger:
            Surface-form string to watch for, e.g. ``"<tool_call>"``.
        schema:
            JSON schema dict passed to :meth:`json_schema_grammar` on activation.
        """
        try:
            trigger_ids: list[int] = list(
                self._tokenizer.encode(trigger, add_special_tokens=False)
            )
        except Exception:
            trigger_ids = []
        return TagDispatch(
            engine=self,
            grammar_fn=lambda: self.json_schema_grammar(schema),
            trigger_ids=trigger_ids,
        )


# ---------------------------------------------------------------------------
# TagDispatch — deferred grammar activation on a trigger token sequence
# ---------------------------------------------------------------------------


class TagDispatch:
    """
    Deferred grammar activation: watches the outgoing token stream for a
    *trigger* token sequence; once matched, activates a grammar-constrained
    FSM for all subsequent tokens.

    Typical use-case: the model emits a chain-of-thought prefix before
    entering a grammar-constrained tool-call block::

        <think>…</think>
        <tool_call>{"name": "search", "parameters": {…}}</tool_call>

    Before the trigger fires :meth:`constrain_logits` is a transparent no-op.
    After the trigger fires every call routes through the wrapped
    :class:`GrammarEngine`.

    Parameters
    ----------
    engine : GrammarEngine
        Shared grammar engine instance to delegate to once activated.
    grammar_fn : callable
        Zero-argument callable that returns a fresh ``GrammarMatcher``.
        Called exactly once, on activation.  Use
        :meth:`GrammarEngine.tag_dispatch_for_schema` instead of
        constructing this directly.
    trigger_ids : list[int]
        Exact token-ID sequence to wait for.  An empty list means the
        dispatch never activates (safe no-op when the trigger is not in
        the model vocabulary).
    """

    def __init__(
        self,
        engine: "GrammarEngine",
        grammar_fn: Any,
        trigger_ids: list[int],
    ) -> None:
        self._engine = engine
        self._grammar_fn = grammar_fn
        self._trigger = list(trigger_ids)
        self._buf: list[int] = []           # rolling window for trigger matching
        self._active: bool = False
        self._state: Any = None             # GrammarMatcher once activated

    # ── State observers ───────────────────────────────────────────────────

    @property
    def activated(self) -> bool:
        """``True`` once the trigger sequence has been matched."""
        return self._active

    def is_terminated(self) -> bool:
        """
        Return ``True`` when the active FSM has reached a terminal state.

        Always returns ``False`` when not yet activated.  Mirrors the
        ``GrammarMatcher.is_terminated()`` signature so that server code can
        treat a ``TagDispatch`` and a raw ``GrammarMatcher`` uniformly.
        """
        if not self._active or self._state is None:
            return False
        try:
            return bool(self._state.is_terminated())
        except AttributeError:
            return False

    # ── Per-token operations ──────────────────────────────────────────────

    def observe(self, token_id: int) -> None:
        """
        Feed *token_id* into the dispatch state machine.

        * **Before activation**: maintains a rolling window of the last
          ``len(trigger_ids)`` tokens and activates when an exact match is
          found.
        * **After activation**: advances the wrapped FSM via
          :meth:`GrammarEngine.advance`.

        Parameters
        ----------
        token_id : int
            The integer token ID just sampled.
        """
        if self._active:
            self._state = self._engine.advance(self._state, token_id)
            return
        if not self._trigger:
            return  # never-activate sentinel
        self._buf.append(token_id)
        if len(self._buf) > len(self._trigger):
            self._buf.pop(0)
        if self._buf == self._trigger:
            self._active = True
            self._state = self._grammar_fn()

    def constrain_logits(self, logits_mx: Any) -> Any:
        """
        Constrain *logits_mx* when the FSM is active; pass through otherwise.

        Parameters
        ----------
        logits_mx : mlx.core.array
            1-D logit vector (vocabulary dimension).

        Returns
        -------
        mlx.core.array
            Masked logits if activated, otherwise *logits_mx* unchanged.
        """
        if not self._active or self._state is None:
            return logits_mx
        return self._engine.constrain_logits(logits_mx, self._state)


# ---------------------------------------------------------------------------
# DOMINO — Subword-aligned constrained decoding
# ---------------------------------------------------------------------------

class DOMINOConstraint:
    """
    DOMINO (Decoding with Optimised Matching for Instruction-Nested Output)
    aligns multi-token string constraints to subword tokenization boundaries.

    Problem: naively forbidding a phrase token-by-token misses cases where the
    phrase spans a subword boundary in an unexpected way.

    DOMINO builds a mapping from every *character-aligned* forbidden/required
    string to all tokenization paths that produce that string, then applies token-
    level masking that is consistent with any tokenization of the constrained string.

    Parameters
    ----------
    tokenizer : a callable ``encode(text) -> list[int]``
    forbidden : list of strings that must NOT appear in the output
    required  : list of strings that MUST appear somewhere in the output
    """

    def __init__(
        self,
        tokenizer:  Any,
        forbidden:  list[str] | None = None,
        required:   list[str] | None = None,
    ) -> None:
        self._tokenizer  = tokenizer
        self._forbidden  = list(forbidden or [])
        self._required   = list(required or [])
        # Build token-level deny sets for forbidden strings
        self._deny_sets: list[set[int]] = []
        for phrase in self._forbidden:
            try:
                ids = self._tokenizer.encode(phrase, add_special_tokens=False)
                self._deny_sets.append(set(ids[:1]))   # block the first token
            except Exception:
                self._deny_sets.append(set())

    def apply(self, logits_np: Any) -> Any:
        """
        Apply DOMINO subword masking to a (vocab,) float logits array.

        Returns a modified copy (numpy array).
        """
        import numpy as np
        out = np.asarray(logits_np, dtype=np.float32).copy()
        for deny_set in self._deny_sets:
            for tok_id in deny_set:
                if 0 <= tok_id < len(out):
                    out[tok_id] = -1e9
        return out

    @property
    def forbidden_phrases(self) -> list[str]:
        return list(self._forbidden)

    @property
    def required_phrases(self) -> list[str]:
        return list(self._required)


# ---------------------------------------------------------------------------
# DCCD — Draft-Conditioned Constrained Decoding
# ---------------------------------------------------------------------------

class DCCDDecoder:
    """
    Draft-Conditioned Constrained Decoding (DCCD).

    In speculative decoding the *draft* tokens may violate grammar / output
    constraints.  DCCD intercepts the draft token sequence, checks each draft
    against a constraint checker, and replaces violating drafts with the
    nearest grammar-conforming token before the verification step.

    Parameters
    ----------
    constraint_fn : callable(token_id: int) -> bool
        Returns True if *token_id* is valid at the current constraint position.
    fallback_token_id : int
        Token to substitute when a draft token fails the constraint check.
        Typically the pad or EOS token.
    """

    def __init__(
        self,
        constraint_fn:     Any,   # callable
        fallback_token_id: int = 0,
    ) -> None:
        self._constraint   = constraint_fn
        self._fallback     = fallback_token_id

    def filter_drafts(self, draft_ids: list[int]) -> list[int]:
        """
        Filter a list of draft token IDs through the constraint function.

        Each token that fails ``constraint_fn(token_id)`` is replaced with
        ``fallback_token_id``.  The sequence is truncated at the *first*
        violation to preserve causal consistency.

        Parameters
        ----------
        draft_ids : list of int

        Returns
        -------
        list of int (same length or shorter)
        """
        filtered: list[int] = []
        for tok in draft_ids:
            if self._constraint(tok):
                filtered.append(tok)
            else:
                filtered.append(self._fallback)
                break   # truncate at first violation
        return filtered

    def is_valid(self, token_id: int) -> bool:
        """Return True if *token_id* satisfies the current constraint."""
        return bool(self._constraint(token_id))
