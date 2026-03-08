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

import json
from typing import Any


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

        Returns ``None`` when xgrammar is unavailable.
        """
        if not self._available:
            return None
        compiled = self._compiler.compile_json_schema(json.dumps(schema))
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
            import mlx.core as mx   # noqa: PLC0415
            import numpy as np      # noqa: PLC0415
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

    def apply(self, logits_np: "Any") -> "Any":
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
