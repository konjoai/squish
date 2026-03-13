"""
squish/copy_spec.py

CopySpec — Copy-and-Paste Speculative Decoding via Suffix Automaton.

Based on:
  "CopySpec: Improving Speculative Decoding by Identifying Copy-and-Paste
   Opportunities in LLM Generation"
  arXiv:2502.08923; github.com/dumitrescurazvan/CopySpec

Problem
-------
Standard speculative decoding methods (N-gram lookup, EAGLE-3) focus on
either the *input prompt* or a *trained draft model* as sources of draft
tokens.  CopySpec exploits a third source: the model's own *prior output*.

LLMs frequently reproduce earlier phrasing when generating structured or
repetitive text:
  - DevOps plans:  "Step 3: verify…  Step 4: verify…" (same continuation)
  - Code:          repeated function bodies with identical boilerplate
  - Emails:        recurring templates with shared sign-off language

CopySpec builds a suffix automaton over the tokens already generated.  When
the current context suffix matches an earlier position in the generation, the
continuation of that earlier occurrence is proposed as draft tokens — free of
any model call.

Method
------
A **suffix automaton** (also called DAWG — Directed Acyclic Word Graph) is a
finite automaton that accepts only the suffixes of a string.  After each
generated token, the automaton is extended incrementally in O(1) amortised
time.  A longest-suffix search in O(k) time (k = search depth) finds the
longest repeat of the last k tokens in the generation history.

If a match of length ≥ ``min_match_len`` is found, the continuation is read
from the history and returned as draft tokens.  The target model verifies them
as in standard speculative decoding.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **Synergy with EAGLE-3 / FR-Spec**: CopySpec is a *pre-filter*.  When it
  hits (repeat found), the EAGLE-3 draft is skipped entirely.  The cascade is:
  CopySpec → N-gram lookup → RASD → EAGLE-3.
- **Independence**: CopySpec operates on generation history only; it does not
  interact with KV cache management, quantization, or attention kernels.

Provides
--------
  CopySpecConfig     — tuning parameters.
  CopySpecDrafter    — incremental suffix automaton with draft recovery.
  CopySpecStats      — hit-rate and speedup counters.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CopySpecConfig",
    "CopySpecDrafter",
    "CopySpecStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CopySpecConfig:
    """Parameters for CopySpec.

    Parameters
    ----------
    min_match_len:
        Minimum number of tokens that must match before CopySpec proposes
        a continuation.  Short matches produce low-quality continuations.
    max_draft_len:
        Maximum number of draft tokens proposed per step.
    max_history_len:
        Maximum generation history retained in the suffix automaton.
        Older tokens are discarded to bound memory usage.
    search_context_len:
        Number of most-recent tokens used as the search query.
    """

    min_match_len: int = 3
    max_draft_len: int = 8
    max_history_len: int = 2048
    search_context_len: int = 16


# ---------------------------------------------------------------------------
# Suffix Automaton (DAWG)
# ---------------------------------------------------------------------------

class _SAMState:
    """Single state in the suffix automaton."""

    __slots__ = ("len", "link", "next", "first_endpos")

    def __init__(self, length: int = 0) -> None:
        self.len: int = length
        self.link: int = -1
        self.next: dict[int, int] = {}
        self.first_endpos: int = -1  # earliest end position of this state's suffixes


class _SuffixAutomaton:
    """Incremental suffix automaton for integer sequences.

    Supports:
    - ``extend(token)``  — O(1) amortised append
    - ``longest_match(query)``  — O(len(query)) longest-suffix prefix search
    """

    def __init__(self) -> None:
        self._states: list[_SAMState] = [_SAMState(0)]  # initial state at index 0
        self._last: int = 0
        self._history: list[int] = []  # tokens appended so far

    @property
    def history(self) -> list[int]:
        return self._history

    def extend(self, token: int) -> None:
        """Append one token to the automaton.  O(1) amortised."""
        cur = len(self._states)
        self._states.append(_SAMState(self._states[self._last].len + 1))
        self._states[cur].first_endpos = len(self._history)
        self._history.append(token)

        p = self._last
        while p != -1 and token not in self._states[p].next:
            self._states[p].next[token] = cur
            p = self._states[p].link

        if p == -1:
            self._states[cur].link = 0
        else:
            q = self._states[p].next[token]
            if self._states[p].len + 1 == self._states[q].len:
                self._states[cur].link = q
            else:
                clone = len(self._states)
                clone_state = _SAMState(self._states[p].len + 1)
                clone_state.link = self._states[q].link
                clone_state.next = dict(self._states[q].next)
                clone_state.first_endpos = self._states[q].first_endpos
                self._states.append(clone_state)
                while p != -1 and self._states[p].next.get(token) == q:
                    self._states[p].next[token] = clone
                    p = self._states[p].link
                self._states[q].link = clone
                self._states[cur].link = clone

        self._last = cur

    def longest_match(self, query: list[int]) -> tuple[int, int]:
        """Find the longest prefix of *query* that appears in the automaton.

        Returns
        -------
        (match_len, endpos):
            ``match_len`` — number of query tokens matched (0 if none).
            ``endpos``    — position in history where the match ends
                           (i.e. ``history[endpos - match_len + 1 : endpos + 1]``
                           equals ``query[:match_len]``).  -1 if no match.
        """
        if not query:
            return 0, -1

        cur = 0
        match_len = 0
        last_endpos = -1

        for token in query:
            if token in self._states[cur].next:
                cur = self._states[cur].next[token]
                match_len += 1
                last_endpos = self._states[cur].first_endpos
            else:
                break

        return match_len, last_endpos


# ---------------------------------------------------------------------------
# CopySpecDrafter
# ---------------------------------------------------------------------------

class CopySpecDrafter:
    """Incremental CopySpec drafter backed by a suffix automaton.

    Usage::

        drafter = CopySpecDrafter()
        for token in generated_so_far:
            drafter.add_token(token)

        drafts = drafter.draft(max_n=8)
        if drafts:
            # submit drafts to target model for verification
            ...
        else:
            # fall through to EAGLE-3
            ...
    """

    def __init__(self, config: CopySpecConfig | None = None) -> None:
        self._config = config or CopySpecConfig()
        self._sam = _SuffixAutomaton()
        self._n_tokens: int = 0

    @property
    def n_tokens(self) -> int:
        """Number of tokens added to the drafter so far."""
        return self._n_tokens

    @property
    def history(self) -> list[int]:
        """The full generation history (read-only view)."""
        return self._sam.history

    def add_token(self, token_id: int) -> None:
        """Append one generated token to the automaton.

        Trims the automaton when history exceeds ``max_history_len`` by
        rebuilding from the truncated suffix.
        """
        self._sam.extend(token_id)
        self._n_tokens += 1

        if len(self._sam.history) > self._config.max_history_len:
            self._rebuild_trimmed()

    def _rebuild_trimmed(self) -> None:
        """Rebuild the automaton from the last ``max_history_len`` tokens."""
        keep = self._sam.history[-self._config.max_history_len :]
        self._sam = _SuffixAutomaton()
        for tok in keep:
            self._sam.extend(tok)

    def draft(self, max_n: int | None = None) -> list[int] | None:
        """Propose up to *max_n* draft tokens via copy-and-paste.

        Searches the generation history for the longest match to the current
        context (last ``search_context_len`` tokens) and returns the
        continuation.

        Returns
        -------
        list of int
            Draft token IDs if a qualifying match was found.
        None
            If no match of sufficient length exists (fall through to
            EAGLE-3 / N-gram / RASD).
        """
        cfg = self._config
        n = min(max_n or cfg.max_draft_len, cfg.max_draft_len)
        history = self._sam.history

        if len(history) < cfg.min_match_len + 1:
            return None

        # Build query: last search_context_len tokens (excluding the latest,
        # which we want to extend)
        query_start = max(0, len(history) - cfg.search_context_len - 1)
        query = history[query_start:-1] if len(history) > 1 else []
        if not query:
            return None

        match_len, endpos = self._sam.longest_match(query)

        if match_len < cfg.min_match_len or endpos == -1:
            return None

        # Continuation starts one token after the match end in history
        cont_start = endpos + 1
        cont_end = min(cont_start + n, len(history))
        drafts = history[cont_start:cont_end]
        if not drafts:
            return None
        return drafts

    def reset(self) -> None:
        """Clear all generation history and reset the automaton."""
        self._sam = _SuffixAutomaton()
        self._n_tokens = 0


# ---------------------------------------------------------------------------
# CopySpecStats
# ---------------------------------------------------------------------------

@dataclass
class CopySpecStats:
    """Hit-rate and speedup counters for CopySpec.

    Attributes
    ----------
    draft_attempts:
        Total number of draft calls made.
    hits:
        Calls where CopySpec found a qualifying continuation.
    misses:
        Calls where CopySpec returned None (no match / too short).
    total_tokens_proposed:
        Total draft tokens proposed across all hits.
    total_tokens_accepted:
        Total draft tokens accepted by the target model.
    """

    draft_attempts: int = 0
    hits: int = 0
    misses: int = 0
    total_tokens_proposed: int = 0
    total_tokens_accepted: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of drafts that produced a CopySpec hit."""
        return self.hits / self.draft_attempts if self.draft_attempts else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed tokens accepted by the target."""
        return (
            self.total_tokens_accepted / self.total_tokens_proposed
            if self.total_tokens_proposed else 0.0
        )

    @property
    def tokens_per_hit(self) -> float:
        """Average tokens proposed per hit."""
        return self.total_tokens_proposed / self.hits if self.hits else 0.0

    def record_hit(self, n_proposed: int, n_accepted: int) -> None:
        self.draft_attempts += 1
        self.hits += 1
        self.total_tokens_proposed += n_proposed
        self.total_tokens_accepted += n_accepted

    def record_miss(self) -> None:
        self.draft_attempts += 1
        self.misses += 1

    def reset(self) -> None:
        self.draft_attempts = 0
        self.hits = 0
        self.misses = 0
        self.total_tokens_proposed = 0
        self.total_tokens_accepted = 0
