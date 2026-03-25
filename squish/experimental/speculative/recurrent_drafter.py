"""
squish/speculative/recurrent_drafter.py

RecurrentDrafter: GRU/LSTM Recurrent Speculative Drafter.

Reference
---------
Zhang et al. "Recurrent Drafter for Fast Speculative Decoding in Language
Models." Apple Research 2024.

Algorithm
---------
The Recurrent Drafter is a lightweight sequence model (GRU or LSTM, ~1 M
parameters) trained to mimic the target model's next-token distribution.  At
each speculative step:

  1. The recurrent state is updated with the last token embedding.
  2. The state is projected to vocabulary logits.
  3. Draft tokens are sampled greedily (or via temperature).
  4. When the target model verifies and accepts a prefix, the accepted tokens
     are used to update the recurrent state for the next round.

This implementation provides:
  * A self-contained NumPy GRU or LSTM cell.
  * No training — weights are initialised from scratch (enables unit testing
    without a pre-trained checkpoint; a real deployment would load saved
    weights via ``load_weights``).
  * Configurable depth, temperature, vocabulary size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RecurrentDrafterConfig:
    """Configuration for RecurrentDrafter."""

    cell_type: Literal["gru", "lstm"] = "gru"
    """Recurrent cell architecture."""

    hidden_size: int = 256
    """Recurrent hidden state dimensionality."""

    embed_size: int = 128
    """Token embedding dimension."""

    vocab_size: int = 32000
    """Vocabulary size."""

    draft_depth: int = 5
    """Number of tokens drafted per speculative step."""

    temperature: float = 0.0
    """Sampling temperature; 0.0 = greedy."""

    seed: int = 42
    """RNG seed for weight initialisation and sampling."""

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.embed_size < 1:
            raise ValueError("embed_size must be >= 1")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.draft_depth < 1:
            raise ValueError("draft_depth must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")


@dataclass
class RecurrentDrafterStats:
    """Runtime counters for RecurrentDrafter."""

    draft_steps: int = 0
    total_drafted: int = 0
    total_accepted: int = 0

    @property
    def mean_acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -20.0, 20.0))


class RecurrentDrafter:
    """GRU/LSTM-based speculative drafter.

    Usage
    -----
    ::

        drafter = RecurrentDrafter()
        # After a verification round:
        drafter.update_state(last_token_id)
        drafts = drafter.draft()
        # After target model verification:
        drafter.accept_feedback(accepted_token_ids)
    """

    def __init__(self, config: Optional[RecurrentDrafterConfig] = None) -> None:
        self.config = config or RecurrentDrafterConfig()
        self.stats = RecurrentDrafterStats()
        self._rng = np.random.default_rng(self.config.seed)
        self._init_weights()
        self._reset_state()

    # ------------------------------------------------------------------
    # Weight initialisation (random; load_weights() for real use)
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        cfg = self.config
        rng = self._rng
        h = cfg.hidden_size
        e = cfg.embed_size
        v = cfg.vocab_size

        # Token embedding
        self._embed = rng.standard_normal((v, e)).astype(np.float32) * 0.02

        # Output projection (hidden → vocab)
        self._W_out = rng.standard_normal((v, h)).astype(np.float32) * 0.02
        self._b_out = np.zeros(v, dtype=np.float32)

        if cfg.cell_type == "gru":
            # GRU: 3 gates (reset, update, new)
            scale = np.sqrt(1.0 / h)
            self._Wz = rng.standard_normal((h, e + h)).astype(np.float32) * scale
            self._bz = np.zeros(h, dtype=np.float32)
            self._Wr = rng.standard_normal((h, e + h)).astype(np.float32) * scale
            self._br = np.zeros(h, dtype=np.float32)
            self._Wn = rng.standard_normal((h, e + h)).astype(np.float32) * scale
            self._bn = np.zeros(h, dtype=np.float32)
        else:
            # LSTM: 4 gates (input, forget, cell, output)
            scale = np.sqrt(1.0 / h)
            self._WIFCO = rng.standard_normal((4 * h, e + h)).astype(np.float32) * scale
            self._bIFCO = np.zeros(4 * h, dtype=np.float32)

    def load_weights(self, weights: dict) -> None:
        """Load pre-trained weights from a dictionary.

        Expected keys depend on ``cell_type``.  Any extra keys are ignored.
        """
        for attr, key in [
            ("_embed", "embed"),
            ("_W_out", "W_out"),
            ("_b_out", "b_out"),
        ]:
            if key in weights:
                setattr(self, attr, np.asarray(weights[key], dtype=np.float32))

        if self.config.cell_type == "gru":
            for attr, key in [
                ("_Wz", "Wz"), ("_bz", "bz"),
                ("_Wr", "Wr"), ("_br", "br"),
                ("_Wn", "Wn"), ("_bn", "bn"),
            ]:
                if key in weights:
                    setattr(self, attr, np.asarray(weights[key], dtype=np.float32))
        else:
            for attr, key in [("_WIFCO", "WIFCO"), ("_bIFCO", "bIFCO")]:
                if key in weights:
                    setattr(self, attr, np.asarray(weights[key], dtype=np.float32))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        h = self.config.hidden_size
        self._h: np.ndarray = np.zeros(h, dtype=np.float32)
        self._c: np.ndarray = np.zeros(h, dtype=np.float32)  # LSTM cell state

    def update_state(self, token_id: int) -> None:
        """Advance the recurrent state with one token.

        Parameters
        ----------
        token_id:
            Integer token index.
        """
        cfg = self.config
        e_vec = self._embed[token_id % cfg.vocab_size]  # (embed_size,)
        xh = np.concatenate([e_vec, self._h])

        if cfg.cell_type == "gru":
            z = _sigmoid(self._Wz @ xh + self._bz)
            r = _sigmoid(self._Wr @ xh + self._br)
            xh_r = np.concatenate([e_vec, r * self._h])
            n = _tanh(self._Wn @ xh_r + self._bn)
            self._h = (1.0 - z) * n + z * self._h
        else:
            gates = self._WIFCO @ xh + self._bIFCO  # (4h,)
            h = cfg.hidden_size
            i_g = _sigmoid(gates[:h])
            f_g = _sigmoid(gates[h:2*h])
            c_g = _tanh(gates[2*h:3*h])
            o_g = _sigmoid(gates[3*h:])
            self._c = f_g * self._c + i_g * c_g
            self._h = o_g * _tanh(self._c)

    # ------------------------------------------------------------------
    # Drafting
    # ------------------------------------------------------------------

    def draft(self) -> List[int]:
        """Generate ``config.draft_depth`` draft token IDs from current state.

        Returns
        -------
        draft_tokens:
            List of ``draft_depth`` candidate token IDs.
        """
        self.stats.draft_steps += 1
        cfg = self.config
        drafts: List[int] = []
        h = self._h.copy()
        c = self._c.copy()

        for _ in range(cfg.draft_depth):
            logits = self._W_out @ h + self._b_out  # (vocab_size,)

            if cfg.temperature == 0.0:
                tok = int(np.argmax(logits))
            else:
                logits = logits / cfg.temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                tok = int(self._rng.choice(cfg.vocab_size, p=probs))

            drafts.append(tok)

            # Step state with drafted token
            e_vec = self._embed[tok % cfg.vocab_size]
            xh = np.concatenate([e_vec, h])
            if cfg.cell_type == "gru":
                z = _sigmoid(self._Wz @ xh + self._bz)
                r = _sigmoid(self._Wr @ xh + self._br)
                xh_r = np.concatenate([e_vec, r * h])
                n = _tanh(self._Wn @ xh_r + self._bn)
                h = (1.0 - z) * n + z * h
            else:
                gates = self._WIFCO @ xh + self._bIFCO
                hs = cfg.hidden_size
                i_g = _sigmoid(gates[:hs])
                f_g = _sigmoid(gates[hs:2*hs])
                c_g = _tanh(gates[2*hs:3*hs])
                o_g = _sigmoid(gates[3*hs:])
                c = f_g * c + i_g * c_g
                h = o_g * _tanh(c)

        self.stats.total_drafted += len(drafts)
        return drafts

    def accept_feedback(self, accepted_tokens: Sequence[int]) -> None:
        """Update state with verified accepted tokens.

        Parameters
        ----------
        accepted_tokens:
            Tokens accepted by the target model this round.
        """
        self.stats.total_accepted += len(accepted_tokens)
        for tok in accepted_tokens:
            self.update_state(tok)

    def reset(self) -> None:
        """Reset recurrent state between requests (weights preserved)."""
        self._reset_state()
        self.stats = RecurrentDrafterStats()
