"""
squish/speculative/ssd.py

Phase 4 — SSD: Speculative Speculative Decoding acceptance predictor.

Algorithm
─────────
Standard speculative decoding verifies *all k* draft tokens against the
target model.  When many draft tokens will be rejected (low acceptance rate),
this verification batch is wasteful — the target wastes compute on tokens it
is about to reject.

SSD addresses this with a lightweight pre-filter:

  1. For each draft token (id_i, prob_i), the predictor estimates the
     acceptance probability  p_acc_i ≈ P(target accepts id_i | context).

  2. The draft sequence is *truncated at the first token* whose predicted
     acceptance falls below ``config.threshold``.

  3. Only the truncated (shorter) draft sequence is sent to the target
     verify pass, reducing the average verify-batch length.

Predictor architecture
──────────────────────
The predictor is a tiny two-layer feedforward network operating on:

    features = concat([
        proj( gru_hidden_i ),   # (feature_dim,) — GRU's view of context
        [p_draft_i],            # (1,) — how confident the drafter is
    ])                          # total: (feature_dim + 1,)

    logit = W2 · ReLU( W1 · features + b1 ) + b2
    p_acc = σ(logit)

This is deliberately tiny: ``feature_dim = 32`` is typical.  The predictor
can be trained on accept/reject logs from a warming run, or randomly
initialised for a conservative pass-through (threshold=0.0 → never filters).

Design
──────
• Pure numpy — no MLX required at import time; fully unit-testable.
• The predictor is *optional*: setting ``config.threshold = 0.0`` or
  using ``SSDConfig(enabled=False)`` disables all filtering.
• ``filter_drafts()`` returns the original lists unchanged when disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from squish.speculative.redrafter import _sigmoid  # reuse numpy sigmoid


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SSDConfig:
    """
    Hyperparameters for the SSD acceptance predictor.

    Parameters
    ----------
    feature_dim : Projection dimension of the GRU hidden state.
    threshold   : Minimum predicted acceptance probability to retain a draft
                  token.  Tokens at or below this value truncate the draft
                  sequence.  0.0 = disable filtering (pass-through).
    enabled     : Master on/off flag.  When False, ``filter_drafts`` is a no-op.
    """
    feature_dim: int   = 32
    threshold:   float = 0.3
    enabled:     bool  = True

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be ≥ 1")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class SSDPredictor:
    """
    Two-layer feedforward acceptance predictor for Speculative Speculative
    Decoding.

    Parameters
    ----------
    config    : :class:`SSDConfig`
    proj_w    : (feature_dim, gru_hidden_dim) — projects GRU hidden → features
    proj_b    : (feature_dim,) bias, or None
    cls_w1    : (hidden1, feature_dim + 1) — first classifier layer
    cls_b1    : (hidden1,) bias
    cls_w2    : (1, hidden1) — output layer
    cls_b2    : (1,) bias
    """

    __slots__ = (
        "config",
        "_proj_w",    # (feature_dim, gru_hidden_dim)
        "_proj_b",    # (feature_dim,) or None
        "_cls_w1",    # (hidden1, feature_dim + 1)
        "_cls_b1",    # (hidden1,)
        "_cls_w2",    # (1, hidden1)
        "_cls_b2",    # (1,)
    )

    def __init__(
        self,
        config:   SSDConfig,
        proj_w:   np.ndarray,
        cls_w1:   np.ndarray,
        cls_b1:   np.ndarray,
        cls_w2:   np.ndarray,
        cls_b2:   np.ndarray,
        proj_b:   np.ndarray | None = None,
    ) -> None:
        self.config  = config
        self._proj_w = proj_w.astype(np.float32)
        self._proj_b = proj_b.astype(np.float32) if proj_b is not None else None
        self._cls_w1 = cls_w1.astype(np.float32)
        self._cls_b1 = cls_b1.astype(np.float32)
        self._cls_w2 = cls_w2.astype(np.float32)
        self._cls_b2 = cls_b2.astype(np.float32)

    # ── prediction ────────────────────────────────────────────────────────────

    def predict_acceptance(
        self,
        gru_hidden:   np.ndarray,  # (gru_hidden_dim,)
        draft_prob:   float,
    ) -> float:
        """
        Estimate the probability that the target model will accept the
        current draft token.

        Parameters
        ----------
        gru_hidden  : ReDrafter GRU hidden state at the draft step.
        draft_prob  : ``draft_probs[token_id]`` — probability the drafter
                      assigned to this token.

        Returns
        -------
        float in [0, 1] — estimated acceptance probability.
        """
        if not self.config.enabled or self.config.threshold == 0.0:
            return 1.0

        # Project GRU hidden → feature vector
        feat = self._proj_w @ gru_hidden
        if self._proj_b is not None:
            feat = feat + self._proj_b                                # (feature_dim,)

        # Concatenate with draft probability scalar
        x = np.concatenate([feat, np.array([draft_prob], dtype=np.float32)])  # (feature_dim+1,)

        # Two-layer MLP
        h1     = np.maximum(0.0, self._cls_w1 @ x + self._cls_b1)   # ReLU hidden
        logit  = (self._cls_w2 @ h1 + self._cls_b2)[0]               # scalar
        return float(_sigmoid(logit))

    def filter_drafts(
        self,
        gru_hiddens:  list[np.ndarray],  # per-step GRU hidden states (len == len(draft_ids))
        draft_ids:    list[int],
        draft_probs:  list[np.ndarray],  # per-step (vocab,) probability vectors
    ) -> tuple[list[int], list[np.ndarray]]:
        """
        Truncate the draft sequence at the first token whose predicted
        acceptance probability falls below ``config.threshold``.

        If trunction occurs at position 0 we still return at least 1 token
        (never return an empty draft — the caller always needs at least one
        token to round-trip through the verify pass).

        Parameters
        ----------
        gru_hiddens  : List of GRU hidden states, one per draft step.
        draft_ids    : Draft token IDs.
        draft_probs  : Per-token probability distributions (vocab,).

        Returns
        -------
        (filtered_ids, filtered_probs)  — possibly shorter than the inputs.
        """
        if (
            not self.config.enabled
            or self.config.threshold == 0.0
            or not draft_ids
        ):
            return draft_ids, draft_probs

        keep_up_to = len(draft_ids)  # default: keep all
        for i, (d_id, d_probs, h_g) in enumerate(
                zip(draft_ids, draft_probs, gru_hiddens, strict=False)):
            p_draft = float(d_probs[d_id])
            p_acc   = self.predict_acceptance(h_g, p_draft)
            if p_acc < self.config.threshold:
                keep_up_to = max(1, i)   # keep at least 1 token
                break

        return draft_ids[:keep_up_to], draft_probs[:keep_up_to]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save predictor weights to .npz."""
        arrays: dict[str, np.ndarray] = {
            "config_feature_dim": np.array(self.config.feature_dim),
            "config_threshold":   np.array(self.config.threshold),
            "proj_w": self._proj_w,
            "cls_w1": self._cls_w1,
            "cls_b1": self._cls_b1,
            "cls_w2": self._cls_w2,
            "cls_b2": self._cls_b2,
        }
        if self._proj_b is not None:
            arrays["proj_b"] = self._proj_b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "SSDPredictor":
        """Load predictor from .npz produced by :meth:`save`."""
        d      = np.load(path)
        config = SSDConfig(
            feature_dim = int(d["config_feature_dim"]),
            threshold   = float(d["config_threshold"]),
        )
        return cls(
            config  = config,
            proj_w  = d["proj_w"],
            proj_b  = d.get("proj_b"),
            cls_w1  = d["cls_w1"],
            cls_b1  = d["cls_b1"],
            cls_w2  = d["cls_w2"],
            cls_b2  = d["cls_b2"],
        )

    @classmethod
    def init_random(
        cls,
        gru_hidden_dim: int,
        feature_dim:    int             = 32,
        threshold:      float           = 0.3,
        rng:            np.random.Generator | None = None,
    ) -> "SSDPredictor":
        """
        Create a randomly-initialised predictor for tests / cold-start.

        In practice, the predictor should be trained on accept/reject logs
        from actual speculation runs.
        """
        rng    = rng or np.random.default_rng(0)
        cfg    = SSDConfig(feature_dim=feature_dim, threshold=threshold)
        hidden1 = feature_dim * 2

        proj_w = rng.standard_normal((feature_dim, gru_hidden_dim)).astype(np.float32) * 0.02
        proj_b = np.zeros(feature_dim, dtype=np.float32)

        in_dim = feature_dim + 1
        cls_w1 = rng.standard_normal((hidden1, in_dim)).astype(np.float32) * 0.02
        cls_b1 = np.zeros(hidden1, dtype=np.float32)
        cls_w2 = rng.standard_normal((1, hidden1)).astype(np.float32) * 0.02
        cls_b2 = np.zeros(1, dtype=np.float32)

        return cls(cfg, proj_w, cls_w1, cls_b1, cls_w2, cls_b2, proj_b=proj_b)
