# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""EAGLE3 — Feature-level draft head for speculative decoding.

EAGLE-3 (Li et al., 2025) predicts the next hidden-state features directly
(rather than tokens) using a lightweight 1-layer transformer.  The feature
prediction has 3.5× higher acceptance rate than token-prediction drafts because
features capture more semantic context.  EAGLE-3 improvements over EAGLE-2:
joint feature+token training and lookahead feature fusion.

Reference:
    Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language
    Models via Training-Time Test", arXiv 2025.
    https://arxiv.org/abs/2503.01840

Usage example::

    import numpy as np
    from squish.speculative.eagle3 import Eagle3Config, Eagle3Decoder

    config = Eagle3Config(hidden_dim=4096, vocab_size=32000, max_draft_len=5)
    decoder = Eagle3Decoder(config)

    rng = np.random.default_rng(7)
    hidden = rng.standard_normal(4096).astype(np.float32)

    steps = decoder.draft_step(hidden, n_steps=5)
    print(f"Draft steps: {len(steps)}")

    draft_tokens = [int(np.argmax(logits)) for _, logits in steps]
    accepted, bonus = decoder.verify_step(draft_tokens, hidden)
    print(f"Accepted: {accepted}, bonus token: {bonus}")
    print(f"Session acceptance rate: {decoder.acceptance_rate:.3f}")
"""

from __future__ import annotations

__all__ = [
    "Eagle3Config",
    "Eagle3DraftHead",
    "Eagle3CompressedDraftHead",
    "Eagle3Decoder",
    "Eagle3Stats",
]

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from squish.quant.hqq import HQQConfig, HQQQuantizer, HQQTensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Eagle3Config:
    """Configuration for the EAGLE-3 feature-level draft head.

    Attributes:
        hidden_dim: Hidden-state dimensionality of the target LLM.
        vocab_size: Target vocabulary size.
        draft_layers: Number of lightweight transformer layers in the draft
            head (1 in the original paper).
        max_draft_len: Maximum number of speculative tokens per draft step.
        acceptance_threshold: Cosine-similarity threshold for feature-level
            acceptance in :meth:`Eagle3Decoder.verify_step`.
        feature_dim: Dimensionality of the draft feature space.  Defaults to
            *hidden_dim* when ``None``.
    """

    hidden_dim: int = 4096
    vocab_size: int = 32000
    draft_layers: int = 1
    max_draft_len: int = 5
    acceptance_threshold: float = 0.7
    feature_dim: int | None = None
    draft_head_bits: int = 0
    """Bit-width for draft head weight compression via HQQ.  0 = disabled."""

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {self.hidden_dim}"
            )
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be a positive integer, got {self.vocab_size}"
            )
        if self.draft_layers <= 0:
            raise ValueError(
                f"draft_layers must be a positive integer, got {self.draft_layers}"
            )
        if self.max_draft_len <= 0:
            raise ValueError(
                f"max_draft_len must be a positive integer, got {self.max_draft_len}"
            )
        if not (0.0 < self.acceptance_threshold <= 1.0):
            raise ValueError(
                f"acceptance_threshold must be in (0.0, 1.0], "
                f"got {self.acceptance_threshold}"
            )
        # Set feature_dim to hidden_dim if not provided
        if self.feature_dim is None:
            object.__setattr__(self, "feature_dim", self.hidden_dim)
        elif self.feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be a positive integer, got {self.feature_dim}"
            )
        if self.draft_head_bits not in (0, 2, 3, 4, 8):
            raise ValueError(
                f"draft_head_bits must be 0 (disabled), 2, 3, 4, or 8; "
                f"got {self.draft_head_bits}"
            )


# ---------------------------------------------------------------------------
# Draft head
# ---------------------------------------------------------------------------

class Eagle3DraftHead:
    """Lightweight draft head that predicts hidden-state features.

    Architecture: two linear projections with a tanh non-linearity between
    them, following the EAGLE-3 design of predicting features rather than
    logits directly.
    """

    def __init__(self, config: Eagle3Config) -> None:
        self._config = config
        feature_dim: int = config.feature_dim  # type: ignore[assignment]
        rng = np.random.default_rng()

        # feature_proj: hidden_dim → feature_dim
        scale_fp = 1.0 / np.sqrt(config.hidden_dim)
        self._feature_proj = rng.normal(
            0.0, scale_fp, (config.hidden_dim, feature_dim)
        ).astype(np.float32)

        # output_proj: feature_dim → vocab_size
        scale_op = 1.0 / np.sqrt(feature_dim)
        self._output_proj = rng.normal(
            0.0, scale_op, (feature_dim, config.vocab_size)
        ).astype(np.float32)

    def compress(self, bits: int = 3) -> "Eagle3CompressedDraftHead":
        """Return an INT-compressed copy of this draft head via HQQ.

        Args:
            bits: Target bit-width (2, 3, 4, or 8).  Default is 3 (INT3).

        Returns:
            :class:`Eagle3CompressedDraftHead` storing :class:`HQQTensor`
            objects for both projection matrices.
        """
        cfg = HQQConfig(bits=bits, group_size=64)
        quant = HQQQuantizer(cfg)
        fp_k = quant.encode(self._feature_proj)
        op_k = quant.encode(self._output_proj)
        return Eagle3CompressedDraftHead(
            config=self._config,
            feature_proj_tensor=fp_k,
            output_proj_tensor=op_k,
        )

    def predict_features(self, hidden: np.ndarray) -> np.ndarray:
        """Project hidden state to draft feature space via tanh activation.

        Args:
            hidden: Float32 vector of shape ``(hidden_dim,)``.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        return np.tanh(hidden @ self._feature_proj)

    def predict_tokens(self, features: np.ndarray) -> np.ndarray:
        """Project feature vector to vocabulary logits.

        Args:
            features: Float32 vector of shape ``(feature_dim,)``.

        Returns:
            Logit vector of shape ``(vocab_size,)``.
        """
        return features @ self._output_proj

    def forward(
        self, hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run full feature prediction and token logit computation.

        Args:
            hidden: Float32 vector of shape ``(hidden_dim,)`` or
                ``(feature_dim,)`` for subsequent draft steps.

        Returns:
            ``(draft_features, token_logits)`` — the feature vector and the
            unnormalized vocabulary logits.
        """
        features = self.predict_features(hidden)
        logits = self.predict_tokens(features)
        return features, logits


# ---------------------------------------------------------------------------
# Compressed draft head (HQQ INT2/INT3/INT4)
# ---------------------------------------------------------------------------


class Eagle3CompressedDraftHead:
    """HQQ-quantized draft head that decodes weights on each forward pass.

    Create via :meth:`Eagle3DraftHead.compress` rather than constructing
    directly.

    Memory use of the two projection matrices is dramatically reduced:
      * float32 baseline:  ``hidden_dim × feature_dim + feature_dim × vocab_size`` float32 bytes
      * INT3 (bits=3):     approximately ``(bits/32)`` × baseline ≈ 9% of float32

    Args:
        config: The :class:`Eagle3Config` this head belongs to.
        feature_proj_tensor: :class:`HQQTensor` for the ``(hidden_dim, feature_dim)`` projection.
        output_proj_tensor: :class:`HQQTensor` for the ``(feature_dim, vocab_size)`` projection.
    """

    def __init__(
        self,
        config: "Eagle3Config",
        feature_proj_tensor: HQQTensor,
        output_proj_tensor: HQQTensor,
    ) -> None:
        self._config = config
        self._fp_tensor = feature_proj_tensor
        self._op_tensor = output_proj_tensor
        self._quant = HQQQuantizer(feature_proj_tensor.config)

    def predict_features(self, hidden: np.ndarray) -> np.ndarray:
        """Project hidden state to draft feature space (decode + matmul)."""
        fp = self._quant.decode(self._fp_tensor)
        return np.tanh(hidden @ fp)

    def predict_tokens(self, features: np.ndarray) -> np.ndarray:
        """Project feature vector to vocabulary logits (decode + matmul)."""
        op = self._quant.decode(self._op_tensor)
        return features @ op

    def forward(
        self, hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run full feature prediction and token logit computation.

        Both projection matrices are decoded from their HQQ representation
        on each call.  For batch or streaming use, cache the decoded weights
        externally and call the projection steps directly.
        """
        features = self.predict_features(hidden)
        logits = self.predict_tokens(features)
        return features, logits

    @property
    def memory_bytes(self) -> int:
        """Approximate compressed storage size in bytes (codes + scales + zeros)."""
        return self._fp_tensor.nbytes() + self._op_tensor.nbytes()

    @property
    def bits(self) -> int:
        """Bit-width used for compression."""
        return self._fp_tensor.config.bits

    def __repr__(self) -> str:
        mb = self.memory_bytes / (1024 ** 2)
        return (
            f"Eagle3CompressedDraftHead(bits={self.bits}, "
            f"memory={mb:.1f} MiB)"
        )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Eagle3Decoder:
    """EAGLE-3 speculative decoder that drafts via iterated feature prediction.

    The decoder runs *n_steps* draft steps starting from the last verified
    hidden state.  Each step reuses the feature output of the previous step
    as the input for the next, forming a lightweight autoregressive chain
    without calling the full target model.
    """

    def __init__(self, config: Eagle3Config) -> None:
        self._config = config
        self._head = Eagle3DraftHead(config)
        self._last_draft_features: np.ndarray | None = None
        self._n_accepted: int = 0
        self._n_total: int = 0
        self._feature_sim_sum: float = 0.0

    # ------------------------------------------------------------------
    # Draft
    # ------------------------------------------------------------------

    def draft_step(
        self,
        hidden: np.ndarray,
        n_steps: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run *n_steps* speculative draft iterations.

        The first step uses *hidden* as input; subsequent steps reuse the
        feature output of the previous step.  This works correctly when
        ``feature_dim == hidden_dim`` (the default); when they differ the
        behaviour is valid as long as the projection dimensions align.

        Args:
            hidden: Starting hidden state, shape ``(hidden_dim,)``.
            n_steps: Number of draft steps; capped at ``config.max_draft_len``.

        Returns:
            List of ``(features, token_logits)`` tuples, one per draft step.
        """
        n_steps = min(n_steps, self._config.max_draft_len)
        if n_steps <= 0:
            return []

        results: list[tuple[np.ndarray, np.ndarray]] = []
        current: np.ndarray = hidden.copy()
        for _ in range(n_steps):
            features, logits = self._head.forward(current)
            results.append((features, logits))
            current = features  # reuse features as next input

        # Cache the most recent draft features for verify_step
        self._last_draft_features = results[-1][0]
        return results

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify_step(
        self,
        draft_tokens: list[int],
        target_hidden: np.ndarray,
    ) -> tuple[bool, int]:
        """Verify the draft against the target model's hidden state.

        Computes target features from *target_hidden* and measures cosine
        similarity against the cached draft features.  The draft is accepted
        if the similarity exceeds ``config.acceptance_threshold``.

        The *bonus_token* is always the argmax of the target logits,
        providing at least one accepted token even when the feature-level
        check fails.

        Args:
            draft_tokens: List of draft token IDs (used for auditing; the
                actual acceptance is feature-based, not token-based).
            target_hidden: Target model hidden-state vector of shape
                ``(hidden_dim,)``.

        Returns:
            ``(accepted, bonus_token)`` where *accepted* is ``True`` when
            cosine similarity exceeds the threshold and *bonus_token* is the
            greedy token from the target projection.
        """
        target_features = self._head.predict_features(target_hidden)
        target_logits = self._head.predict_tokens(target_features)
        bonus_token = int(np.argmax(target_logits))

        self._n_total += 1

        if self._last_draft_features is None:
            # No draft to compare against — conservative rejection.
            self._feature_sim_sum += 0.0
            return False, bonus_token

        draft_f = self._last_draft_features
        norm_draft = float(np.linalg.norm(draft_f))
        norm_target = float(np.linalg.norm(target_features))

        if norm_draft < 1e-8 or norm_target < 1e-8:
            sim = 0.0
        else:
            sim = float(
                np.dot(draft_f, target_features) / (norm_draft * norm_target)
            )

        self._feature_sim_sum += sim
        accepted = sim > self._config.acceptance_threshold
        if accepted:
            self._n_accepted += 1

        return accepted, bonus_token

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_accepted(self) -> int:
        """Cumulative number of accepted verify_step calls."""
        return self._n_accepted

    @property
    def n_total(self) -> int:
        """Total number of verify_step calls made."""
        return self._n_total

    @property
    def acceptance_rate(self) -> float:
        """Fraction of verify_step calls where the draft was accepted."""
        if self._n_total == 0:
            return 0.0
        return self._n_accepted / self._n_total

    def get_stats(self) -> Eagle3Stats:
        """Return a snapshot of current decoding statistics."""
        return Eagle3Stats(
            total_draft_steps=self._n_total,
            total_accepted=self._n_accepted,
            feature_sim_sum=self._feature_sim_sum,
            n_verifications=self._n_total,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class Eagle3Stats:
    """Aggregate statistics for an EAGLE-3 decoding session.

    Attributes:
        total_draft_steps: Total number of draft steps produced.
        total_accepted: Number of verify_step calls where draft was accepted.
        feature_sim_sum: Cumulative cosine similarity across all verifications.
        n_verifications: Total number of :meth:`Eagle3Decoder.verify_step`
            calls.
    """

    total_draft_steps: int = 0
    total_accepted: int = 0
    feature_sim_sum: float = 0.0
    n_verifications: int = 0

    @property
    def mean_feature_similarity(self) -> float:
        """Mean cosine similarity between draft and target features."""
        if self.n_verifications == 0:
            return 0.0
        return self.feature_sim_sum / self.n_verifications

    @property
    def acceptance_rate(self) -> float:
        """Fraction of verification calls that accepted the draft."""
        if self.n_verifications == 0:
            return 0.0
        return self.total_accepted / self.n_verifications
