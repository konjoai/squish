"""
squish/c2t.py

C2T — Classifier-based Candidate Tree Construction for Speculative Decoding.

Based on:
  "Classifier-based Candidate Trees for Speculative Decoding"
  arXiv:2502.02227

Problem
-------
Standard speculative decoding uses a *uniform* tree structure: the draft model
proposes the same number of candidates at every depth level.  This wastes
draft budget on token positions where the draft model is already very
confident (almost no improvement from a second branch), while underinvesting
at uncertain positions (where a second branch would often catch the correct
token).

C2T uses a lightweight binary classifier that observes the draft model's
*current hidden state and top-2 logit gap* to predict whether a position is
"wide" (classifier outputs 1 → allocate 2 branches) or "narrow" (0 → keep
1 branch).  This adapts the tree shape per position at decode time.

Method
------
1. **C2TClassifier** — binary logistic regressor over a feature vector derived
   from the draft model's output:
   - Top-1 logit (confidence signal)
   - Top-2 logit gap (certainty gap)
   - Entropy of the softmax distribution

   ``classify(features) → int``  — 1 = wide, 0 = narrow.

2. **C2TFeatures** — helper that computes the (3,) feature vector from a
   ``(vocab_size,)`` logit array.

3. **AdaptiveTreeBuilder** — uses the classifier to build a variable-width
   draft tree.  ``build(draft_fn, context, depth)`` calls the draft model
   repeatedly, branching at uncertain positions.

4. **C2TTrainer** — online online logistic regression update: after each
   decode step, updates the classifier based on whether branching improved
   acceptance at that position.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **Synergy with EAGLE-3**: C2T sits on top of EAGLE-3's generate call,
  deciding *how many* candidates to request per depth.
- **Synergy with XGrammar**: structural token positions (forced by grammar)
  are always narrow — classifier output is overridden.
- **Independence**: no conflict with KV management, quantization, or FR-Spec.

Provides
--------
  C2TConfig            — configuration parameters.
  C2TFeatures          — logit-to-features extractor.
  C2TClassifier        — lightweight binary logistic classifier.
  AdaptiveTreeBuilder  — variable-width tree construction.
  C2TTrainer           — online classifier training.
  C2TStats             — branching statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "C2TConfig",
    "C2TFeatures",
    "C2TClassifier",
    "AdaptiveTreeBuilder",
    "C2TTrainer",
    "C2TStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class C2TConfig:
    """Parameters for C2T adaptive tree construction.

    Parameters
    ----------
    tree_depth:
        Number of draft steps per decode call.
    wide_branches:
        Number of candidates to propose at a *wide* (uncertain) position.
    narrow_branches:
        Number of candidates at a *narrow* (confident) position.
    classify_threshold:
        Logistic score threshold → 1 (wide) when score >= threshold.
    learning_rate:
        Logistic regression step size for the online trainer.
    feature_dim:
        Number of features input to the classifier (default 3:
        top-1 logit, top-2 gap, entropy).
    """

    tree_depth: int = 5
    wide_branches: int = 2
    narrow_branches: int = 1
    classify_threshold: float = 0.5
    learning_rate: float = 1e-3
    feature_dim: int = 3


# ---------------------------------------------------------------------------
# C2TFeatures
# ---------------------------------------------------------------------------

class C2TFeatures:
    """Extract classifier input features from a draft model logit vector.

    Features (in order):
      0 — top-1 logit value (max, unnormalised)
      1 — top-1 minus top-2 logit gap (confidence delta)
      2 — entropy of the softmax distribution (uncertainty)
    """

    _ENTROPY_SCALE: float = 10.0  # scale entropy to be ~same range as logits

    @staticmethod
    def compute(logits: np.ndarray) -> np.ndarray:
        """Compute the 3-D feature vector from a logit array.

        Parameters
        ----------
        logits:
            Shape ``(vocab_size,)`` — raw draft model output.

        Returns
        -------
        features:
            Shape ``(3,)`` float32.
        """
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D; got shape {logits.shape}")
        top2 = np.partition(logits, -2)[-2:]  # two largest
        top2_sorted = np.sort(top2)[::-1]
        top1 = float(top2_sorted[0])
        gap = float(top2_sorted[0] - top2_sorted[1])

        # Entropy via softmax
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        return np.array([top1, gap, entropy], dtype=np.float32)


# ---------------------------------------------------------------------------
# C2TClassifier
# ---------------------------------------------------------------------------

class C2TClassifier:
    """Lightweight binary logistic classifier for wide/narrow branching.

    The model is a single linear layer:
        score = sigmoid(w · features + b)
    Width decision = 1 (wide) when score >= threshold.

    Parameters
    ----------
    feature_dim:
        Number of input features (default 3).
    threshold:
        Decision threshold (default 0.5).
    seed:
        RNG seed for weight initialisation.
    """

    def __init__(
        self,
        feature_dim: int = 3,
        threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._w = rng.standard_normal(feature_dim).astype(np.float32) * 0.01
        self._b = np.float32(0.0)
        self._threshold = threshold
        self._feature_dim = feature_dim

    @property
    def weights(self) -> np.ndarray:
        return self._w.copy()

    @property
    def bias(self) -> float:
        return float(self._b)

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(self, features: np.ndarray) -> float:
        """Logistic score in [0, 1]."""
        if features.ndim != 1 or len(features) != self._feature_dim:
            raise ValueError(
                f"features must be 1-D of length {self._feature_dim}; "
                f"got shape {features.shape}"
            )
        z = float(self._w @ features) + float(self._b)
        return float(1.0 / (1.0 + np.exp(-z)))

    def classify(self, features: np.ndarray) -> int:
        """Return 1 (wide) or 0 (narrow) for the given features."""
        return int(self.score(features) >= self._threshold)

    def update(self, features: np.ndarray, label: int, lr: float) -> None:
        """Logistic regression gradient step.

        Parameters
        ----------
        features:
            Input feature vector, shape ``(feature_dim,)``.
        label:
            Ground-truth label: 1 (branching helped) or 0 (branching hurt).
        lr:
            Learning rate for this step.
        """
        s = self.score(features)
        err = s - float(label)
        self._w -= lr * err * features
        self._b -= lr * err

    def reset(self) -> None:
        """Zero-initialise weights and bias."""
        self._w[:] = 0.0
        self._b = np.float32(0.0)


# ---------------------------------------------------------------------------
# AdaptiveTreeBuilder
# ---------------------------------------------------------------------------

class AdaptiveTreeBuilder:
    """Build a variable-width draft tree using a C2T classifier.

    ``build()`` calls *draft_fn* once per level of the tree.  At each
    position it evaluates the classifier on the draft logits and allocates
    ``wide_branches`` or ``narrow_branches`` candidates accordingly.

    Parameters
    ----------
    config:
        C2T configuration.
    classifier:
        Trained (or freshly initialised) classifier.
    """

    def __init__(
        self,
        config: Optional[C2TConfig] = None,
        classifier: Optional[C2TClassifier] = None,
    ) -> None:
        self._config = config or C2TConfig()
        self._classifier = classifier or C2TClassifier(
            feature_dim=self._config.feature_dim,
            threshold=self._config.classify_threshold,
        )

    @property
    def classifier(self) -> C2TClassifier:
        return self._classifier

    def classify_position(self, logits: np.ndarray) -> int:
        """Return branch width (wide or narrow) for a draft position.

        Parameters
        ----------
        logits:
            Draft model logits at this position, shape ``(vocab_size,)``.

        Returns
        -------
        ``wide_branches`` or ``narrow_branches`` from config.
        """
        feats = C2TFeatures.compute(logits)
        decision = self._classifier.classify(feats)
        return self._config.wide_branches if decision else self._config.narrow_branches

    def build(
        self,
        draft_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        root_hidden: np.ndarray,
        forced_narrow: Optional[Sequence[bool]] = None,
    ) -> List[List[int]]:
        """Construct the adaptive draft tree.

        Parameters
        ----------
        draft_fn:
            Callable ``(hidden: ndarray) -> (logits: ndarray, next_hidden: ndarray)``
            representing one draft model forward pass.
        root_hidden:
            Hidden state at the current decoding position, shape ``(h,)``.
        forced_narrow:
            Optional boolean mask of length ``tree_depth``.  If
            ``forced_narrow[d]`` is True, the position at depth *d* uses
            ``narrow_branches`` regardless of the classifier.

        Returns
        -------
        List of draft paths, each a ``list[int]`` of token IDs.
        """
        cfg = self._config
        depth = cfg.tree_depth
        fn = forced_narrow or [False] * depth

        # BFS expansion
        # Each entry: (hidden_state, prefix_so_far)
        queue: List[Tuple[np.ndarray, List[int]]] = [(root_hidden, [])]
        paths: List[List[int]] = []

        for d in range(depth):
            next_queue: List[Tuple[np.ndarray, List[int]]] = []
            for hidden, prefix in queue:
                logits, next_hidden = draft_fn(hidden)
                # Determine branch width
                if fn[d]:
                    n_branches = cfg.narrow_branches
                else:
                    n_branches = self.classify_position(logits)

                # Select top-n_branches tokens
                top_indices = np.argsort(logits)[-n_branches:][::-1]
                for tok in top_indices:
                    new_prefix = prefix + [int(tok)]
                    next_queue.append((next_hidden, new_prefix))

            queue = next_queue

        for hidden, prefix in queue:
            if prefix:
                paths.append(prefix)

        return paths


# ---------------------------------------------------------------------------
# C2TTrainer
# ---------------------------------------------------------------------------

class C2TTrainer:
    """Online logistic regression trainer for the C2T classifier.

    After each decode step, the trainer updates the classifier based on
    whether branching at a position improved the acceptance outcome.

    Parameters
    ----------
    classifier:
        The :class:`C2TClassifier` to train.
    config:
        C2T configuration providing ``learning_rate``.
    """

    def __init__(
        self,
        classifier: C2TClassifier,
        config: Optional[C2TConfig] = None,
    ) -> None:
        self._classifier = classifier
        self._config = config or C2TConfig()
        self._n_updates: int = 0

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def update(
        self,
        features: np.ndarray,
        branching_helped: bool,
    ) -> None:
        """Update classifier weights for one observed position.

        Parameters
        ----------
        features:
            The ``C2TFeatures`` vector at that draft position.
        branching_helped:
            True if the wider branch accepted a token that the narrow branch
            would have missed.
        """
        label = int(branching_helped)
        self._classifier.update(features, label, self._config.learning_rate)
        self._n_updates += 1

    def reset(self) -> None:
        self._n_updates = 0


# ---------------------------------------------------------------------------
# C2TStats
# ---------------------------------------------------------------------------

@dataclass
class C2TStats:
    """Branching statistics.

    Attributes
    ----------
    wide_decisions:
        Number of positions classified as wide.
    narrow_decisions:
        Number of positions classified as narrow.
    wide_helped:
        Wide decisions where the extra branch was accepted.
    """

    wide_decisions: int = 0
    narrow_decisions: int = 0
    wide_helped: int = 0

    @property
    def wide_fraction(self) -> float:
        total = self.wide_decisions + self.narrow_decisions
        return self.wide_decisions / total if total else 0.0

    @property
    def wide_help_rate(self) -> float:
        return self.wide_helped / self.wide_decisions if self.wide_decisions else 0.0

    def record_wide(self, helped: bool) -> None:
        self.wide_decisions += 1
        if helped:
            self.wide_helped += 1

    def record_narrow(self) -> None:
        self.narrow_decisions += 1

    def reset(self) -> None:
        self.wide_decisions = 0
        self.narrow_decisions = 0
        self.wide_helped = 0
