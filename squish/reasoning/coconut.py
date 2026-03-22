"""CoconutDecoder: Continuous Chain-of-Thought reasoning in latent space.

Hao et al. (Meta AI, arXiv 2412.06769, NeurIPS 2024).  Maps intermediate reasoning
steps to continuous latent vectors rather than discrete tokens; executes reasoning via
breadth-first latent search; decodes only the final answer.  Needs a COCONUT-fine-tuned
model for production use; falls back to standard token-by-token decoding transparently.

Reference: Hao et al., "Training Large Language Models to Reason in a Continuous Latent
Space", arXiv 2412.06769, NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CoconutConfig",
    "LatentThoughtState",
    "CoconutResult",
    "CoconutDecoder",
]


@dataclass
class CoconutConfig:
    """Configuration for :class:`CoconutDecoder`.

    Attributes:
        max_latent_steps: Maximum breadth-first latent reasoning steps.
        beam_width: Number of latent beams in the BFS.
        latent_dim: Dimensionality of the latent reasoning space.
        fallback_to_token_decode: If True, fall back to token decoding when
            no projection head is installed.
        projection_scale_init: Initial scale of the projection weight matrix
            (used for the NumPy-based test stub).
        seed: RNG seed.
    """

    max_latent_steps: int = 8
    beam_width: int = 4
    latent_dim: int = 256
    fallback_to_token_decode: bool = True
    projection_scale_init: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_latent_steps < 1:
            raise ValueError(
                f"max_latent_steps must be ≥ 1, got {self.max_latent_steps}"
            )
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be ≥ 1, got {self.beam_width}")
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be ≥ 1, got {self.latent_dim}")


@dataclass
class LatentThoughtState:
    """One beam in the latent BFS search.

    Attributes:
        latent: Current latent vector of shape ``(latent_dim,)``.
        score: Cumulative beam score.
        step: Depth in the BFS tree.
        history: All latent vectors visited by this beam.
    """

    latent: np.ndarray
    score: float = 0.0
    step: int = 0
    history: List[np.ndarray] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return self.step


@dataclass
class CoconutResult:
    """Output of one :meth:`CoconutDecoder.decode` call.

    Attributes:
        answer: Decoded final answer string.
        n_latent_steps: Number of latent reasoning steps performed.
        used_fallback: True if token-level fallback was used.
        best_beam: The highest-scored beam at completion.
        token_reduction_ratio: Approximate ratio of latent-to-token steps.
    """

    answer: str
    n_latent_steps: int
    used_fallback: bool
    best_beam: Optional[LatentThoughtState] = None

    @property
    def token_reduction_ratio(self) -> float:
        """Approx ratio of token savings (0 = no savings, < 1 = fewer tokens)."""
        if self.n_latent_steps == 0:
            return 0.0
        return 1.0 / max(self.n_latent_steps, 1)


# ---------------------------------------------------------------------------
# Projection-head type alias
# ---------------------------------------------------------------------------
# Maps (hidden_state, latent) → next_latent
ProjectionHead = Callable[[np.ndarray, np.ndarray], np.ndarray]
# Maps latent → answer string
AnswerDecoder = Callable[[np.ndarray], str]


class CoconutDecoder:
    """Continuous Chain-of-Thought decoder with BFS latent search.

    Provides a NumPy-backed simulation path for testing.  In production,
    inject *projection_head* and *answer_decoder* callables that call the
    actual model's continuous-reasoning layers.

    Usage::

        cfg = CoconutConfig(max_latent_steps=8, beam_width=4)
        decoder = CoconutDecoder(cfg)
        result = decoder.decode("Solve: 2+2=?")
        # result.answer, result.n_latent_steps, result.used_fallback

    """

    def __init__(
        self,
        config: CoconutConfig,
        projection_head: Optional[ProjectionHead] = None,
        answer_decoder: Optional[AnswerDecoder] = None,
    ) -> None:
        self.config = config
        self._projection_head = projection_head
        self._answer_decoder = answer_decoder
        self._rng = np.random.default_rng(config.seed)
        # Stub projection weight (used when no real projection_head provided)
        self._W_proj = (
            self._rng.standard_normal((config.latent_dim, config.latent_dim)).astype(np.float32)
            * config.projection_scale_init
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def decode(
        self,
        prompt: str,
        hidden_state: Optional[np.ndarray] = None,
    ) -> CoconutResult:
        """Run BFS latent search and decode the final answer.

        Parameters
        ----------
        prompt:
            Input prompt (used to derive initial latent when *hidden_state* is None).
        hidden_state:
            Optional pre-computed hidden state of shape ``(latent_dim,)``.
        """
        if self._projection_head is None and not self.config.fallback_to_token_decode:
            raise RuntimeError(
                "No projection_head installed and fallback_to_token_decode=False."
            )

        if self._projection_head is None:
            return self._fallback_decode(prompt)

        initial = self._initialise_latent(prompt, hidden_state)
        best_beam = self._bfs_search(initial)
        if self._answer_decoder is not None:
            answer = self._answer_decoder(best_beam.latent)
        else:
            answer = self._latent_to_answer_stub(best_beam.latent)

        return CoconutResult(
            answer=answer,
            n_latent_steps=best_beam.step,
            used_fallback=False,
            best_beam=best_beam,
        )

    def install_projection_head(self, head: ProjectionHead) -> None:
        """Attach a trained projection head callable at runtime."""
        self._projection_head = head

    def install_answer_decoder(self, decoder: AnswerDecoder) -> None:
        """Attach a trained answer-decoder callable at runtime."""
        self._answer_decoder = decoder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialise_latent(
        self,
        prompt: str,
        hidden_state: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return an initial latent vector from hidden state or a hash stub."""
        if hidden_state is not None:
            hs = np.asarray(hidden_state, dtype=np.float32).ravel()
            dim = self.config.latent_dim
            if hs.size >= dim:
                return hs[:dim]
            return np.pad(hs, (0, dim - hs.size))
        # Stub: hash the prompt to get a pseudo-deterministic latent
        h = hash(prompt) % (2 ** 31)
        rng = np.random.default_rng(h)
        return rng.standard_normal(self.config.latent_dim).astype(np.float32)

    def _bfs_search(self, initial: np.ndarray) -> LatentThoughtState:
        """Breadth-first search over latent space."""
        beams: List[LatentThoughtState] = [
            LatentThoughtState(latent=initial.copy(), score=0.0, step=0)
        ]
        for _ in range(self.config.max_latent_steps):
            candidates: List[LatentThoughtState] = []
            for beam in beams:
                assert self._projection_head is not None
                next_latent = self._projection_head(initial, beam.latent)
                next_latent = np.asarray(next_latent, dtype=np.float32)
                score = beam.score + float(np.dot(next_latent, beam.latent)) * 0.01
                history = beam.history + [beam.latent.copy()]
                candidates.append(
                    LatentThoughtState(
                        latent=next_latent,
                        score=score,
                        step=beam.step + 1,
                        history=history,
                    )
                )
            # Keep top beam_width by score
            candidates.sort(key=lambda b: b.score, reverse=True)
            beams = candidates[: self.config.beam_width]

        return beams[0]

    def _fallback_decode(self, prompt: str) -> CoconutResult:
        """Simple token-count simulation fallback (no GPU needed)."""
        rng = np.random.default_rng(hash(prompt) % (2 ** 31))
        n_steps = int(rng.integers(1, self.config.max_latent_steps + 1))
        return CoconutResult(
            answer=f"[fallback: {prompt[:16]}...]",
            n_latent_steps=n_steps,
            used_fallback=True,
        )

    def _latent_to_answer_stub(self, latent: np.ndarray) -> str:
        """Stub: encode first two floats as answer representation."""
        v = latent.ravel()
        return f"latent_answer({v[0]:.4f},{v[1]:.4f})" if v.size >= 2 else "latent_answer(?)"
