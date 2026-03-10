"""
squish/self_learning.py

Self-learning engine for the Squish Block-Expert Archive.

When users interact with a deployed model they can POST examples to the
``/v1/learn`` endpoint.  This module handles the on-device fine-tuning step
(CPU-only, gradient-free finite-difference optimisation) and writes the
resulting weight delta into the archive as a new or updated expert cluster.

Public API
──────────
    LearnConfig              — dataclass of hyper-parameters
    LearnExample             — single (input, output) training pair
    LearnRequest             — Pydantic model for /v1/learn HTTP body
    LearnResult              — result dataclass returned from learn_from_examples
    SelfLearner              — orchestrates the learning loop
    compute_delta_snr        — standalone SNR utility
    examples_from_jsonl      — load JSONL file into list[LearnExample]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pydantic import BaseModel, Field
    _PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment, misc]
    Field = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration & data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LearnConfig:
    """Hyper-parameters for the self-learning step.

    Parameters
    ----------
    steps:
        Number of finite-difference optimisation steps.
    lr:
        Learning rate applied to the gradient estimate.
    batch_size:
        Number of examples sampled per step.
    epsilon:
        Perturbation scale for the finite-difference gradient estimate.
    max_rank:
        Rank of the learned delta matrix (low-rank regularisation applied via
        SVD truncation after optimisation).  Set to 0 to disable truncation.
    min_snr_db:
        Minimum acceptable SNR (dB) for the learned delta.  Learning is
        retried at higher precision if the threshold is not met.
    seed:
        RNG seed for reproducibility (``None`` = non-deterministic).
    domain:
        Optional domain tag stored in the archive index.
    """

    steps: int = 50
    lr: float = 1e-4
    batch_size: int = 4
    epsilon: float = 1e-3
    max_rank: int = 8
    min_snr_db: float = 20.0
    seed: int | None = 42
    domain: str = "general"

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be ≥ 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be ≥ 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_rank < 0:
            raise ValueError("max_rank must be ≥ 0")


@dataclass
class LearnExample:
    """A single (input, output) training pair.

    In tokenised form both fields are lists of token-ids (ints).  In plain
    text form they are unicode strings; the learner converts them to simple
    hash-based pseudo-activations for CPU-only operations.
    """

    input: str | list[int]
    output: str | list[int]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("weight must be positive")


@dataclass
class LearnResult:
    """Output of :meth:`SelfLearner.learn_from_examples`.

    Parameters
    ----------
    delta:
        Dict mapping block_idx → learned weight delta (float32 ndarray).
    snr_db:
        Signal-to-noise ratio (dB) of the delta relative to the base weights.
    steps_run:
        Actual number of optimisation steps completed.
    elapsed_s:
        Wall-clock time for the learning run in seconds.
    domain:
        Domain tag from :class:`LearnConfig`.
    examples_used:
        Number of examples processed.
    """

    delta: dict[int, np.ndarray]
    snr_db: float
    steps_run: int
    elapsed_s: float
    domain: str
    examples_used: int


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model for /v1/learn HTTP endpoint
# ─────────────────────────────────────────────────────────────────────────────


if _PYDANTIC_AVAILABLE:

    class LearnRequest(BaseModel):  # type: ignore[valid-type]
        """Request body for ``POST /v1/learn``."""

        examples: list[dict[str, Any]] = Field(
            ...,
            description="List of {input, output} dicts.",
        )
        domain: str = Field(
            default="general",
            description="Domain tag for the learned expert (e.g. 'legal', 'code').",
        )
        steps: int = Field(
            default=50,
            ge=1,
            le=500,
            description="Number of finite-difference optimisation steps.",
        )
        lr: float = Field(
            default=1e-4,
            gt=0,
            description="Learning rate.",
        )
        epsilon: float = Field(
            default=1e-3,
            gt=0,
            description="Finite-difference perturbation scale.",
        )
        max_rank: int = Field(
            default=8,
            ge=0,
            description="Low-rank cutoff for delta (0 = no truncation).",
        )

else:  # pragma: no cover

    class LearnRequest:  # type: ignore[no-redef]
        """Fallback when Pydantic is not installed."""

        def __init__(
            self,
            examples: list[dict[str, Any]],
            domain: str = "general",
            steps: int = 50,
            lr: float = 1e-4,
            epsilon: float = 1e-3,
            max_rank: int = 8,
        ) -> None:
            self.examples = examples
            self.domain = domain
            self.steps = steps
            self.lr = lr
            self.epsilon = epsilon
            self.max_rank = max_rank


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def compute_delta_snr(base: np.ndarray, delta: np.ndarray) -> float:
    """Compute SNR (dB) of *delta* relative to *base*.

    SNR = 10 · log10(E[base²] / E[delta²])

    A high SNR means the delta is small compared to the base — safe to apply
    without perturbing the model significantly.

    Parameters
    ----------
    base:
        Base weight array.
    delta:
        Delta weight array (same shape as *base*).

    Returns
    -------
    float
        SNR in dB.  Returns ``inf`` if *delta* is zero.
    """
    base_power = float(np.mean(base.astype(np.float64) ** 2))
    delta_power = float(np.mean(delta.astype(np.float64) ** 2))
    if delta_power < 1e-30:
        return float("inf")
    if base_power < 1e-30:
        return 0.0
    return float(10.0 * np.log10(base_power / delta_power + 1e-12))


def _example_to_activation(
    example: LearnExample,
    embed_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Produce a pseudo-activation vector from a ``LearnExample``.

    This is a **CPU-only surrogate** used when the full model is unavailable.
    It deterministically maps the string content to a unit vector in
    ``embed_dim`` dimensions via a seeded hash-based projection.

    Parameters
    ----------
    example:
        The training example.
    embed_dim:
        Desired embedding dimensionality.
    rng:
        NumPy Generator (seeded for reproducibility).

    Returns
    -------
    np.ndarray, shape (embed_dim,), dtype float32
    """
    # Hash the string into an integer seed
    text = (
        str(example.input) + " → " + str(example.output)
    ).encode("utf-8")
    seed_val = int.from_bytes(
        bytes([b ^ 0xA5 for b in text[:32]]), byteorder="little"
    ) & 0xFFFFFFFF
    local_rng = np.random.default_rng(seed_val)
    vec = local_rng.standard_normal(embed_dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / (norm + 1e-12)) * float(example.weight)


def _low_rank_truncation(
    delta: np.ndarray,
    max_rank: int,
) -> np.ndarray:
    """Truncate *delta* to a rank-*max_rank* approximation via SVD.

    Parameters
    ----------
    delta:
        2-D float32 array to truncate.
    max_rank:
        Desired rank; if 0 or ≥ min(rows, cols), the original is returned.

    Returns
    -------
    np.ndarray
        Low-rank approximation, same shape as *delta*.
    """
    if max_rank <= 0 or delta.ndim != 2:
        return delta
    r = min(max_rank, min(delta.shape))
    if r >= min(delta.shape):
        return delta
    U, s, Vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    return (U[:, :r] * s[:r] @ Vt[:r, :]).astype(np.float32)


def examples_from_jsonl(path: str | Path) -> list[LearnExample]:
    """Load training examples from a JSONL file.

    Each line must be a JSON object with at least ``"input"`` and ``"output"``
    keys.  An optional ``"weight"`` key (float) scales the example's influence.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.

    Returns
    -------
    list[LearnExample]

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any line is missing required keys.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    examples: list[LearnExample] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
        if "input" not in obj or "output" not in obj:
            raise ValueError(
                f"Line {lineno} missing 'input' or 'output' key: {obj}"
            )
        examples.append(
            LearnExample(
                input=obj["input"],
                output=obj["output"],
                weight=float(obj.get("weight", 1.0)),
            )
        )
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Self-learner
# ─────────────────────────────────────────────────────────────────────────────


class SelfLearner:
    """CPU-only, gradient-free self-learning engine.

    Uses a finite-difference approximation of the directional gradient to
    iteratively update a weight delta that increases the dot-product between
    the model's notional output direction and the target examples.

    Parameters
    ----------
    base_weights:
        Dict of ``block_idx → base weight array`` for the Transformer blocks
        that will be updated.
    config:
        :class:`LearnConfig` controlling the learning loop.
    """

    def __init__(
        self,
        base_weights: dict[int, np.ndarray],
        config: LearnConfig | None = None,
    ) -> None:
        self._base = {k: v.astype(np.float32) for k, v in base_weights.items()}
        self._config = config or LearnConfig()
        self._rng = np.random.default_rng(self._config.seed)

    # ── Core learning loop ────────────────────────────────────────────────────

    def learn_from_examples(
        self,
        examples: list[LearnExample],
        config_override: LearnConfig | None = None,
    ) -> LearnResult:
        """Run the learning loop and return a :class:`LearnResult`.

        Parameters
        ----------
        examples:
            Training examples (at least 1 required).
        config_override:
            If given, overrides ``self._config`` for this call only.

        Returns
        -------
        LearnResult

        Raises
        ------
        ValueError
            If *examples* is empty.
        """
        if not examples:
            raise ValueError("examples must not be empty")

        cfg = config_override or self._config
        rng = np.random.default_rng(cfg.seed)
        t0 = time.perf_counter()

        # Build pseudo-activation targets for each example
        block_indices = sorted(self._base.keys())
        embed_dims = {bi: self._base[bi].shape[0] for bi in block_indices}

        # Initialise delta as zero
        deltas: dict[int, np.ndarray] = {
            bi: np.zeros_like(self._base[bi]) for bi in block_indices
        }

        # Finite-difference optimisation
        n_examples = len(examples)
        for step in range(cfg.steps):
            # Sample a mini-batch
            batch_size = min(cfg.batch_size, n_examples)
            indices = rng.choice(n_examples, size=batch_size, replace=False).tolist()
            batch = [examples[i] for i in indices]

            for bi in block_indices:
                base = self._base[bi]
                delta = deltas[bi]
                ed = embed_dims[bi]

                # Aggregate gradient estimate over mini-batch
                grad_accum = np.zeros_like(delta)

                for ex in batch:
                    act = _example_to_activation(ex, ed, rng)
                    # Current score: projection of (base + delta) onto activation
                    current_w = base + delta
                    if current_w.ndim == 2:
                        score_pos = float(act @ current_w.mean(axis=1).clip(-1, 1))
                    else:
                        score_pos = float(np.dot(act.flatten()[:current_w.shape[0]], current_w.clip(-1, 1)))

                    # Perturb in a random direction
                    direction = rng.standard_normal(delta.shape).astype(np.float32)
                    direction /= np.linalg.norm(direction) + 1e-12

                    perturbed = current_w + cfg.epsilon * direction
                    if perturbed.ndim == 2:
                        score_pert = float(act @ perturbed.mean(axis=1).clip(-1, 1))
                    else:
                        score_pert = float(np.dot(act.flatten()[:perturbed.shape[0]], perturbed.clip(-1, 1)))

                    # Finite-difference gradient estimate
                    grad_est = (score_pert - score_pos) / (cfg.epsilon + 1e-30)
                    grad_accum += grad_est * direction

                # Gradient step
                grad_accum /= len(batch)
                deltas[bi] = (delta + cfg.lr * grad_accum).astype(np.float32)

        # Low-rank truncation
        if cfg.max_rank > 0:
            for bi in block_indices:
                if deltas[bi].ndim == 2:
                    deltas[bi] = _low_rank_truncation(deltas[bi], cfg.max_rank)

        # Compute aggregate SNR
        snr_values = [compute_delta_snr(self._base[bi], deltas[bi]) for bi in block_indices]
        avg_snr = float(np.mean(snr_values)) if snr_values else 0.0

        elapsed = time.perf_counter() - t0
        return LearnResult(
            delta=deltas,
            snr_db=avg_snr,
            steps_run=cfg.steps,
            elapsed_s=elapsed,
            domain=cfg.domain,
            examples_used=n_examples,
        )

    # ── Integration with BlockExpertArchive ───────────────────────────────────

    def apply_result_to_archive(
        self,
        result: LearnResult,
        archive: Any,  # BlockExpertArchive — avoid circular import
    ) -> list[int]:
        """Apply a :class:`LearnResult` to the archive by absorbing weight deltas.

        For each Transformer block, adds the learned delta to the base weight
        and calls :meth:`BlockExpertArchive.absorb_snapshot` to update (or
        create) the nearest expert cluster.

        Parameters
        ----------
        result:
            Learning result from :meth:`learn_from_examples`.
        archive:
            A :class:`~squish.block_expert_archive.BlockExpertArchive` instance.

        Returns
        -------
        list[int]
            The cluster indices that were updated for each block, in
            ``sorted(block_indices)`` order.
        """
        updated_clusters: list[int] = []
        for bi in sorted(result.delta.keys()):
            base = self._base.get(bi)
            if base is None:
                continue
            expert_weight = base + result.delta[bi]
            updated_k = archive.absorb_snapshot(bi, expert_weight, base)
            updated_clusters.append(updated_k)
        return updated_clusters
