"""squish/speculative/multi_exit_spec.py

MultiExitSpec — Early-Layer Confidence Exit for Self-Speculative Decoding.

Reference
---------
Gao et al. "Multi-Exit Speculative Decoding."
ACL Findings 2024 (arXiv:2403.15381).

Algorithm
---------
Deep transformers often converge to their final prediction at intermediate
layers.  MultiExitSpec exploits this by:

1. Attaching lightweight language-model heads at several intermediate
   layers.
2. At each exit layer, computing the top-1 probability from the
   intermediate head.
3. If top-1 probability exceeds an exit_threshold, shortcircuit and
   return the early prediction without running deeper layers.
4. Otherwise, continue to the next exit checkpoint.

This is "self-speculative" — no separate draft model, 1.5× decode
speedup at typical confidence thresholds.

Key properties
--------------
* NumPy-only.
* ``exit_layers`` — list of layer indices where exits are attempted.
* ``exit_threshold`` — confidence threshold for early exit.
* ``n_layers`` — total transformer depth.
* ``hidden_dim`` — hidden state dimension.
* ``vocab_size`` — vocabulary size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "MultiExitSpecConfig",
    "ExitResult",
    "MultiExitSpec",
]


@dataclass
class MultiExitSpecConfig:
    """Configuration for :class:`MultiExitSpec`.

    Attributes:
        n_layers: Total number of transformer layers.
        hidden_size: Hidden state dimension.
        vocab_size: Vocabulary size.
        exit_layers: Layer indices at which to attempt early exit.
        threshold: Confidence (top-1 prob) threshold for exit.
    """

    n_layers: int = 32
    hidden_size: int = 4096
    vocab_size: int = 32000
    exit_layers: List[int] = field(default_factory=lambda: [8, 16, 24])
    threshold: float = 0.90

    def __post_init__(self) -> None:
        for el in self.exit_layers:
            if not 0 <= el < self.n_layers:
                raise ValueError(f"exit_layer {el} out of range [0, {self.n_layers})")


@dataclass
class ExitResult:
    """Result of a multi-exit decode step.

    Attributes:
        token: Predicted token ID.
        exit_layer: Layer at which exit occurred (None if no exit fired before final).
        confidence: Top-1 probability at the exit layer.
        is_early_exit: True if exit happened before the final layer.
    """

    token: int
    exit_layer: Optional[int]
    confidence: float
    is_early_exit: bool


class MultiExitSpec:
    """Multi-exit self-speculative decoder.

    Parameters
    ----------
    config:
        MultiExitSpec configuration.
    seed:
        RNG seed for random exit heads.
    """

    def __init__(self, config: Optional[MultiExitSpecConfig] = None, seed: int = 0) -> None:
        self._cfg = config or MultiExitSpecConfig()
        rng = np.random.default_rng(seed)
        # Simulated exit head weights: one (vocab, hidden_size) per exit layer
        scale = 1.0 / np.sqrt(self._cfg.hidden_size)
        self._exit_heads: List[np.ndarray] = [
            rng.standard_normal((self._cfg.vocab_size, self._cfg.hidden_size)).astype(np.float32)
            * scale
            for _ in self._cfg.exit_layers
        ]
        self._early_exits: int = 0
        self._total_calls: int = 0

    @property
    def config(self) -> MultiExitSpecConfig:
        return self._cfg

    @property
    def early_exit_rate(self) -> float:
        return self._early_exits / max(1, self._total_calls)

    def attempt_exits(
        self,
        hidden_states_per_layer,
    ) -> ExitResult:
        """Try early exits at configured layers.

        Parameters
        ----------
        hidden_states_per_layer:
            Dict mapping layer index to hidden state array ``(hidden_size,)``,
            OR a list of ``n_layers`` hidden states each ``(hidden_size,)``.

        Returns
        -------
        ExitResult
            The first exit that meets the confidence threshold, or the final
            layer's prediction if no early exit fires.
        """
        self._total_calls += 1
        # Accept both dict {layer_idx: hidden} and list [h0, h1, ...]
        if isinstance(hidden_states_per_layer, dict):
            state_dict = hidden_states_per_layer
        else:
            state_dict = {i: h for i, h in enumerate(hidden_states_per_layer)}

        for exit_idx, layer_idx in enumerate(self._cfg.exit_layers):
            if layer_idx not in state_dict:
                continue
            h = np.asarray(state_dict[layer_idx], dtype=np.float32)
            W = self._exit_heads[exit_idx]
            logits = (W @ h).astype(np.float64)
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            confidence = float(probs.max())
            if confidence >= self._cfg.threshold:
                token = int(probs.argmax())
                self._early_exits += 1
                return ExitResult(
                    token=token,
                    exit_layer=layer_idx,
                    confidence=confidence,
                    is_early_exit=True,
                )

        # No early exit: use last available hidden state with last exit head
        # Try to use any available state, preferring later layers
        available_layers = sorted(state_dict.keys(), reverse=True)
        if available_layers:
            final_h = np.asarray(state_dict[available_layers[0]], dtype=np.float32)
            W_last = self._exit_heads[-1] if self._exit_heads else np.eye(
                self._cfg.vocab_size, self._cfg.hidden_size, dtype=np.float32
            )
            logits = (W_last @ final_h).astype(np.float64)
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            return ExitResult(
                token=int(probs.argmax()),
                exit_layer=None,
                confidence=float(probs.max()),
                is_early_exit=False,
            )
        # No hidden states at all
        return ExitResult(token=0, exit_layer=None, confidence=0.0, is_early_exit=False)

    def reset_stats(self) -> None:
        self._early_exits = 0
        self._total_calls = 0
