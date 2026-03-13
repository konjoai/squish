# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""QuantSpecDecode — INT4 draft + FP16 verify speculative decoding.

The draft model runs in INT4 to reduce memory 4× vs FP16, while verification
uses FP16 (or FP32) target logits.  Acceptance uses standard rejection
sampling.  INT4 is simulated via uint8 storage with a per-tensor scale.

References:
    Leviathan et al., "Fast Inference from Transformers via Speculative
    Decoding", ICML 2023.  https://arxiv.org/abs/2211.17192

    Kim et al., "Speculative Decoding with Big Little Decoder", NeurIPS 2023.
    https://arxiv.org/abs/2302.07863

Usage::

    from squish.speculative.quant_spec_decode import QuantSpecDecoder, QSDConfig
    import numpy as np

    cfg     = QSDConfig(n_draft_tokens=4, vocab_size=32000, temperature=1.0)
    decoder = QuantSpecDecoder(cfg)

    draft_logits  = np.random.randn(4, 32000).astype(np.float32)
    target_logits = np.random.randn(4, 32000).astype(np.float32)

    step   = decoder.quantize_draft(draft_logits)
    tokens, n = decoder.verify(step, target_logits)
    print(f"accepted={n}/{cfg.n_draft_tokens}, rate={decoder.stats.acceptance_rate:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "QSDConfig",
    "DraftStep",
    "QuantSpecDecoder",
    "QSDStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QSDConfig:
    """Configuration for quantised speculative decoding.

    Attributes:
        n_draft_tokens:  Number of tokens the draft model generates per step.
        vocab_size:      Vocabulary size of the shared tokeniser.
        draft_quant_bits: Bit-width for draft logit quantisation.  Must be 4.
        temperature:     Sampling temperature applied during rejection
                         sampling.  Must be > 0.
    """

    n_draft_tokens: int = 4
    vocab_size: int = 32000
    draft_quant_bits: int = 4
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.n_draft_tokens < 1:
            raise ValueError(
                f"n_draft_tokens must be >= 1; got {self.n_draft_tokens}"
            )
        if self.vocab_size < 2:
            raise ValueError(
                f"vocab_size must be >= 2; got {self.vocab_size}"
            )
        if self.draft_quant_bits != 4:
            raise ValueError(
                f"draft_quant_bits must be 4; got {self.draft_quant_bits}"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0; got {self.temperature}"
            )

    @property
    def quant_max(self) -> int:
        """Maximum positive integer value for symmetric INT4 quantisation."""
        return (1 << (self.draft_quant_bits - 1)) - 1  # 7 for 4-bit


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DraftStep:
    """Quantised output from one draft-model forward pass.

    Attributes:
        tokens:   Draft token IDs chosen (e.g. argmax of dequantised logits),
                  shape ``(n_draft,)``, int32.
        logits_q: INT4-simulated logits stored as uint8, shape
                  ``(n_draft, vocab_size)``.  Dequantise with
                  ``(logits_q - quant_max) * scale``.
        scale:    Per-step dequantisation scalar (float).
    """

    tokens: np.ndarray
    logits_q: np.ndarray
    scale: float


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class QSDStats:
    """Cumulative statistics for :class:`QuantSpecDecoder`.

    Attributes:
        total_draft:    Total draft tokens presented for verification.
        total_accepted: Total draft tokens accepted by the target model.
    """

    total_draft: int = 0
    total_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted across all verification calls.

        Returns 0.0 when no tokens have been presented yet.
        """
        if self.total_draft == 0:
            return 0.0
        return self.total_accepted / self.total_draft


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _softmax_t(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    Args:
        logits:      1-D float array of shape ``(vocab_size,)``.
        temperature: Positive temperature scalar.

    Returns:
        Probability vector, float32.
    """
    scaled = logits.astype(np.float64) / temperature
    shifted = scaled - np.max(scaled)
    exp_v = np.exp(shifted)
    return (exp_v / exp_v.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# QuantSpecDecoder
# ---------------------------------------------------------------------------


class QuantSpecDecoder:
    """INT4 draft + FP32 verify speculative decoder.

    Quantises draft model logits to INT4 (simulated via uint8 + scale) to
    represent the draft model's reduced-precision output, then verifies each
    draft token against full-precision target logits using standard rejection
    sampling.

    Args:
        config: :class:`QSDConfig` instance.
    """

    def __init__(self, config: QSDConfig) -> None:
        self._config = config
        self._stats = QSDStats()
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize_draft(self, logits: np.ndarray) -> DraftStep:
        """Quantise draft model logits to INT4 representation.

        The entire logit tensor is scaled symmetrically: the scale is
        ``max(|logits|) / quant_max``, values are rounded and clamped to
        ``[-quant_max, +quant_max]``, then stored as uint8 offset by
        ``quant_max``.  Draft tokens are the argmax of the dequantised
        (or equivalently, the original) logits.

        Args:
            logits: Draft model logits, shape ``(n_draft, vocab_size)``,
                    float32.

        Returns:
            :class:`DraftStep` with quantised logits, draft token IDs, and
            the dequantisation scale.

        Raises:
            ValueError: if ``logits`` shape does not match
                        ``(n_draft_tokens, vocab_size)``.
        """
        cfg = self._config
        expected = (cfg.n_draft_tokens, cfg.vocab_size)
        if logits.shape != expected:
            raise ValueError(
                f"logits shape {logits.shape} does not match expected "
                f"{expected}"
            )

        q_max = cfg.quant_max
        abs_max = float(np.max(np.abs(logits)))
        scale = abs_max / q_max if abs_max > 1e-30 else 1.0

        logits_scaled = logits / scale
        logits_clipped = np.clip(np.round(logits_scaled), -q_max, q_max)
        logits_q = (logits_clipped + q_max).astype(np.uint8)

        # Tokens are the argmax over the original logits (equivalent to
        # argmax over quantised logits since scale is monotone positive).
        tokens = np.argmax(logits, axis=-1).astype(np.int32)

        return DraftStep(tokens=tokens, logits_q=logits_q, scale=scale)

    def verify(
        self,
        draft: DraftStep,
        target_logits: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Verify draft tokens against target model logits via rejection sampling.

        For each draft token ``t_i`` at position ``i``:
        - Dequantise draft logits to obtain ``p_draft``.
        - Compute ``p_target`` from target logits.
        - Accept with probability ``min(1, p_target(t_i) / p_draft(t_i))``.
        - On rejection, sample a correction from the residual distribution
          and stop.

        Args:
            draft:         :class:`DraftStep` produced by
                           :meth:`quantize_draft`.
            target_logits: Target model logits, shape
                           ``(n_draft, vocab_size)``, float32.

        Returns:
            ``(accepted_tokens, n_accepted)`` where ``accepted_tokens`` is a
            1-D int32 array of accepted (and possibly one correction) token
            IDs, and ``n_accepted`` is the count.

        Raises:
            ValueError: if ``target_logits`` shape does not match
                        ``(n_draft_tokens, vocab_size)``.
        """
        cfg = self._config
        expected = (cfg.n_draft_tokens, cfg.vocab_size)
        if target_logits.shape != expected:
            raise ValueError(
                f"target_logits shape {target_logits.shape} does not match "
                f"expected {expected}"
            )

        q_max = cfg.quant_max
        # Dequantise: shift offset then multiply by scale.
        draft_logits_fp = (draft.logits_q.astype(np.float32) - q_max) * draft.scale

        accepted: list[int] = []
        n_draft = cfg.n_draft_tokens

        for i in range(n_draft):
            token = int(draft.tokens[i])
            p_d = _softmax_t(draft_logits_fp[i], cfg.temperature)
            p_t = _softmax_t(target_logits[i], cfg.temperature)

            accept_prob = min(1.0, float(p_t[token]) / (float(p_d[token]) + 1e-30))
            if self._rng.random() < accept_prob:
                accepted.append(token)
            else:
                # Sample correction from residual distribution.
                residual = np.maximum(0.0, p_t - p_d)
                residual_sum = float(residual.sum())
                if residual_sum > 1e-30:
                    residual /= residual_sum
                    correction = int(self._rng.choice(len(residual), p=residual))
                    accepted.append(correction)
                break

        n_accepted = len(accepted)
        accepted_arr = np.array(accepted, dtype=np.int32)

        self._stats.total_draft += n_draft
        self._stats.total_accepted += n_accepted

        return accepted_arr, n_accepted

    @property
    def stats(self) -> QSDStats:
        """Cumulative quantised speculative decoding statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset cumulative statistics to zero."""
        self._stats = QSDStats()
