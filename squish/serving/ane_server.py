# [Experimental] This module is part of Squish v43+ (Wave 69).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""ANEServingRuntime — serving path for Apple Neural Engine inference.

Provides the same streaming token-generation interface as the Metal GPU
path, but internally routes prefill and decode through
:class:`~squish.loaders.coreml_loader.CoreMLRuntime` (which may use real
CoreML or a NumPy simulation depending on availability).

When the ANE path is unavailable (non-macOS, no CoreML appendix, router
decision), the serving runtime falls back to a lightweight NumPy decode
loop that is functionally identical from the client's perspective.

Usage::

    from squish.serving.ane_server import ANEServingRuntime, ANEServerConfig

    cfg = ANEServerConfig(port=11436, fallback_to_gpu=True, max_tokens=512)
    runtime = ANEServingRuntime(config=cfg)
    runtime.prepare(squizd_path="model.squizd")

    for token, finish_reason in runtime.generate_stream("Hello, world!", max_tokens=64):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from squish.loaders.coreml_loader import CoreMLLoader, CoreMLLoaderConfig, CoreMLRuntime
from squish.platform.ane_router import ANERouter, get_ane_router

__all__ = [
    "ANEServerConfig",
    "ANEServingRuntime",
    "GenerationResult",
]

# ---------------------------------------------------------------------------
# Minimal vocabulary / tokeniser simulation
# ---------------------------------------------------------------------------

# Default vocabulary size used throughout this module when no tokeniser is
# available.  Matches the Qwen3 / LLaMA-3 default.
_DEFAULT_VOCAB_SIZE: int = 32_000

# EOS token id used to end generation in simulation mode.
_EOS_TOKEN_ID: int = 2

# Lookup table for simulation: maps integer token id → display string.
# Only a small range is defined; ids outside map to "<tok_{id}>".
_SIM_TOKENS: dict[int, str] = {
    0: "<pad>",
    1: "<s>",
    2: "</s>",
    **{i: f" {chr(97 + (i % 26))}" for i in range(3, 200)},
}


def _decode_token(token_id: int) -> str:
    """Map a token id to a display string (simulation only)."""
    return _SIM_TOKENS.get(token_id, f"<tok_{token_id}>")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ANEServerConfig:
    """Configuration for an :class:`ANEServingRuntime` instance.

    Attributes:
        port: TCP port on which the ANE serving endpoint listens.  For the
            current pure-Python implementation this is informational only —
            actual network binding is handled by the ASGI server.
        fallback_to_gpu: Fall back to the NumPy GPU-simulation path when
            CoreML is unavailable or the ANE appendix is missing.
        max_tokens: Hard cap on generated tokens per request.
        vocab_size: Vocabulary size (default: 32 000).
        temperature: Sampling temperature for next-token selection.
        top_p: Nucleus sampling probability threshold (0 < top_p ≤ 1.0).
    """

    port: int = 11436
    fallback_to_gpu: bool = True
    max_tokens: int = 4_096
    vocab_size: int = _DEFAULT_VOCAB_SIZE
    temperature: float = 1.0
    top_p: float = 0.95


@dataclass
class GenerationResult:
    """Aggregated result returned after a full generation run.

    Attributes:
        text: Concatenated generated text.
        tokens_generated: Number of tokens emitted.
        backend: The backend used (``"coreml_ane"`` or ``"gpu_fallback"``).
        ttft_ms: Time-to-first-token in milliseconds.
        total_ms: Total generation wall time in milliseconds.
    """

    text: str
    tokens_generated: int
    backend: str
    ttft_ms: float
    total_ms: float


# ---------------------------------------------------------------------------
# ANEServingRuntime
# ---------------------------------------------------------------------------

class ANEServingRuntime:
    """Serving runtime that dispatches inference through CoreML or NumPy.

    Parameters:
        config: :class:`ANEServerConfig`; defaults used if ``None``.
        _router: Inject an :class:`ANERouter` for testing.
        _loader: Inject a :class:`CoreMLLoader` for testing.
    """

    def __init__(
        self,
        config: Optional[ANEServerConfig] = None,
        *,
        _router: Optional[ANERouter] = None,
        _loader: Optional[CoreMLLoader] = None,
    ) -> None:
        self.config = config or ANEServerConfig()
        self._router: ANERouter = _router or get_ane_router()
        self._loader: CoreMLLoader = _loader or CoreMLLoader(
            CoreMLLoaderConfig(
                fallback_to_gpu=self.config.fallback_to_gpu,
                ane_router=self._router,
            )
        )
        self._runtime: Optional[CoreMLRuntime] = None
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self, squizd_path: Union[str, os.PathLike]) -> None:
        """Load the model from *squizd_path* and prepare for inference.

        Args:
            squizd_path: Path to the ``.squizd`` model file.  May contain an
                ANE_COREML appendix; if absent and ``fallback_to_gpu`` is
                ``True``, the runtime still becomes ready in fallback mode.
        """
        self._runtime = self._loader.load(str(squizd_path))
        self._ready = self._runtime.is_loaded()

    def is_ready(self) -> bool:
        """Return ``True`` if the runtime has been prepared and is ready."""
        return self._ready

    def backend(self) -> str:
        """Return the active backend name.

        Returns:
            ``"coreml_ane"`` when CoreML is active, ``"gpu_fallback"`` when
            the NumPy simulation path is active, or ``"uninitialized"`` if
            :meth:`prepare` has not been called.
        """
        if self._runtime is None:
            return "uninitialized"
        raw = self._runtime.backend()
        return "coreml_ane" if raw == "coreml" else "gpu_fallback"

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[str, Optional[str]]]:
        """Stream generated tokens one at a time.

        Yields ``(token_text, finish_reason)`` pairs.  ``finish_reason`` is
        ``None`` for intermediate tokens; ``"stop"`` on EOS; ``"length"`` when
        the token cap is reached.

        Args:
            prompt: Plain-text prompt string.
            max_tokens: Override :attr:`ANEServerConfig.max_tokens`.
            temperature: Override :attr:`ANEServerConfig.temperature`.
            top_p: Override :attr:`ANEServerConfig.top_p`.
            seed: Deterministic seed for the sampling RNG.

        Yields:
            ``(token_str, finish_reason_or_None)``

        Raises:
            RuntimeError: If :meth:`prepare` has not been called first.
        """
        if not self._ready:
            raise RuntimeError(
                "ANEServingRuntime is not ready; call prepare() first"
            )

        n_max = min(
            max_tokens if max_tokens is not None else self.config.max_tokens,
            self.config.max_tokens,
        )
        temp = temperature if temperature is not None else self.config.temperature
        p_top = top_p if top_p is not None else self.config.top_p

        # Simulate tokenisation: split on whitespace, map to fake token ids.
        input_ids = self._pseudo_tokenise(prompt)
        rng = np.random.default_rng(seed=seed)

        for step in range(n_max):
            logits = self._runtime.predict(  # type: ignore[union-attr]
                input_ids=input_ids
            )
            token_id = self._sample(logits[0], rng, temperature=temp, top_p=p_top)
            token_text = _decode_token(token_id)

            if token_id == _EOS_TOKEN_ID:
                yield (token_text, "stop")
                return

            if step == n_max - 1:
                yield (token_text, "length")
                return

            yield (token_text, None)

            # Append newly generated token for next step (simulate autoregression).
            input_ids = np.concatenate(
                [input_ids, np.array([[token_id]])], axis=1
            )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """Non-streaming generation; returns after all tokens are produced.

        Args:
            prompt: Plain-text prompt string.
            max_tokens: Override :attr:`ANEServerConfig.max_tokens`.
            temperature: Override :attr:`ANEServerConfig.temperature`.
            top_p: Override :attr:`ANEServerConfig.top_p`.
            seed: Deterministic seed.

        Returns:
            A :class:`GenerationResult` with full text and timing.
        """
        start = time.perf_counter()
        ttft_ms = 0.0
        parts: List[str] = []
        finish = "length"

        for i, (tok, fr) in enumerate(
            self.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        ):
            if i == 0:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            parts.append(tok)
            if fr is not None:
                finish = fr
                break

        total_ms = (time.perf_counter() - start) * 1000.0
        return GenerationResult(
            text="".join(parts),
            tokens_generated=len(parts),
            backend=self.backend(),
            ttft_ms=ttft_ms,
            total_ms=total_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pseudo_tokenise(text: str) -> np.ndarray:
        """Map prompt text to a (1, seq_len) integer array (simulation)."""
        words = text.split() or ["<empty>"]
        # Map each word to an integer id via hash, bounded to a safe range.
        ids = [(abs(hash(w)) % (_DEFAULT_VOCAB_SIZE - 3)) + 3 for w in words]
        return np.array([ids], dtype=np.int64)

    @staticmethod
    def _sample(
        logits: np.ndarray,
        rng: np.random.Generator,
        *,
        temperature: float,
        top_p: float,
    ) -> int:
        """Sample a token id from *logits* using temperature + nucleus sampling.

        Args:
            logits: 1-D float array of shape ``(vocab_size,)``.
            rng: NumPy random Generator for reproducibility.
            temperature: Scales logit magnitudes.
            top_p: Nucleus probability threshold.

        Returns:
            Sampled integer token id.
        """
        if temperature <= 0.0:
            return int(np.argmax(logits))

        scaled = logits / temperature
        # Numerical stability: subtract max before softmax.
        shifted = scaled - scaled.max()
        probs = np.exp(shifted)
        probs /= probs.sum()

        # Nucleus (top-p) filtering.
        if 0.0 < top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[sorted_idx])
            cutoff = int(np.searchsorted(cumulative, top_p)) + 1
            mask = np.zeros_like(probs, dtype=bool)
            mask[sorted_idx[:cutoff]] = True
            probs = np.where(mask, probs, 0.0)
            total = probs.sum()
            if total > 0:
                probs /= total

        return int(rng.choice(len(probs), p=probs))
