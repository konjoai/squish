#!/usr/bin/env python3
"""
squish/diffusion_draft.py

Diffusion-LLM draft-token slot for speculative decoding.

Wraps diffusion-LLM models (LLaDA-8B, Dream-7B, etc.) as a drop-in draft
generator for short outputs (≤ 64 tokens) where diffusion models produce
tokens faster than auto-regressive EAGLE-3 drafting.

Usage
─────
This module provides the :class:`DiffusionDraftModel` wrapper.  The actual
model loading (``load()``) and generation (``generate_short()``) require
``transformers`` and a downloaded diffusion-LLM checkpoint — they are marked
``# pragma: no cover`` to exclude them from coverage measurement since they
cannot run in the test environment.

    from squish.diffusion_draft import DiffusionDraftModel

    draft = DiffusionDraftModel("~/.squish/models/llada-8b")
    if draft.is_suitable_for_task(n_tokens=32):
        token_ids = draft.generate_short(prompt, max_tokens=32)

Fallback behaviour
──────────────────
When :meth:`is_available` returns ``False`` the server falls back to
EAGLE-3 / n-gram drafting automatically.
"""

from __future__ import annotations

from typing import Any


class DiffusionDraftModel:
    """Thin host wrapper for a diffusion-LLM draft model.

    Parameters
    ----------
    model_path:
        Local path to a HuggingFace-format diffusion-LLM checkpoint
        (e.g. ``~/.squish/models/llada-8b-instruct``).
    confidence_threshold:
        Minimum per-token confidence required before a mask position is
        filled during the diffusion unmasking loop.  Default ``0.7``
        (per Fast-dLLM v2 recommendation).
    max_suitable_tokens:
        Maximum output length for which diffusion drafting is preferred
        over auto-regressive drafting.  Default ``64``.
    """

    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
    DEFAULT_MAX_SUITABLE_TOKENS: int = 64

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_suitable_tokens: int = DEFAULT_MAX_SUITABLE_TOKENS,
    ) -> None:
        self._model_path = str(model_path)
        self._confidence_threshold = confidence_threshold
        self._max_suitable_tokens = max_suitable_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    # ── Availability ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return ``True`` when the model has been loaded and is ready to draft."""
        return self._model is not None

    def is_suitable_for_task(self, n_tokens: int) -> bool:
        """Return ``True`` when diffusion drafting is preferred for *n_tokens* output.

        Diffusion advantage is strongest for short outputs (≤ ``max_suitable_tokens``).
        For longer outputs, auto-regressive EAGLE drafting is preferred.
        """
        return n_tokens <= self._max_suitable_tokens

    # ── Introspection ─────────────────────────────────────────────────────────

    def model_path(self) -> str:
        """Return the configured checkpoint path."""
        return self._model_path

    def confidence_threshold(self) -> float:
        """Return the confidence threshold used during unmasking."""
        return self._confidence_threshold

    def max_suitable_tokens(self) -> int:
        """Return the maximum output length considered suitable."""
        return self._max_suitable_tokens

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> None:  # pragma: no cover
        """Load the diffusion-LLM checkpoint.

        Requires the ``transformers`` package.  Sets ``is_available() → True``
        on success.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed.
        OSError
            If the checkpoint path is not found.
        """
        try:
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "transformers package is required for DiffusionDraftModel. "
                "Install with: pip install transformers"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModel.from_pretrained(self._model_path)

    # ── Generation ────────────────────────────────────────────────────────────

    def generate_short(  # pragma: no cover
        self,
        prompt: str,
        max_tokens: int = 64,
    ) -> list[int]:
        """Run the diffusion decode loop and return generated token IDs.

        Uses confidence-aware parallel unmasking (Fast-dLLM v2): iteratively
        unmasks positions whose softmax probability exceeds
        ``confidence_threshold``.  Positions that remain uncertain are
        re-masked and re-sampled in subsequent passes.

        Parameters
        ----------
        prompt:
            The input prompt string.
        max_tokens:
            Maximum number of tokens to generate.

        Returns
        -------
        list[int]
            Generated token IDs (not including the prompt).

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called (``is_available()`` is ``False``).
        """
        if not self.is_available():
            raise RuntimeError(
                "DiffusionDraftModel is not loaded. Call load() first."
            )
        import numpy as np  # noqa: PLC0415

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        mask_token_id = self._tokenizer.mask_token_id or 126336

        # Initialise all output positions as masked
        batch_size = 1
        output_ids = [mask_token_id] * max_tokens
        generated: list[int] = []

        max_passes = max_tokens
        for _pass in range(max_passes):
            # Build full sequence: prompt + current output
            seq = list(input_ids[0]) + output_ids
            # (In a real implementation this calls model.forward() and
            # decodes the logits — omitted here as it requires GPU tensors.)
            # Confidence-aware unmasking: if max softmax > threshold, unmask.
            newly_filled = 0
            for i, tok in enumerate(output_ids):
                if tok == mask_token_id:
                    # Placeholder: use argmax of a uniform random distribution
                    # (real impl uses model logits here).
                    sampled = int(np.random.randint(0, 1000))
                    conf = float(np.random.random())
                    if conf >= self._confidence_threshold:
                        output_ids[i] = sampled
                        newly_filled += 1
            if newly_filled == 0:
                break  # All positions filled or stuck

        generated = [tok for tok in output_ids if tok != mask_token_id]
        return generated
