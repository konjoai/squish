"""squish/eval.py

Platform-aware evaluation router.

Selects the correct lm-evaluation-harness LM subclass based on the current
platform:
  - macOS (Apple Silicon / MLX) → SquishCompressedLM  (squish_lm_eval)
  - Linux / Windows (PyTorch)   → SquishCompressedLMTorch  (_eval_torch)

Public API::

    from squish.eval import get_compressed_lm, get_reference_lm

    lm = get_compressed_lm(
        model_dir="~/models/Qwen2.5-7B-Instruct-bf16",
        compressed_dir="~/models/Qwen2.5-7B-Instruct-compressed",
    )

Both classes expose the standard lm-eval interface:
    loglikelihood(), loglikelihood_rolling(), generate_until()

Platform detection can be overridden via the ``platform`` keyword argument
("darwin" | "linux" | "auto").
"""

from __future__ import annotations

__all__ = [
    "get_compressed_lm",
    "get_reference_lm",
    "PLATFORM",
]

import sys

PLATFORM: str = "darwin" if sys.platform == "darwin" else "linux"


def get_compressed_lm(
    *,
    platform: str = "auto",
    **kwargs,
):
    """Return a platform-appropriate SquishCompressedLM instance.

    Args:
        platform: ``"darwin"``, ``"linux"``, or ``"auto"`` (default).
                  ``"auto"`` selects based on ``sys.platform``.
        **kwargs: Forwarded to the underlying LM class constructor.
            common keys: model_dir, compressed_dir, batch_size, max_length.

    Returns:
        SquishCompressedLM on macOS, SquishCompressedLMTorch on Linux/Windows.
    """
    if platform == "auto":
        _plat = PLATFORM
    else:
        _plat = platform

    if _plat == "darwin":
        from squish.squish_lm_eval import SquishCompressedLM
        return SquishCompressedLM(**kwargs)
    else:
        from squish._eval_torch import SquishCompressedLMTorch
        return SquishCompressedLMTorch(**kwargs)


def get_reference_lm(
    *,
    platform: str = "auto",
    **kwargs,
):
    """Return a platform-appropriate reference (uncompressed BF16) LM instance.

    Args:
        platform: ``"darwin"``, ``"linux"``, or ``"auto"`` (default).
        **kwargs: Forwarded to the underlying LM class constructor.
            common key: model_dir.

    Returns:
        SquishReferenceLM on macOS, SquishReferenceLMTorch on Linux/Windows.
    """
    if platform == "auto":
        _plat = PLATFORM
    else:
        _plat = platform

    if _plat == "darwin":
        from squish.squish_lm_eval import SquishReferenceLM
        return SquishReferenceLM(**kwargs)
    else:
        from squish._eval_torch import SquishReferenceLMTorch
        return SquishReferenceLMTorch(**kwargs)
