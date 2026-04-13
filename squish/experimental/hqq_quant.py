"""Deprecated — use squish.quant.hqq instead.

This module is kept for backward compatibility only.  All symbols have moved
to the production path at :mod:`squish.quant.hqq`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "squish.experimental.hqq_quant is deprecated; use squish.quant.hqq instead.",
    DeprecationWarning,
    stacklevel=2,
)

from squish.quant.hqq import HQQConfig, HQQQuantizer, HQQTensor  # noqa: F401, E402

__all__ = [
    "HQQConfig",
    "HQQTensor",
    "HQQQuantizer",
]
