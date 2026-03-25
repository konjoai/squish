"""squish/serving/blazing.py — Blazing-mode configuration helpers.

This module provides the public API for blazing-mode (sub-3s TTFT) detection
and configuration.  The actual implementations live in ``server.py`` (Wave 81);
this module re-exports them so that external code and tests can import from a
stable location without pulling in all of ``server.py``.

Public API
──────────
CHIP_FAMILIES_BLAZING  — set of chip family strings that support auto-blazing
auto_blazing_eligible  — return True if the chip qualifies for auto-enable
BlazingPreset          — dataclass with per-chip preset values
get_preset             — return a :class:`BlazingPreset` for a chip/RAM combo
"""
from __future__ import annotations

__all__ = [
    "CHIP_FAMILIES_BLAZING",
    "auto_blazing_eligible",
    "BlazingPreset",
    "get_preset",
]

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Chip eligibility for auto-blazing
# ---------------------------------------------------------------------------

#: Apple Silicon chips that qualify for auto-enabled blazing mode.
CHIP_FAMILIES_BLAZING: frozenset[str] = frozenset({
    "m3", "m3 pro", "m3 max", "m3 ultra",
    "m4", "m4 pro", "m4 max", "m4 ultra",
    "m5", "m5 pro", "m5 max", "m5 ultra",
})


def auto_blazing_eligible(chip_name: str, ram_gb: float) -> bool:
    """Return ``True`` if the chip should have blazing mode auto-enabled.

    Requires:
    - Chip family in :data:`CHIP_FAMILIES_BLAZING` (M3 or newer generation)
    - At least 16 GB unified memory

    Parameters
    ----------
    chip_name:
        Chip identification string, e.g. ``"Apple M3 Pro"`` (case-insensitive).
    ram_gb:
        Total unified memory in gigabytes.

    Returns
    -------
    bool
    """
    name_lower = chip_name.lower()
    eligible_chip = any(f in name_lower for f in CHIP_FAMILIES_BLAZING)
    return eligible_chip and ram_gb >= 16.0


# ---------------------------------------------------------------------------
# BlazingPreset — per-chip/RAM tuning values
# ---------------------------------------------------------------------------

@dataclass
class BlazingPreset:
    """Tuning knobs applied when blazing mode is active.

    Attributes
    ----------
    quant_bits:
        Weight quantization bits for the KV cache (2 or 4).
    chunk_prefill_size:
        Tokenizer chunk size for TTFT-optimised prefill.
    max_kv_size:
        Clamped KV context window size (tokens).
    metal_cache_limit_mb:
        Metal allocator pool cap in MB.
    fast_gelu:
        Use fast-GELU approximation.
    note:
        Human-readable description of the preset.
    """
    quant_bits:           int   = 4
    chunk_prefill_size:   int   = 128
    max_kv_size:          int   = 4096
    metal_cache_limit_mb: int   = 64
    fast_gelu:            bool  = True
    note:                 str   = ""


def get_preset(chip_name: str = "", ram_gb: float = 0.0) -> BlazingPreset:
    """Return the best :class:`BlazingPreset` for the given chip and RAM.

    Parameters
    ----------
    chip_name:
        Chip identification string (case-insensitive).
    ram_gb:
        Total unified memory in GB.

    Returns
    -------
    :class:`BlazingPreset`
        A preset appropriate for the hardware.  Falls back to conservative
        defaults when chip_name is unrecognised or ram_gb is unknown (0).
    """
    name_lower = chip_name.lower()

    # 70B+ models on 24 GB+ systems — use INT4 (better quality than INT2)
    is_large_model_capable = ram_gb >= 24
    is_m4_or_newer = any(f in name_lower for f in (
        "m4", "m5",
    ))

    if is_large_model_capable and is_m4_or_newer:
        return BlazingPreset(
            quant_bits=4,
            chunk_prefill_size=256,
            max_kv_size=8192,
            metal_cache_limit_mb=128,
            fast_gelu=True,
            note="M4/M5 + 24 GB: INT4 KV, large context",
        )

    if is_large_model_capable:
        return BlazingPreset(
            quant_bits=4,
            chunk_prefill_size=128,
            max_kv_size=4096,
            metal_cache_limit_mb=96,
            fast_gelu=True,
            note="24 GB: INT4 KV, standard context",
        )

    # Default 16 GB M3+ preset
    return BlazingPreset(
        quant_bits=2,
        chunk_prefill_size=128,
        max_kv_size=4096,
        metal_cache_limit_mb=64,
        fast_gelu=True,
        note="16 GB M3+: INT2 KV, tight Metal pool",
    )
