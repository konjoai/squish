"""KernelDispatcher — format-aware Metal kernel selector for Squish Wave 67.

Reads the :class:`~squish.format.squish_header.SquizdFlag` bits from a model
header and the :class:`~squish.hardware.capability_probe.HardwareCapabilities`
from the hardware probe to deterministically choose the best inference kernel
for the running configuration.

The selection result, :class:`KernelDispatch`, is a frozen dataclass
containing the kernel name, the shader path relative to ``squish/kernels/``,
whether the kernel supports batched token generation (i.e., multiple input
vectors per call), and which inference phase it is optimised for.

Typical usage::

    from squish.hardware.kernel_dispatch import get_kernel_dispatcher

    dispatcher = get_kernel_dispatcher()
    dispatch   = dispatcher.select(header.flags, caps, seq_len=1)
    print(dispatch.kernel_name)  # e.g. "fused_int4_gemv"

Design
──────
Priority order (highest → lowest):

  1. ASTC             → astc_gemv           (Wave 64 — hardware texture)
  2. TCA_TBE, decode  → zip_gemv            (Wave 65 — lossless bitmap decode)
  2. TCA_TBE, prefill → zip_gemm            (Wave 65 — lossless bitmap prefill)
  3. INT4 + SPARSE    → sparse_gemv         (Wave 66 — structured sparsity)
  4. INT4,  seq_len=1 → fused_int4_gemv     (Wave 67 — fused INT4 decode)
  4. INT4,  seq_len>1 → fused_int4_gemm     (Wave 67 — tiled INT4 prefill)
  5. INT2             → lut_int2_gemv       (Wave 67 — INT2 LUT-GEMM)
  6. (no match)       → legacy_dequant_matmul

Thread safety
─────────────
:func:`get_kernel_dispatcher` returns a process-global singleton initialised
lazily on first call.  :func:`reset_kernel_dispatcher` clears the singleton
(for use in tests only — do not call in production).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from squish.format.squish_header import SquizdFlag
from squish.hardware.capability_probe import HardwareCapabilities

__all__ = [
    "KernelDispatch",
    "KernelDispatcher",
    "get_kernel_dispatcher",
    "reset_kernel_dispatcher",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelDispatch:
    """Resolved kernel selection for a given model + hardware combination.

    Attributes:
        kernel_name: Short, stable identifier for the kernel function.
            Matches the Metal ``kernel`` function name in the shader file.
        metal_shader_path: Path to the ``.metal`` source file, relative to the
            ``squish/kernels/`` directory.  Empty string for the legacy path
            which does not use a Metal compute kernel.
        supports_batched: ``True`` if the shader exposes a ``_batched``
            variant that accepts multiple input vectors in one dispatch.
        phase: Intended inference phase.  One of ``"decode"``, ``"prefill"``,
            or ``"both"``.
    """

    kernel_name: str
    metal_shader_path: str
    supports_batched: bool
    phase: str

    def __post_init__(self) -> None:
        valid_phases = {"decode", "prefill", "both"}
        if self.phase not in valid_phases:
            raise ValueError(
                f"KernelDispatch.phase must be one of {sorted(valid_phases)!r}, "
                f"got {self.phase!r}"
            )


# ---------------------------------------------------------------------------
# Pre-built KernelDispatch instances (avoids constructing them on every call)
# ---------------------------------------------------------------------------

_ASTC_GEMV = KernelDispatch(
    kernel_name="astc_gemv",
    metal_shader_path="astc_gemv.metal",
    supports_batched=True,
    phase="both",
)

_ZIP_GEMV = KernelDispatch(
    kernel_name="zip_gemv",
    metal_shader_path="zip_gemv.metal",
    supports_batched=True,
    phase="decode",
)

_ZIP_GEMM = KernelDispatch(
    kernel_name="zip_gemm",
    metal_shader_path="zip_gemm.metal",
    supports_batched=False,
    phase="prefill",
)

_SPARSE_GEMV = KernelDispatch(
    kernel_name="sparse_gemv",
    metal_shader_path="sparse_gemv.metal",
    supports_batched=True,
    phase="both",
)

_FUSED_INT4_GEMV = KernelDispatch(
    kernel_name="fused_int4_gemv",
    metal_shader_path="fused_int4_gemv.metal",
    supports_batched=True,
    phase="decode",
)

_FUSED_INT4_GEMM = KernelDispatch(
    kernel_name="fused_int4_gemm",
    metal_shader_path="fused_int4_gemm.metal",
    supports_batched=False,
    phase="prefill",
)

_LUT_INT2_GEMV = KernelDispatch(
    kernel_name="lut_int2_gemv",
    metal_shader_path="lut_int2_gemv.metal",
    supports_batched=True,
    phase="both",
)

_LEGACY = KernelDispatch(
    kernel_name="legacy_dequant_matmul",
    metal_shader_path="",
    supports_batched=False,
    phase="both",
)


# ---------------------------------------------------------------------------
# KernelDispatcher
# ---------------------------------------------------------------------------

class KernelDispatcher:
    """Selects the optimal Metal inference kernel for a given SQUIZD model.

    Kernels are selected based on the combination of :class:`SquizdFlag` bits
    from the model header and the :class:`HardwareCapabilities` of the running
    machine.

    The dispatcher maintains an internal LRU-free dict cache keyed by
    ``(flags, seq_len)`` so repeated calls for the same model are O(1)
    dictionary lookups with no recomputation.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[SquizdFlag, int], KernelDispatch] = {}

    def select(
        self,
        flags: SquizdFlag,
        caps: HardwareCapabilities,  # noqa: ARG002 — reserved for future cap-gating
        *,
        seq_len: int = 1,
    ) -> KernelDispatch:
        """Return the :class:`KernelDispatch` for *flags* at *seq_len*.

        Args:
            flags:   SQUIZD feature flags read from the model header.
            caps:    Hardware capabilities from :func:`.get_capability_probe`.
                     Currently used as a future extension point — all current
                     kernels run on any M-series chip.  Reserved for M5-only
                     MXFP4 or ANE-specific paths.
            seq_len: Number of tokens being processed in this forward pass.
                     Use ``1`` for decode/generation (default) and ``>1`` for
                     prefill.  Affects whether GEMV or GEMM variant is chosen.

        Returns:
            A frozen :class:`KernelDispatch` describing the best kernel.

        Raises:
            ValueError: If *seq_len* is less than 1.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len!r}")

        cache_key = (flags, seq_len)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._resolve(flags, seq_len)
        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(flags: SquizdFlag, seq_len: int) -> KernelDispatch:
        """Apply the priority decision table and return a :class:`KernelDispatch`."""
        # Priority 1 — ASTC hardware texture path (highest)
        if flags.has(SquizdFlag.ASTC):
            return _ASTC_GEMV

        # Priority 2 — TCA-TBE lossless bitmap
        if flags.has(SquizdFlag.TCA_TBE):
            return _ZIP_GEMV if seq_len == 1 else _ZIP_GEMM

        # Priority 3 — INT4 + SPARSE structured sparsity
        if flags.has(SquizdFlag.INT4) and flags.has(SquizdFlag.SPARSE):
            return _SPARSE_GEMV

        # Priority 4 — INT4 fused (no BF16 staging buffer)
        if flags.has(SquizdFlag.INT4):
            return _FUSED_INT4_GEMV if seq_len == 1 else _FUSED_INT4_GEMM

        # Priority 5 — INT2 LUT-GEMM
        if flags.has(SquizdFlag.INT2):
            return _LUT_INT2_GEMV

        # Fallback — legacy dequantisation path
        return _LEGACY


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DISPATCHER: Optional[KernelDispatcher] = None


def get_kernel_dispatcher() -> KernelDispatcher:
    """Return the process-global :class:`KernelDispatcher` singleton.

    Initialises lazily on first call.  Thread-safe under CPython's GIL for
    typical ML serving workloads (single-threaded model loading).
    """
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = KernelDispatcher()
    return _DISPATCHER


def reset_kernel_dispatcher() -> None:
    """Reset the global singleton.

    Intended **only** for use in unit tests.  Do not call in production code.
    """
    global _DISPATCHER
    _DISPATCHER = None
