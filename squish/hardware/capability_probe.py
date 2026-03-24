# [Experimental] This module is part of Squish v44+ (Wave 70).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""CapabilityProbe — Apple Silicon hardware capability detection.

Detects M1–M5 chip generation and probes for specific hardware features
required by the SQUIZD runtime:

  * ASTC texture sampling — available on Apple GPU from A8 / M1+
  * ANE availability     — confirmed via ``system_profiler SPHardwareDataType``
  * Metal 3 features     — available on M2+ (Metal GPU Family Apple 9)
  * MXFP4 precision      — available on M5+ hardware

Results are cached to ``~/.squish/hardware_caps.json`` so subsequent calls
are instant.  Pass *force_refresh=True* to bypass the cache.

Usage::

    from squish.hardware.capability_probe import get_capability_probe

    probe = get_capability_probe()
    caps  = probe.probe()
    print(caps.has_metal3, caps.chip_generation)
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from squish.hardware.chip_detector import AppleChipGeneration, ChipDetector

__all__ = [
    "HardwareCapabilities",
    "CapabilityProbe",
    "get_capability_probe",
]

_DEFAULT_CACHE: Path = Path.home() / ".squish" / "hardware_caps.json"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareCapabilities:
    """Probed hardware capabilities relevant to the SQUIZD runtime.

    Attributes:
        chip_generation: Numeric Apple Silicon generation (1 = M1 … 5 = M5).
        has_astc_texture_sampling: True if the GPU supports ASTC sampling in
            the Metal shader pipeline (required for the ASTC kernel path).
        has_ane: True if the Apple Neural Engine is present and usable.
        has_metal3: True if the GPU supports the Metal GPU Family Apple 9
            (M2 Pro / M3 / M4 / M5 and later).
        has_mxfp4: True if the chip supports native MXFP4 matrix operations
            (currently M5+ only).
        ane_memory_budget_gb: Estimated ANE tensor-memory budget in GB.
    """

    chip_generation: int
    has_astc_texture_sampling: bool
    has_ane: bool
    has_metal3: bool
    has_mxfp4: bool
    ane_memory_budget_gb: float

    @classmethod
    def _from_generation(cls, gen: AppleChipGeneration) -> "HardwareCapabilities":
        """Derive capabilities from a known chip generation.

        All M1+ chips support ASTC texture sampling.  Metal 3 features first
        appeared on M2 Pro.  MXFP4 is M5+.
        """
        g = int(gen)
        return cls(
            chip_generation=g,
            has_astc_texture_sampling=g >= int(AppleChipGeneration.M1),
            has_ane=g >= int(AppleChipGeneration.M1),
            has_metal3=g >= int(AppleChipGeneration.M2),
            has_mxfp4=g >= int(AppleChipGeneration.M5),
            ane_memory_budget_gb=_ane_budget(g),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareCapabilities":
        """Reconstruct from a plain dict (e.g. loaded from JSON cache)."""
        return cls(
            chip_generation=int(d["chip_generation"]),
            has_astc_texture_sampling=bool(d["has_astc_texture_sampling"]),
            has_ane=bool(d["has_ane"]),
            has_metal3=bool(d["has_metal3"]),
            has_mxfp4=bool(d["has_mxfp4"]),
            ane_memory_budget_gb=float(d["ane_memory_budget_gb"]),
        )


# ---------------------------------------------------------------------------
# Per-generation ANE memory budget heuristic (GB)
# ---------------------------------------------------------------------------

def _ane_budget(gen: int) -> float:
    """Return a conservative ANE tensor memory budget estimate."""
    return {1: 4.0, 2: 8.0, 3: 12.0, 4: 16.0, 5: 24.0}.get(gen, 0.0)


# ---------------------------------------------------------------------------
# CapabilityProbe
# ---------------------------------------------------------------------------

class CapabilityProbe:
    """Probes Apple Silicon hardware for SQUIZD-relevant capabilities.

    Parameters:
        cache_path: Path to the JSON cache file.  Defaults to
            ``~/.squish/hardware_caps.json``.
    """

    def __init__(self, cache_path: Path = _DEFAULT_CACHE) -> None:
        self._cache_path = cache_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self, *, force_refresh: bool = False) -> HardwareCapabilities:
        """Return :class:`HardwareCapabilities`, using or updating the cache.

        Args:
            force_refresh: If ``True``, skip the cache and re-probe hardware.

        Returns:
            A :class:`HardwareCapabilities` dataclass.
        """
        if not force_refresh:
            cached = self.load_cache()
            if cached is not None:
                return cached

        caps = self._run_probe()
        self._save_cache(caps)
        return caps

    def load_cache(self) -> Optional[HardwareCapabilities]:
        """Load cached capabilities from disk.

        Returns:
            :class:`HardwareCapabilities` if the cache exists and is valid;
            ``None`` otherwise.
        """
        if not self._cache_path.exists():
            return None
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return HardwareCapabilities.from_dict(raw)
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

    def cache(self, caps: HardwareCapabilities, path: Optional[Path] = None) -> None:
        """Write *caps* to the JSON cache at *path* (or the default path).

        Args:
            caps: Capabilities to persist.
            path: Override cache file location.
        """
        target = path or self._cache_path
        self._save_cache(caps, target)

    def invalidate_cache(self) -> None:
        """Remove the on-disk capability cache."""
        if self._cache_path.exists():
            self._cache_path.unlink()

    # ------------------------------------------------------------------
    # Internal probe logic
    # ------------------------------------------------------------------

    def _run_probe(self) -> HardwareCapabilities:
        """Execute the hardware probe and return :class:`HardwareCapabilities`."""
        gen = self._detect_generation()
        caps = HardwareCapabilities._from_generation(gen)

        # ANE confirmation via system_profiler on macOS.
        if platform.system() == "Darwin":
            ane_confirmed = self._confirm_ane()
            if ane_confirmed is not None:
                # Override the generation-derived value with the explicit reading.
                caps = HardwareCapabilities(
                    chip_generation=caps.chip_generation,
                    has_astc_texture_sampling=caps.has_astc_texture_sampling,
                    has_ane=ane_confirmed,
                    has_metal3=caps.has_metal3,
                    has_mxfp4=caps.has_mxfp4,
                    ane_memory_budget_gb=caps.ane_memory_budget_gb,
                )
        return caps

    @staticmethod
    def _detect_generation() -> AppleChipGeneration:
        """Detect the Apple Silicon generation via ChipDetector."""
        try:
            profile = ChipDetector().detect()
            return profile.generation
        except Exception:  # noqa: BLE001
            return AppleChipGeneration.UNKNOWN

    @staticmethod
    def _confirm_ane() -> Optional[bool]:
        """Return whether the Neural Engine is present via system_profiler.

        Returns ``None`` if the check cannot be completed (e.g. on non-macOS
        or if ``system_profiler`` is unavailable).
        """
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if result.returncode != 0:
                return None
            output = result.stdout
            # "Neural Engine Cores" appears in the system profiler output for
            # M-series chips.  Its absence indicates no ANE.
            return "Neural Engine" in output
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _save_cache(
        self,
        caps: HardwareCapabilities,
        path: Optional[Path] = None,
    ) -> None:
        target = path or self._cache_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(caps.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            # Cache is best-effort; failures must not break inference.
            pass


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_capability_probe(cache_path: Optional[Path] = None) -> CapabilityProbe:
    """Return a :class:`CapabilityProbe` using the given (or default) cache path.

    Args:
        cache_path: Override the default ``~/.squish/hardware_caps.json`` path.

    Returns:
        A :class:`CapabilityProbe` instance.
    """
    return CapabilityProbe(cache_path or _DEFAULT_CACHE)
