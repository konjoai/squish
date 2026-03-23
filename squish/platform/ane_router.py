# [Experimental] This module is part of Squish v43+ (Wave 69).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""ANERouter — Apple Neural Engine availability detection and routing policy.

Determines whether a given model should be dispatched to the Apple Neural
Engine (ANE) or the Metal GPU path based on chip generation, model size,
and user configuration.

Rules
─────
1. Non-macOS platforms (Linux, Windows) always return ``"gpu"``.
2. ``SQUISH_ANE_ENABLED=0`` env var disables ANE; ``=1`` forces it (up to the
   8B cap).
3. Models with more than 8 billion parameters always route to ``"gpu"``
   (ANE memory budget cannot accommodate them in one or two-chunk passes).
4. Chip generation M1–M5 is detected via :mod:`squish.hardware.chip_detector`;
   unknown/fallback chips default to ``"gpu"``.
5. Results are cached to ``~/.squish/hardware_caps.json`` for subsequent
   process launches (shared with ``capability_probe.py``).

Usage::

    from squish.platform.ane_router import get_ane_router

    router = get_ane_router()
    backend = router.route(param_count=3_800_000_000)  # "ane" | "gpu"
    print(router.policy.chip_generation)               # 3  (M3)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

# Inline-import guard so the module imports cleanly on Linux/Windows
try:
    from squish.hardware.chip_detector import (
        AppleChipGeneration,
        ChipDetector,
        ChipProfile,
    )
    _CHIP_DETECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CHIP_DETECTOR_AVAILABLE = False

__all__ = [
    "ANERoutingPolicy",
    "ANERouter",
    "get_ane_router",
    "reset_ane_router",
    "ANE_PARAM_LIMIT",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Models larger than this many parameters always go to the GPU path.
ANE_PARAM_LIMIT: int = 8_000_000_000

# ANE memory budget per chunk (conservative; CoreML allocator may use more).
_ANE_MEMORY_BUDGET_GB_DEFAULT: float = 4.0  # GB, per ANE chunk

# Per-chip ANE memory budgets (GB) based on empirical CoreML limits.
_ANE_MEMORY_BUDGET_BY_GEN: Dict[int, float] = {
    1: 2.0,   # M1 — 2 GB ANE accessible memory
    2: 2.0,   # M2
    3: 4.0,   # M3
    4: 4.0,   # M4
    5: 8.0,   # M5 — Neural Accelerators extend budget
}

_DEFAULT_CAPS_PATH = Path.home() / ".squish" / "hardware_caps.json"


# ---------------------------------------------------------------------------
# ANERoutingPolicy — frozen result of a routing decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ANERoutingPolicy:
    """Immutable result of an ANE routing decision.

    Attributes:
        enabled: True if ANE is the preferred backend for the given
            ``param_count`` that produced this policy.
        reason: Human-readable justification string.
        preferred_backend: ``"ane"`` or ``"gpu"``.
        chip_generation: Integer chip generation (1=M1 … 5=M5; 0=unknown).
        ane_memory_budget_gb: Estimated ANE memory available per chunk (GB).
    """

    enabled: bool
    reason: str
    preferred_backend: str          # "ane" | "gpu"
    chip_generation: int            # 0 = unknown
    ane_memory_budget_gb: float


# ---------------------------------------------------------------------------
# ANERouter
# ---------------------------------------------------------------------------

class ANERouter:
    """Detect ANE availability and route model inference to ANE or GPU.

    Parameters:
        _detector_override: Inject a custom :class:`ChipDetector` instance for
            testing.  Production code uses the default auto-detecting instance.
        _caps_path: Override the hardware caps cache path (testing only).
    """

    def __init__(
        self,
        _detector_override: Optional[Any] = None,
        _caps_path: Optional[Path] = None,
    ) -> None:
        self._caps_path: Path = _caps_path or _DEFAULT_CAPS_PATH
        self._lock = Lock()
        self._cached_profile: Optional[Any] = None
        self._caps_loaded: bool = False

        if _detector_override is not None:
            self._detector = _detector_override
        elif _CHIP_DETECTOR_AVAILABLE:
            self._detector = ChipDetector()
        else:
            self._detector = None

        # Eagerly detect so .policy is immediately available.
        self._chip_generation: int = self._detect_chip_generation()
        self._ane_budget_gb: float = _ANE_MEMORY_BUDGET_BY_GEN.get(
            self._chip_generation, _ANE_MEMORY_BUDGET_GB_DEFAULT
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, param_count: int) -> str:
        """Return the preferred backend (``"ane"`` or ``"gpu"``) for *param_count*.

        Args:
            param_count: Total number of model parameters (e.g. 3_800_000_000).

        Returns:
            ``"ane"`` if ANE inference is preferred; ``"gpu"`` otherwise.
        """
        return self._build_policy(param_count).preferred_backend

    def is_ane_available(self) -> bool:
        """Return True if ANE is available for *any* supported model size."""
        if not self._is_macos():
            return False
        env_val = os.environ.get("SQUISH_ANE_ENABLED", "").strip()
        if env_val == "0":
            return False
        if self._chip_generation == 0:
            return False
        return True

    def get_policy(self, param_count: int) -> ANERoutingPolicy:
        """Return the full :class:`ANERoutingPolicy` for *param_count*."""
        return self._build_policy(param_count)

    def cache_caps(self, path: Optional[Path] = None) -> None:
        """Persist detected hardware capabilities to a JSON file.

        Args:
            path: Override the target path.  Defaults to
                ``~/.squish/hardware_caps.json``.
        """
        target = path or self._caps_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "chip_generation": self._chip_generation,
            "ane_budget_gb": self._ane_budget_gb,
            "ane_available": self.is_ane_available(),
        }
        target.write_text(json.dumps(payload, indent=2))

    def load_caps(self, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load previously cached hardware capabilities from disk.

        Returns:
            Parsed capability dict or ``None`` if the file does not exist.
        """
        target = path or self._caps_path
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_macos() -> bool:
        return sys.platform == "darwin"

    def _detect_chip_generation(self) -> int:
        """Return chip generation integer; 0 on non-Apple or failure."""
        if not self._is_macos():
            return 0
        if self._detector is None:
            return 0
        try:
            profile: Any = self._detector.detect()
            gen = profile.generation
            # Support both IntEnum and plain int
            return int(gen)
        except Exception:  # pylint: disable=broad-except
            return 0

    def _build_policy(self, param_count: int) -> ANERoutingPolicy:
        gen = self._chip_generation
        budget = self._ane_budget_gb

        # --- Non-Apple ---
        if not self._is_macos():
            return ANERoutingPolicy(
                enabled=False,
                reason="non-macOS platform; ANE unavailable",
                preferred_backend="gpu",
                chip_generation=gen,
                ane_memory_budget_gb=budget,
            )

        # --- Env override: disabled ---
        env_val = os.environ.get("SQUISH_ANE_ENABLED", "").strip()
        if env_val == "0":
            return ANERoutingPolicy(
                enabled=False,
                reason="SQUISH_ANE_ENABLED=0; ANE disabled by environment",
                preferred_backend="gpu",
                chip_generation=gen,
                ane_memory_budget_gb=budget,
            )

        # --- Unknown chip ---
        if gen == 0:
            return ANERoutingPolicy(
                enabled=False,
                reason="unknown chip generation; defaulting to GPU",
                preferred_backend="gpu",
                chip_generation=gen,
                ane_memory_budget_gb=budget,
            )

        # --- Model > 8B ---
        if param_count > ANE_PARAM_LIMIT:
            return ANERoutingPolicy(
                enabled=False,
                reason=(
                    f"model has {param_count:,} parameters (> {ANE_PARAM_LIMIT:,}); "
                    "ANE only supports models ≤ 8B on current hardware"
                ),
                preferred_backend="gpu",
                chip_generation=gen,
                ane_memory_budget_gb=budget,
            )

        # --- Env override: forced ---
        if env_val == "1":
            return ANERoutingPolicy(
                enabled=True,
                reason="SQUISH_ANE_ENABLED=1; ANE forced by environment",
                preferred_backend="ane",
                chip_generation=gen,
                ane_memory_budget_gb=budget,
            )

        # --- Default: ANE preferred for supported chips + model ---
        return ANERoutingPolicy(
            enabled=True,
            reason=(
                f"M{gen} chip with {budget:.1f} GB ANE budget; "
                f"{param_count / 1e9:.2f}B param model; routing to ANE"
            ),
            preferred_backend="ane",
            chip_generation=gen,
            ane_memory_budget_gb=budget,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router_instance: Optional[ANERouter] = None
_router_lock = Lock()


def get_ane_router() -> ANERouter:
    """Return the process-level singleton :class:`ANERouter`.

    The singleton is created lazily on first call and reused thereafter.
    Use :func:`reset_ane_router` in tests to force re-detection.
    """
    global _router_instance
    with _router_lock:
        if _router_instance is None:
            _router_instance = ANERouter()
        return _router_instance


def reset_ane_router() -> None:
    """Clear the singleton (for testing only)."""
    global _router_instance
    with _router_lock:
        _router_instance = None
