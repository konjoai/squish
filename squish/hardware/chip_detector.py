"""
ChipDetector — Apple Silicon M1–M5 Hardware Detection & Adaptive Tuning.

Key motivation (from research briefing, March 2026):
  • M5 provides 153 GB/s memory bandwidth vs 100 GB/s for M3 — 53% more.
  • M5 Neural Accelerators give up to 4× TTFT improvement over M4 with MLX.
  • Squish's custom Rust/PyO3 kernels are benchmarked on M1–M3; newer chips
    should not be artificially constrained by conservative defaults.

Approach:
  1. Read chip info from `sysctl hw.model` and `system_profiler`.
  2. Select the matching ChipProfile from the static table.
  3. Expose recommended_chunk_prefill, recommended_kv_bits, etc. so the
     server wiring layer can apply chip-appropriate defaults without
     hard-coding generation-specific values.
"""
from __future__ import annotations

import json
import pathlib
import platform
import re
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# On-disk chip-detection cache
# ---------------------------------------------------------------------------
# Avoids the slow `system_profiler` fallback on subsequent process starts.
# Cache key = platform.machine() + platform.version() — static per OS install.
# Invalidated automatically when either changes (e.g. after macOS upgrade).

_DISK_CACHE_PATH: pathlib.Path = pathlib.Path.home() / ".squish" / "hw_cache.json"


def _disk_cache_key() -> str:
    return platform.machine() + platform.version()


def _load_disk_cache() -> str | None:
    """Return cached chip string if cache key matches, else None."""
    try:
        data = json.loads(_DISK_CACHE_PATH.read_text())
        if data.get("key") == _disk_cache_key():
            return data.get("chip_str", "")
    except Exception:
        pass
    return None


def _save_disk_cache(chip_str: str) -> None:
    """Persist chip string to ~/.squish/hw_cache.json (best-effort)."""
    try:
        _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DISK_CACHE_PATH.write_text(
            json.dumps({"key": _disk_cache_key(), "chip_str": chip_str})
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chip generation enum
# ---------------------------------------------------------------------------

class AppleChipGeneration(IntEnum):
    UNKNOWN = 0
    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5


# ---------------------------------------------------------------------------
# Per-chip profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipProfile:
    """Hardware capability constants for a specific Apple Silicon generation."""
    generation: AppleChipGeneration
    memory_bandwidth_gbps: float    # peak memory bandwidth
    neural_engine_tops: float       # Neural Engine tera-ops/sec
    max_memory_gb: int              # maximum unified memory supported
    recommended_chunk_prefill: int  # chunked-prefill token budget (throughput-optimised)
    recommended_kv_bits: int        # KV cache quantization bits (4 or 8)
    mlx_dispatch_unrestricted: bool = True  # True → do not override MLX dispatch
    # Wave 81: blazing-mode fields
    recommended_model_bits: int = 4         # weight quantization bits for optimal speed/quality
    recommended_chunk_prefill_ttft: int = 128  # TTFT-optimised chunk size (smaller → faster first token)


# ---------------------------------------------------------------------------
# Static profile table
# ---------------------------------------------------------------------------

CHIP_PROFILES: Dict[AppleChipGeneration, ChipProfile] = {
    AppleChipGeneration.M1: ChipProfile(
        generation=AppleChipGeneration.M1,
        memory_bandwidth_gbps=68.25,
        neural_engine_tops=11.0,
        max_memory_gb=16,
        recommended_chunk_prefill=512,
        recommended_kv_bits=8,
        recommended_model_bits=4,
        recommended_chunk_prefill_ttft=64,
    ),
    AppleChipGeneration.M2: ChipProfile(
        generation=AppleChipGeneration.M2,
        memory_bandwidth_gbps=100.0,
        neural_engine_tops=15.8,
        max_memory_gb=24,
        recommended_chunk_prefill=768,
        recommended_kv_bits=8,
        recommended_model_bits=4,
        recommended_chunk_prefill_ttft=64,
    ),
    AppleChipGeneration.M3: ChipProfile(
        generation=AppleChipGeneration.M3,
        memory_bandwidth_gbps=100.0,
        neural_engine_tops=18.0,
        max_memory_gb=24,
        # M3 16GB: INT2 keeps 7B model at ~2GB, leaving 14GB headroom.
        # M3 24GB: INT3 gives better quality with enough headroom.
        # We default to INT2 because 16GB is the most common M3 config;
        # get_recommended_model_bits() refines based on actual RAM.
        recommended_chunk_prefill=1024,
        recommended_kv_bits=4,
        recommended_model_bits=2,
        recommended_chunk_prefill_ttft=128,
    ),
    AppleChipGeneration.M4: ChipProfile(
        generation=AppleChipGeneration.M4,
        memory_bandwidth_gbps=120.0,
        neural_engine_tops=38.0,
        max_memory_gb=32,
        # M4 16GB: INT2 again optimal; 24GB+: INT3 is fine.
        recommended_chunk_prefill=1536,
        recommended_kv_bits=4,
        recommended_model_bits=2,
        recommended_chunk_prefill_ttft=128,
    ),
    AppleChipGeneration.M5: ChipProfile(
        generation=AppleChipGeneration.M5,
        memory_bandwidth_gbps=153.0,
        neural_engine_tops=58.0,
        max_memory_gb=64,
        # M5 has 64GB max — quality over size; INT4 preserves more accuracy.
        recommended_chunk_prefill=2048,
        recommended_kv_bits=4,
        recommended_model_bits=4,
        recommended_chunk_prefill_ttft=256,
    ),
}

# Fallback for unknown chips — use conservative M3-equivalent settings
_UNKNOWN_PROFILE = ChipProfile(
    generation=AppleChipGeneration.UNKNOWN,
    memory_bandwidth_gbps=100.0,
    neural_engine_tops=16.0,
    max_memory_gb=16,
    recommended_chunk_prefill=512,
    recommended_kv_bits=8,
    recommended_model_bits=4,
    recommended_chunk_prefill_ttft=128,
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ChipDetector:
    """Detect the running Apple Silicon generation and provide tuning parameters.

    Non-Apple platforms return the UNKNOWN profile — all logic degrades
    gracefully to safe defaults.
    """

    def __init__(self, _override: Optional[str] = None) -> None:
        """
        Args:
            _override: chip string override for testing (e.g. "Apple M4 Pro").
            Not for production use.
        """
        self._override = _override
        self._profile: Optional[ChipProfile] = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self) -> ChipProfile:
        """Detect chip and return its profile (cached after first call)."""
        if self._profile is not None:
            return self._profile
        chip_str = self._read_chip_string()
        gen = self._parse_generation(chip_str)
        self._profile = CHIP_PROFILES.get(gen, _UNKNOWN_PROFILE)
        return self._profile

    def _read_chip_string(self) -> str:
        """Read raw chip description from the OS.

        Order of precedence:
          1. ``_override`` (test injection only)
          2. ``~/.squish/hw_cache.json`` (persisted from a prior run)
          3. ``sysctl -n machdep.cpu.brand_string`` (fast, < 1 ms)
          4. ``system_profiler SPHardwareDataType`` (slow, 1-4 s fallback)
        """
        if self._override is not None:
            return self._override
        if platform.system() != "Darwin":
            return ""

        # Fast path: disk cache hit — avoids all subprocess calls.
        cached = _load_disk_cache()
        if cached is not None:
            return cached

        chip_str = ""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                chip_str = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if not chip_str:
            try:
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "Chip" in line or "Processor" in line:
                            chip_str = line
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        _save_disk_cache(chip_str)
        return chip_str

    @staticmethod
    def _parse_generation(chip_str: str) -> AppleChipGeneration:
        """Parse chip generation from a string like 'Apple M3 Pro'."""
        s = chip_str.upper()
        # Match "M5", "M4", "M3", "M2", "M1" — longest first to avoid M1 matching M10
        for gen_val, gen_enum in [
            (5, AppleChipGeneration.M5),
            (4, AppleChipGeneration.M4),
            (3, AppleChipGeneration.M3),
            (2, AppleChipGeneration.M2),
            (1, AppleChipGeneration.M1),
        ]:
            # Match " M5", " M5 ", "M5 ", but NOT M55 or M15
            pattern = rf"\bM{gen_val}\b"
            if re.search(pattern, s):
                return gen_enum
        return AppleChipGeneration.UNKNOWN

    # ------------------------------------------------------------------
    # Tuning helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # RAM detection (Wave 81)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_ram_gb() -> float:
        """Return total system RAM in GB by reading ``sysctl hw.physmem``.

        Returns 0.0 on non-Darwin platforms or when the sysctl call fails.
        """
        if platform.system() != "Darwin":
            return 0.0
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.physmem"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            pass
        return 0.0

    def get_recommended_model_bits(self, ram_gb: float, model_params_b: float) -> int:
        """Return the recommended weight quantization bit-width for this chip+RAM.

        Rules (Wave 81 M3-16GB Sprint):
        - 7-8B model on ≤16 GB → INT2 (~2GB weights, 14GB headroom)
        - 7-8B model on 24 GB  → INT3 (~2.6GB weights, 21GB headroom)
        - 7-8B model on ≥32 GB → INT4 (quality headroom)
        - <4B  model            → INT4 regardless (already small enough)
        - Honour chip's recommended_model_bits as the floor.

        Parameters
        ----------
        ram_gb : total system RAM in GB (0.0 = unknown, use chip default)
        model_params_b : model parameter count in billions
        """
        profile = self.detect()
        chip_default = profile.recommended_model_bits

        if ram_gb <= 0.0:
            return chip_default

        if model_params_b < 4.0:
            # Small models fit easily in INT4 on any M-series chip.
            return 4

        if ram_gb <= 16.0:
            return 2
        if ram_gb <= 20.0:
            return 3
        if ram_gb <= 28.0:
            return 4
        # ≥32 GB: quality wins; use chip's recommendation (usually 4)
        return chip_default

    def get_optimal_chunk_size(self, model_size_gb: float) -> int:
        """Return recommended chunk-prefill token count for a given model size.

        Larger models consume more memory per token → smaller chunks to avoid
        peak-memory spikes. Scale down linearly once model > 8 GB.
        """
        profile = self.detect()
        base = profile.recommended_chunk_prefill
        if model_size_gb <= 8.0:
            return base
        scale = max(0.25, 8.0 / model_size_gb)
        return max(128, int(base * scale))

    def get_ttft_chunk_size(self, model_size_gb: float) -> int:
        """Return the TTFT-optimised chunk-prefill size for this chip.  Smaller
        than ``get_optimal_chunk_size`` — yields the first token after fewer
        prompt tokens processed, at the cost of lower total prefill throughput.
        """
        profile = self.detect()
        base = profile.recommended_chunk_prefill_ttft
        if model_size_gb > 8.0:
            # Very large models need even smaller chunks to stay under peak memory.
            base = max(64, base // 2)
        return base

    def get_recommended_kv_bits(self, available_ram_gb: float) -> int:
        """Return recommended KV cache quantization bits.

        If available RAM is below 12 GB we drop to 4-bit regardless of chip.
        """
        profile = self.detect()
        if available_ram_gb < 12.0:
            return 4
        return profile.recommended_kv_bits

    def should_enable_metal_dispatch(self) -> bool:
        """True → do not add any flags that restrict MLX's Metal dispatch."""
        profile = self.detect()
        return profile.mlx_dispatch_unrestricted

    def bandwidth_ratio_vs_m3(self) -> float:
        """Memory bandwidth relative to M3 baseline (useful for scaling estimates)."""
        profile = self.detect()
        m3_bw = CHIP_PROFILES[AppleChipGeneration.M3].memory_bandwidth_gbps
        return profile.memory_bandwidth_gbps / m3_bw

    def __repr__(self) -> str:
        profile = self.detect()
        return (
            f"ChipDetector(gen={profile.generation.name}, "
            f"bw={profile.memory_bandwidth_gbps:.0f}GB/s, "
            f"chunk={profile.recommended_chunk_prefill})"
        )
