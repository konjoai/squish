"""squish/runtime/auto_profile.py — Wave 79 automatic optimization profiling.

Detects hardware capabilities and model features at startup, then returns an
:class:`OptimizationProfile` that encodes the best settings without requiring
the user to pass any flags.

Design goals
------------
* Zero required flags: every optimization is auto-detected.
* Backward compatible: existing explicit flags are never overridden.
* Fast: all detection logic is I/O-bound (config.json / file-exists); no model
  loading or network calls.
* Testable: all detection is pure-Python and works offline.

Usage (from server.py)::

    from squish.runtime.auto_profile import ModelCapabilityDetector

    detector = ModelCapabilityDetector()
    profile  = detector.detect(
        model_dir      = args.model_dir,
        compressed_dir = args.compressed_dir,
        chip_profile   = _chip_profile,   # existing ChipProfile object
        ram_gb         = ChipDetector.detect_ram_gb(),
    )
    # Apply profile — only sets flags that were not explicitly passed
    profile.apply_defaults(args)
    # Get compact status line for startup output
    print(profile.status_line(model_name, load_time_s))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "OptimizationProfile",
    "ModelCapabilityDetector",
]


# ---------------------------------------------------------------------------
# OptimizationProfile
# ---------------------------------------------------------------------------

@dataclass
class OptimizationProfile:
    """Complete set of auto-detected optimization settings for one model run.

    All fields have safe default values — a profile with no detection
    applied is always valid and results in conservative-but-correct inference.
    """

    # ── Kernel dispatch ──────────────────────────────────────────────────────
    # "lut_int2"  — 2-bit weights, 256-entry FP16 LUT in threadgroup metal kernel
    # "fused_int3"— 3-bit packed (8 weights in 3 bytes), in-register unpack
    # "fused_int4"— 4-bit fused Metal GEMV, no staging buffer
    # "astc"      — hardware ASTC texture decode on Apple GPU
    # "numpy"     — pure-NumPy reference path (CI / non-Apple)
    kernel_path: str = "fused_int4"

    # ── KV cache ─────────────────────────────────────────────────────────────
    # "int2"  — AgentKV asymmetric 2-bit (6× memory reduction)
    # "int4"  — 4-bit quantized KV (2× reduction)
    # "fp16"  — full-precision (default, safest)
    kv_mode: str = "fp16"

    # ── Speculative decoding ─────────────────────────────────────────────────
    use_eagle3: bool = False
    eagle3_head_dir: str = ""

    # ── FFN sparsity ─────────────────────────────────────────────────────────
    use_sparsity: bool = False
    sparsity_mask_path: str = ""

    # ── MoE ──────────────────────────────────────────────────────────────────
    use_moe_lazy: bool = False   # JIT expert materialization

    # ── Compute tuning ───────────────────────────────────────────────────────
    chunk_prefill_size: int = 512
    metal_cache_mb: int = 256

    # ── Status display ───────────────────────────────────────────────────────
    # Features that will appear in the compact startup status line.
    active_features: list[str] = field(default_factory=list)

    # ── Internal bookkeeping ─────────────────────────────────────────────────
    # Tracks the source of each setting for diagnostics / --trace output.
    _sources: dict[str, str] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Defaults application
    # ------------------------------------------------------------------

    def apply_defaults(self, args: Any) -> None:
        """Apply profile settings to *args* (argparse Namespace) as defaults.

        Only sets a field when the user has not already supplied an explicit
        value.  Explicit values are detected by comparing against argparse
        defaults or checking ``args._user_set_<flag>`` sentinel booleans.

        Parameters
        ----------
        args:
            Parsed argparse Namespace (or any mutable mapping-like object).
        """
        # chunk-prefill size — only override when at its argparse default of 512
        if int(getattr(args, "chunk_prefill_size", 512)) == 512:
            args.chunk_prefill_size = self.chunk_prefill_size
            self._sources["chunk_prefill_size"] = "auto"

        # Metal cache limit — only override when at its default of 256
        if int(getattr(args, "_blazing_metal_cache_mb", 256)) == 256:
            # Store on args so _cap_metal_cache() picks it up
            setattr(args, "_blazing_metal_cache_mb", self.metal_cache_mb)
            self._sources["metal_cache_mb"] = "auto"

        # EAGLE-3 head — auto-load when file is present and --eagle-head-dir
        # was not explicitly passed
        if self.use_eagle3 and not getattr(args, "eagle_head_dir", ""):
            args.eagle_head_dir = self.eagle3_head_dir
            self._sources["eagle3"] = "auto"

        # AgentKV — enable when kv_mode == "int2" and not already set
        if self.kv_mode == "int2" and not getattr(args, "agent_kv", False):
            args.agent_kv = True
            self._sources["agent_kv"] = "auto"

    # ------------------------------------------------------------------
    # Status line generation
    # ------------------------------------------------------------------

    def status_line(self, model_name: str, load_time_s: float) -> str:
        """Return a compact single-line startup status string.

        Example output::

            squish  Qwen3-8B-int2  loaded in 2.3s  [lut_int2 · eagle3 · sparse]
        """
        parts = list(self.active_features)
        features = "  [" + " · ".join(parts) + "]" if parts else ""
        return f"squish  {model_name}  loaded in {load_time_s:.1f}s{features}"


# ---------------------------------------------------------------------------
# ModelCapabilityDetector
# ---------------------------------------------------------------------------

class ModelCapabilityDetector:
    """Inspect a model directory and chip profile to produce an OptimizationProfile.

    All detection is purely file-system / config-file based — no model weights
    are loaded by this class.
    """

    def detect(
        self,
        model_dir: str,
        compressed_dir: str = "",
        chip_profile: Any = None,
        ram_gb: float = 0.0,
    ) -> OptimizationProfile:
        """Run all detection passes and return a complete :class:`OptimizationProfile`.

        Parameters
        ----------
        model_dir:
            Path to the base (BF16) model directory.
        compressed_dir:
            Path to the compressed/quantized model directory.  Defaults to
            ``"{model_dir}-compressed"`` when empty.
        chip_profile:
            :class:`~squish.hardware.chip_detector.ChipProfile` from the
            hardware detector.  Pass ``None`` on non-Apple platforms.
        ram_gb:
            Total system RAM in GB.  0.0 triggers conservative defaults.
        """
        profile = OptimizationProfile()

        model_path = Path(model_dir) if model_dir else Path()
        comp_path  = (
            Path(compressed_dir)
            if compressed_dir
            else (model_path.parent / (model_path.name + "-compressed"))
        )

        # Detection passes (order matters — later passes may override earlier)
        self._detect_hardware(profile, chip_profile, ram_gb)
        self._detect_model_config(profile, model_path)
        self._detect_eagle3(profile, model_path, comp_path)
        self._detect_sparsity(profile, comp_path)
        self._build_feature_list(profile)

        return profile

    # ------------------------------------------------------------------
    # Private detection passes
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_hardware(
        profile: OptimizationProfile,
        chip_profile: Any,
        ram_gb: float,
    ) -> None:
        """Select kernel and cache settings from chip generation and RAM."""
        if chip_profile is None:
            return

        try:
            from squish.hardware.chip_detector import AppleChipGeneration
            gen = chip_profile.generation
        except (ImportError, AttributeError):
            return

        # Chunk-prefill size: smaller chunks = lower TTFT on memory-constrained M3
        try:
            profile.chunk_prefill_size = chip_profile.recommended_chunk_prefill_ttft
        except AttributeError:
            pass

        # Metal buffer cache cap: M3/M4/M5 benefit from tighter caps
        if int(gen) >= 3:
            # Scale with RAM: 16 GB → 64 MB, 24 GB → 96 MB, 32 GB → 128 MB
            profile.metal_cache_mb = max(64, min(128, int(ram_gb * 4)))

        # KV cache quantization: INT4 on M3+ 16 GB (tight RAM)
        try:
            profile.kv_mode = (
                "int4"
                if chip_profile.recommended_kv_bits == 4
                else "fp16"
            )
        except AttributeError:
            pass

        # Kernel path: SELECT based on recommended_model_bits if available
        try:
            bits = chip_profile.recommended_model_bits
            if bits <= 2:
                profile.kernel_path = "lut_int2"
            elif bits <= 3:
                profile.kernel_path = "fused_int3"
            else:
                profile.kernel_path = "fused_int4"
        except AttributeError:
            pass

    @staticmethod
    def _detect_model_config(
        profile: OptimizationProfile,
        model_path: Path,
    ) -> None:
        """Read config.json to detect architecture capabilities."""
        config_path = model_path / "config.json"
        if not config_path.exists():
            return
        try:
            with open(config_path, encoding="utf-8") as f:
                config: dict = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        arch = ""
        archs = config.get("architectures") or []
        if archs:
            arch = archs[0].lower()

        # MoE: Qwen3Moe, DeepseekV2/V3, Mixtral, Phimoe, etc.
        _MOE_ARCH_PREFIXES = (
            "mixtral",
            "deepseekv2",
            "deepseekv3",
            "phimoe",
        )
        is_moe = (
            "moe" in arch
            or any(arch.startswith(p) for p in _MOE_ARCH_PREFIXES)
            or config.get("num_experts", 0) > 0
            or config.get("num_local_experts", 0) > 0
        )
        if is_moe:
            profile.use_moe_lazy = True

        # Dense INT2 attention fix: pure INT2 degrades multi-head attention;
        # fall back to fused_int4 kernel path for attention layers.
        # This is handled at runtime by the kernel dispatcher — no change here.

    @staticmethod
    def _detect_eagle3(
        profile: OptimizationProfile,
        model_path: Path,
        comp_path: Path,
    ) -> None:
        """Detect EAGLE-3 draft head file presence."""
        search_dirs = [
            comp_path,
            model_path,
            model_path.parent,
            comp_path.parent,
        ]
        eagle_patterns = [
            "eagle3_head.safetensors",
            "eagle3_head",
            "eagle_head.safetensors",
        ]
        for directory in search_dirs:
            if not directory.exists():
                continue
            for pattern in eagle_patterns:
                candidate = directory / pattern
                if candidate.exists():
                    profile.use_eagle3 = True
                    # Point to the directory containing the head file
                    profile.eagle3_head_dir = str(
                        candidate if candidate.is_dir() else candidate.parent
                    )
                    return

    @staticmethod
    def _detect_sparsity(
        profile: OptimizationProfile,
        comp_path: Path,
    ) -> None:
        """Detect structured FFN sparsity mask file presence."""
        mask_candidates = [
            comp_path / "sparse_masks.npz",
            comp_path / "sparsity_masks.npz",
        ]
        for candidate in mask_candidates:
            if candidate.exists():
                profile.use_sparsity = True
                profile.sparsity_mask_path = str(candidate)
                return

    @staticmethod
    def _build_feature_list(profile: OptimizationProfile) -> None:
        """Populate *active_features* for the startup status line."""
        features: list[str] = []
        features.append(profile.kernel_path)
        if profile.kv_mode != "fp16":
            features.append(f"kv-{profile.kv_mode}")
        if profile.use_eagle3:
            features.append("eagle3")
        if profile.use_sparsity:
            features.append("sparse")
        if profile.use_moe_lazy:
            features.append("moe-lazy")
        profile.active_features = features
