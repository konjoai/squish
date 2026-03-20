"""squish/install/dependency_resolver.py — Platform-aware dependency resolver.

Determines which Python packages are required for the current hardware
platform (macOS Apple Silicon, Linux CUDA, Linux ROCm, CPU-only) and
generates the correct pip install command.

Classes
───────
DependencyGroup         — A named group of related packages.
InstallSpec             — Single package install specification.
DependencyResolverConfig — Configuration dataclass.
DependencyResolverStats  — Runtime statistics.
DependencyResolver       — Main resolver class.

Usage::

    from squish.platform.detector import UnifiedPlatformDetector
    from squish.install.dependency_resolver import DependencyResolver

    info     = UnifiedPlatformDetector().detect()
    resolver = DependencyResolver(platform_info=info)
    specs    = resolver.resolve()
    cmd      = resolver.get_install_command()
    missing  = resolver.check_missing()
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstallSpec:
    """A single package install specification.

    Attributes
    ----------
    package:
        PyPI package name, e.g. 'torch'.
    version_spec:
        PEP 440 version constraint, e.g. '>=2.1.0'. Empty string = no pin.
    extras:
        pip extras, e.g. ['cu121'] for 'torch[cu121]'.
    platform_constraint:
        Optional regex-like platform kind string: 'LINUX_CUDA', 'MACOS_*', etc.
        Empty = applicable to all platforms.
    optional:
        If True, the package is recommended but not required.
    """
    package:             str
    version_spec:        str        = ""
    extras:              tuple      = ()
    platform_constraint: str        = ""
    optional:            bool       = False

    @property
    def pip_token(self) -> str:
        """Return the string passed to pip, e.g. 'torch[cu121]>=2.1.0'."""
        tok = self.package
        if self.extras:
            tok += f"[{','.join(self.extras)}]"
        if self.version_spec:
            tok += self.version_spec
        return tok


@dataclass
class DependencyGroup:
    """A named group of related InstallSpecs."""
    name:          str
    description:   str
    packages:      List[InstallSpec] = field(default_factory=list)
    platform_kinds: List[str]        = field(default_factory=list)
    """List of PlatformKind.name values this group applies to."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DependencyResolverConfig:
    """Configuration for DependencyResolver.

    Attributes
    ----------
    auto_install:
        If True, actually run pip on check_missing() results. Default False.
        (Explicit opt-in required to avoid unintended side-effects.)
    pip_extra_index:
        Optional extra PyPI index URL, e.g. the PyTorch CUDA index.
    include_optional:
        Include optional packages in resolve() results. Default True.
    """
    auto_install:      bool            = False
    pip_extra_index:   Optional[str]   = None
    include_optional:  bool            = True


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class DependencyResolverStats:
    """Runtime statistics for DependencyResolver."""
    resolve_calls:     int = 0
    install_calls:     int = 0
    validation_calls:  int = 0
    missing_detected:  int = 0


# ---------------------------------------------------------------------------
# Package manifests (platform-specific)
# ---------------------------------------------------------------------------

# Core packages required on every platform
_CORE_PACKAGES = [
    InstallSpec("numpy",          ">=1.24.0"),
    InstallSpec("safetensors",    ">=0.4.0"),
    InstallSpec("huggingface-hub",">=0.20.0"),
    InstallSpec("tqdm",           ">=4.65.0"),
    InstallSpec("requests",       ">=2.30.0"),
]

_MLX_PACKAGES = [
    InstallSpec("mlx",            ">=0.5.0",  platform_constraint="MACOS_APPLE_SILICON"),
    InstallSpec("mlx-lm",         ">=0.3.0",  platform_constraint="MACOS_APPLE_SILICON"),
]

_CUDA_PACKAGES = [
    InstallSpec("torch",          ">=2.1.0",  extras=("cu121",), platform_constraint="LINUX_CUDA"),
    InstallSpec("torchvision",    ">=0.16.0", extras=("cu121",), platform_constraint="LINUX_CUDA"),
    InstallSpec("flash-attn",     ">=2.3.0",  platform_constraint="LINUX_CUDA", optional=True),
    InstallSpec("xformers",       ">=0.0.23", platform_constraint="LINUX_CUDA", optional=True),
    InstallSpec("bitsandbytes",   ">=0.41.0", platform_constraint="LINUX_CUDA", optional=True),
]

_ROCM_PACKAGES = [
    InstallSpec("torch",          ">=2.1.0",  extras=("rocm5.7",), platform_constraint="LINUX_ROCM"),
    InstallSpec("torchvision",    ">=0.16.0", extras=("rocm5.7",), platform_constraint="LINUX_ROCM"),
]

_CPU_PACKAGES = [
    InstallSpec("torch",          ">=2.1.0",  platform_constraint="LINUX_CPU"),
    InstallSpec("torchvision",    ">=0.16.0", platform_constraint="LINUX_CPU"),
]

_WIN_PACKAGES = [
    InstallSpec("torch",          ">=2.1.0",  extras=("cu121",), platform_constraint="WINDOWS_WSL"),
    InstallSpec("torch",          ">=2.1.0",  platform_constraint="WINDOWS_NATIVE"),
]

_TORCH_EXTRA_INDEX_URLS: Dict[str, str] = {
    "LINUX_CUDA":     "https://download.pytorch.org/whl/cu121",
    "LINUX_ROCM":     "https://download.pytorch.org/whl/rocm5.7",
    "LINUX_CPU":      "https://download.pytorch.org/whl/cpu",
    "WINDOWS_WSL":    "https://download.pytorch.org/whl/cu121",
    "WINDOWS_NATIVE": "https://download.pytorch.org/whl/cpu",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DependencyResolver:
    """Determine required packages for the current hardware platform.

    Usage::

        resolver = DependencyResolver(platform_info=info)
        specs    = resolver.resolve()
        print(resolver.get_install_command())
        missing  = resolver.check_missing()
    """

    def __init__(
        self,
        config: Optional[DependencyResolverConfig] = None,
        platform_info: Optional[Any] = None,
    ) -> None:
        self._cfg      = config or DependencyResolverConfig()
        self._platform = platform_info
        self.stats     = DependencyResolverStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, platform_info: Optional[Any] = None) -> List[InstallSpec]:
        """Return the list of InstallSpecs applicable to this platform.

        Parameters
        ----------
        platform_info : optional override; uses constructor value if None.
        """
        self.stats.resolve_calls += 1
        pi   = platform_info or self._platform
        kind = self._kind_name(pi)

        all_specs = (
            _CORE_PACKAGES
            + _MLX_PACKAGES
            + _CUDA_PACKAGES
            + _ROCM_PACKAGES
            + _CPU_PACKAGES
            + _WIN_PACKAGES
        )
        result: List[InstallSpec] = []
        for spec in all_specs:
            if not self._spec_applies(spec, kind):
                continue
            if spec.optional and not self._cfg.include_optional:
                continue
            result.append(spec)
        return result

    def validate(self, platform_info: Optional[Any] = None) -> Dict[str, bool]:
        """Check which resolved packages are importable.

        Returns a dict mapping package name → importable.
        """
        self.stats.validation_calls += 1
        specs = self.resolve(platform_info)
        result: Dict[str, bool] = {}
        for spec in specs:
            module = _PACKAGE_TO_MODULE.get(spec.package, spec.package.replace("-", "_"))
            try:
                importlib.import_module(module)
                result[spec.package] = True
            except ImportError:
                result[spec.package] = False
        return result

    def get_install_command(self, platform_info: Optional[Any] = None) -> str:
        """Return a full pip install command string for this platform."""
        pi   = platform_info or self._platform
        kind = self._kind_name(pi)

        specs   = self.resolve(pi)
        tokens  = " ".join(s.pip_token for s in specs)
        extra   = self._cfg.pip_extra_index or _TORCH_EXTRA_INDEX_URLS.get(kind, "")
        cmd     = f"pip install {tokens}"
        if extra:
            cmd += f" --extra-index-url {extra}"
        return cmd

    def check_missing(self, platform_info: Optional[Any] = None) -> List[str]:
        """Return a list of package names that are not importable."""
        validation = self.validate(platform_info)
        missing = [pkg for pkg, ok in validation.items() if not ok]
        self.stats.missing_detected += len(missing)
        return missing

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_applies(spec: InstallSpec, kind: str) -> bool:
        """Return True if this spec applies to the given platform kind."""
        if not spec.platform_constraint:
            return True
        constraint = spec.platform_constraint.upper()
        # Simple glob-style: 'MACOS_*' matches 'MACOS_APPLE_SILICON'
        if constraint.endswith("*"):
            return kind.startswith(constraint[:-1])
        return kind == constraint

    @staticmethod
    def _kind_name(platform_info: Optional[Any]) -> str:
        if platform_info is None:
            import sys
            if sys.platform == "darwin":
                return "MACOS_APPLE_SILICON"
            try:
                import torch
                if torch.cuda.is_available():
                    return "LINUX_CUDA"
            except ImportError:
                pass
            return "LINUX_CPU"
        return platform_info.kind.name

    def __repr__(self) -> str:
        kind = self._kind_name(self._platform)
        return (
            f"DependencyResolver("
            f"platform={kind}, "
            f"resolve_calls={self.stats.resolve_calls})"
        )


# ---------------------------------------------------------------------------
# Helper: package name → importable module name
# ---------------------------------------------------------------------------

_PACKAGE_TO_MODULE: Dict[str, str] = {
    "huggingface-hub":  "huggingface_hub",
    "safetensors":      "safetensors",
    "flash-attn":       "flash_attn",
    "bitsandbytes":     "bitsandbytes",
    "mlx-lm":           "mlx_lm",
    "mlx":              "mlx.core",
    "torch":            "torch",
    "torchvision":      "torchvision",
    "numpy":            "numpy",
    "tqdm":             "tqdm",
    "requests":         "requests",
    "xformers":         "xformers",
}
