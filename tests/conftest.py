"""
tests/conftest.py
Shared pytest configuration for all Squish tests.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Metal-safe guard — two-layer defence against SIGABRT in sandbox
#
# In the VS Code sandboxed terminal `import mlx.core` triggers Metal GPU
# initialisation which raises SIGABRT — a C-level signal that Python's
# try/except cannot catch — killing pytest immediately.
#
# Layer 1 — import hook
#   Intercept every `import mlx*` / `import mlx_lm*` attempt and raise
#   ImportError instead of allowing Metal to initialise.  This means every
#   `try: import mlx.core except ImportError:` guard in squish source code
#   works correctly, and tests that require mlx fail with a clean error
#   rather than a process crash.
#
# Layer 2 — collection skip
#   Tests in subdirectories whose files import mlx at module level WITHOUT
#   a guard (e.g. `import mlx.core as mx`) are excluded from collection
#   entirely so they don't appear as collection ERRORs.
#
# In CI (GITHUB_ACTIONS / CI env vars) both layers are disabled — the full
# suite runs with real Metal.
# ---------------------------------------------------------------------------
_IN_CI = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


# ---- Layer 1: import hook --------------------------------------------------
if not _IN_CI:
    import importlib.abc
    import importlib.machinery

    _MLX_BLOCKED_PREFIXES = ("mlx",)

    class _MetalImportBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Meta-path hook that raises ImportError for mlx* imports in sandbox.

        Uses the Python 3.4+ find_spec / exec_module protocol.
        Prevents Metal GPU initialisation (=> SIGABRT) from occurring in the
        VS Code sandboxed terminal.  Active only when CI env vars are absent.
        """

        def find_spec(  # type: ignore[override]
            self,
            fullname: str,
            path: Any,
            target: Any = None,
        ) -> "importlib.machinery.ModuleSpec | None":
            if fullname in sys.modules:
                return None  # already loaded — let Python return cached module
            if any(
                fullname == p or fullname.startswith(p + ".")
                for p in _MLX_BLOCKED_PREFIXES
            ):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec: "importlib.machinery.ModuleSpec") -> None:
            return None  # use default module creation

        def exec_module(self, module: ModuleType) -> None:
            raise ImportError(
                f"{module.__name__!r} is blocked in the VS Code sandbox "
                "(Metal GPU initialisation would SIGABRT). "
                "Run with CI=1 or outside the sandbox to use mlx."
            )

    sys.meta_path.insert(0, _MetalImportBlocker())



# ---- Layer 2: collection skip ----------------------------------------------
# Subdirectories whose test files have unguarded mlx imports at module level.
# Skipping them avoids collection ERRORs.  CI runs everything.
_METAL_SUBDIRS: frozenset[str] = frozenset({
    "integration",
    "kv",
    "quant",
    "streaming",
    "serving",
    "speculative",
    "io",
})

# Individual top-level files with unguarded module-level mlx imports.
_METAL_FILES: frozenset[str] = frozenset({
    "test_backend_unit.py",
    "test_wave78_perf_quality.py",
})

collect_ignore: list[str] = [] if _IN_CI else list(_METAL_FILES)


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:  # type: ignore[override]
    """Skip Metal-crashing subdirs in non-CI environments."""
    if _IN_CI:
        return None
    parts = collection_path.parts
    if "tests" in parts:
        idx = list(parts).index("tests")
        subdir = parts[idx + 1] if len(parts) > idx + 1 else ""
        if subdir in _METAL_SUBDIRS:
            return True
    return None



def pytest_addoption(parser):
    parser.addoption(
        "--model", default=None,
        help="Model hint passed to squish --model  (e.g. '14b', '7b', full path)"
    )
    parser.addoption(
        "--run-hardware", action="store_true", default=False,
        help="Run @pytest.mark.hardware tests (requires Apple Silicon + MLX + a cached model)"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="Run @pytest.mark.integration tests (requires a live squish server at localhost:11435)"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: require Apple Silicon + MLX + a cached model (skipped in CI unless --run-hardware is set)",
    )
    config.addinivalue_line(
        "markers",
        "integration: requires live squish serve (skipped unless --run-integration or -m integration)",
    )
    # SwigPy warnings from compiled MLX/Metal bindings — not actionable
    warnings.filterwarnings("ignore", message="builtin type SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="builtin type swigvar.*", category=DeprecationWarning)
    # Starlette TestClient timeout kwarg deprecation — upstream issue
    warnings.filterwarnings("ignore", message="You should not use the 'timeout'.*", category=DeprecationWarning)


def pytest_collection_modifyitems(config, items):
    """Skip hardware and integration tests unless the corresponding flag is set."""
    run_hardware = config.getoption("--run-hardware")
    run_integration = config.getoption("--run-integration")
    # Also allow -m integration to opt in without the flag.
    mark_expr = getattr(config.option, "markexpr", "") or ""
    m_integration = "integration" in mark_expr

    skip_hw = pytest.mark.skip(reason="hardware test — pass --run-hardware to enable")
    skip_int = pytest.mark.skip(reason="integration test — pass --run-integration to enable")

    for item in items:
        if not run_hardware and item.get_closest_marker("hardware"):
            item.add_marker(skip_hw)
        if not run_integration and not m_integration and item.get_closest_marker("integration"):
            item.add_marker(skip_int)
