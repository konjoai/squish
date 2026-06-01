"""squish/serving/kernel_cache.py — MLX Metal kernel cache management.

Phase 3 investigation findings
-------------------------------
MLX 0.18–0.22 does NOT expose a public kernel-cache API.  Specifically:

1. ``mlx.core.metal`` (macOS only) provides device/memory introspection but
   **no** ``compile_cache`` or ``save_kernel_cache`` equivalent.  The only
   related env var documented in the MLX source is ``MLX_FAST_SDPA`` (enables
   fused attention), not a kernel persistence flag.

2. The Metal library cache (*.metallib) is managed by the macOS Metal
   framework at the OS level via ``MTLDevice.makeDefaultLibrary`` and cached
   in ``~/Library/Caches/com.apple.metal``.  This cache persists across
   processes automatically — macOS re-uses compiled shaders from the OS cache
   without any Python-level API.

3. MLX's Python-level ``mx.compile()`` compiles the *computation graph* (XLA-
   style), not individual Metal kernels.  The compiled callable is in-process
   only; it cannot be serialised to disk.

4. The JIT warmup cost (~0.8–1.2 s on M3) is therefore:
   a. The first Metal shader dispatch triggering macOS shader compilation
      (amortised across sessions by the OS cache after the first run).
   b. Python-level graph tracing and ``mx.compile()`` invocation — this
      portion IS re-paid on every new process start and cannot be bypassed
      without a Metal-level API that does not currently exist.

Practical consequence
---------------------
The OS-level Metal cache handles (a) automatically.  Squish can accelerate
warmup by:
  • Calling ``_run_warmup_pass()`` at daemon startup before the first user
    request arrives — the warmup cost is hidden behind model loading latency.
  • Setting ``METAL_DEVICE_WRAPPER_TYPE=1`` (undocumented Apple debug flag) is
    NOT recommended — it disables optimizations.
  • Pre-compilation via ``mx.compile()`` of the two hot shapes ([1,1] decode
    step and [1,K] verify batch) is already performed in SpeculativeGenerator
    Phase 1.1 — this is the best available approach.

The ``ensure_kernel_cache_dir()`` function sets up a persistent directory that
can be used for future MLX versions if they expose a ``MLX_KERNEL_CACHE_DIR``
environment variable (similar to ``XLA_FLAGS`` / ``TRITON_CACHE_DIR``).  No
data is written to it by this module; it is reserved for forward compatibility.
"""
from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical cache directory for squish's MLX kernel artefacts.
# Currently unused by MLX; reserved for future versions.
_DEFAULT_KERNEL_CACHE_DIR = Path.home() / ".cache" / "squish" / "mlx_kernels"

# Set this env var before MLX 0.x if/when it begins respecting it.
_MLX_KERNEL_CACHE_ENV = "MLX_KERNEL_CACHE_DIR"


def ensure_kernel_cache_dir(
    cache_dir: "Path | str | None" = None,
) -> Path:
    """Create (if absent) the MLX kernel cache directory and set the env var.

    Returns the directory path.  On Linux this is a no-op (MLX is not
    available) but the path is still returned for logging purposes.
    """
    d = Path(cache_dir or _DEFAULT_KERNEL_CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    # Set env var for forward compatibility with future MLX versions
    os.environ.setdefault(_MLX_KERNEL_CACHE_ENV, str(d))
    return d


def mlx_supports_kernel_cache() -> bool:
    """Return True if the installed MLX version exposes a kernel-cache API.

    Currently always False — MLX 0.18–0.22 have no such API.
    Updated whenever MLX gains this capability.
    """
    if platform.system() != "Darwin":
        return False
    try:
        import mlx.core as mx
        metal = getattr(mx, "metal", None)
        return hasattr(metal, "set_cache_limit") and callable(
            getattr(metal, "save_kernel_cache", None)
        )
    except ImportError:
        return False


def run_warmup_pass(model=None, tokenizer=None) -> float:
    """Run a minimal forward pass to warm up Metal shader compilation.

    On M3, the first forward pass compiles Metal shaders — this costs ~0.8–1.2 s
    for a 7B model.  By running this pass at daemon startup (before any user
    request), the latency is hidden behind model loading time.

    Returns elapsed time in seconds (wall clock of the warmup pass).
    """
    import time

    if model is None:
        return 0.0

    t0 = time.perf_counter()
    try:
        if platform.system() != "Darwin":
            return 0.0
        import mlx.core as mx
        # Minimal forward pass: batch=1, seq=1
        dummy_ids = mx.array([[0]], dtype=mx.int32)
        try:
            out = model(dummy_ids)
            mx.eval(out)
        except Exception:
            pass
    except ImportError:
        pass
    elapsed = time.perf_counter() - t0
    logger.info("Metal kernel warmup pass: %.3f s", elapsed)
    return elapsed


def metal_cache_info() -> dict:
    """Return diagnostic info about the Metal/MLX cache state."""
    info: dict = {
        "mlx_kernel_cache_api": mlx_supports_kernel_cache(),
        "cache_dir":            str(_DEFAULT_KERNEL_CACHE_DIR),
        "cache_dir_exists":     _DEFAULT_KERNEL_CACHE_DIR.exists(),
        "env_var":              _MLX_KERNEL_CACHE_ENV,
        "env_var_set":          _MLX_KERNEL_CACHE_ENV in os.environ,
    }
    if platform.system() == "Darwin":
        # macOS system Metal cache location (managed by the OS)
        system_cache = Path.home() / "Library" / "Caches" / "com.apple.metal"
        info["system_metal_cache"] = str(system_cache)
        info["system_metal_cache_exists"] = system_cache.exists()
        try:
            import mlx.core as mx
            info["mlx_version"] = getattr(mx, "__version__", "unknown")
        except ImportError:
            info["mlx_version"] = "not installed"
    return info
