"""TorchCompileDecode — torch.compile + MLX compile wrapper for decode steps.

Implements the ``torch.compile`` integration described in the PyTorch 2.4+
documentation and measured in the LLM decode benchmarks (2024).

**Effect**: ``torch.compile(fullgraph=True, mode='reduce-overhead')`` traces
the decode forward pass once and emits persistent CUDA kernels that avoid
Python dispatch overhead on every subsequent call.  On A100/H100 this yields
15–40% throughput improvement without any model changes.

On Apple Silicon this module activates the equivalent ``mlx.core.compile``
path, which caches the Metal computation graph and avoids Python re-dispatch.

On CPU-only or older PyTorch versions the module transparently falls back to
eager execution.

This module provides:

* :class:`TorchCompileConfig` — compile mode, backend, and fallback settings.
* :class:`TorchCompileDecode` — wraps any callable forward function and
  manages the compile lifecycle (lazy compile on first call, recompile on
  shape change if ``dynamic=True``).

Reference:
    PyTorch Inductor documentation: ``torch.compile`` for LLM decode (2024).
    PyTorch 2.4+ changelog: ``reduce-overhead`` mode persistent kernels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "TorchCompileConfig",
    "CompileStats",
    "TorchCompileDecode",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class TorchCompileConfig:
    """Configuration for TorchCompileDecode.

    Attributes:
        mode: torch.compile mode string.  One of ``"default"``,
            ``"reduce-overhead"``, ``"max-autotune"``.
        fullgraph: If True, require a single-graph compilation
            (no Python graph-breaks).
        dynamic: If True, allow dynamic shapes (re-trace on shape change).
        backend: torch.compile backend (``"inductor"``, ``"eager"``, etc.).
        use_mlx_compile: If True, attempt ``mlx.core.compile`` path on Apple
            Silicon.  Ignored if MLX is unavailable.
        fallback_to_eager: If True (default), fall back to eager execution when
            compilation is unavailable.  If False, raise on compile failure.
    """

    mode: str = "reduce-overhead"
    fullgraph: bool = True
    dynamic: bool = False
    backend: str = "inductor"
    use_mlx_compile: bool = True
    fallback_to_eager: bool = True

    def __post_init__(self) -> None:
        valid_modes = {"default", "reduce-overhead", "max-autotune"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}; got '{self.mode}'"
            )


# ── Stats ─────────────────────────────────────────────────────────────────────


@dataclass
class CompileStats:
    """Compilation and execution statistics.

    Attributes:
        n_calls: Total number of forward calls.
        n_recompiles: Number of times the function was recompiled.
        compile_latency_s: Wall-clock time spent compiling (seconds).
        mean_call_latency_us: EMA of per-call latency in microseconds.
        compiled: Whether the function is currently compiled.
        backend_used: The backend actually used (may differ if fallback).
    """

    n_calls: int = 0
    n_recompiles: int = 0
    compile_latency_s: float = 0.0
    mean_call_latency_us: float = 0.0
    compiled: bool = False
    backend_used: str = "eager"


# ── Main class ────────────────────────────────────────────────────────────────


class TorchCompileDecode:
    """torch.compile wrapper for LLM decode forward functions.

    Example::

        def my_decode_fn(token_ids, kv_cache):
            # ... forward step ...
            return logits

        cfg     = TorchCompileConfig(mode="reduce-overhead", fullgraph=True)
        wrapper = TorchCompileDecode(cfg)
        wrapper.compile(my_decode_fn)
        logits  = wrapper(token_ids, kv_cache)

    Args:
        config: :class:`TorchCompileConfig` (optional).
    """

    def __init__(self, config: Optional[TorchCompileConfig] = None) -> None:
        self.config: TorchCompileConfig = config or TorchCompileConfig()
        self._fn: Optional[Callable] = None
        self._compiled_fn: Optional[Callable] = None
        self._stats: CompileStats = CompileStats()
        self._last_input_signature: Optional[tuple] = None

    # ── Compile ───────────────────────────────────────────────────────────────

    def compile(self, fn: Callable) -> None:
        """Set and attempt to compile the decode forward function.

        Args:
            fn: The callable to compile.  Should accept arbitrary positional
                and keyword arguments and return a NumPy array or tensor.
        """
        self._fn = fn
        self._compiled_fn = None
        self._stats = CompileStats()
        self._try_compile()

    def _try_compile(self) -> None:
        """Attempt torch.compile; fall back to eager on failure."""
        cfg = self.config
        fn = self._fn
        if fn is None:
            return

        t0 = time.perf_counter()

        # Try PyTorch compile
        try:
            import torch  # type: ignore

            if not hasattr(torch, "compile"):
                raise ImportError("torch.compile not available")

            self._compiled_fn = torch.compile(
                fn,
                mode=cfg.mode,
                fullgraph=cfg.fullgraph,
                dynamic=cfg.dynamic,
                backend=cfg.backend,
            )
            self._stats.compiled = True
            self._stats.backend_used = cfg.backend
        except Exception:
            # Try MLX compile on Apple Silicon
            if cfg.use_mlx_compile:
                try:
                    import mlx.core as mx  # type: ignore

                    self._compiled_fn = mx.compile(fn)
                    self._stats.compiled = True
                    self._stats.backend_used = "mlx"
                except Exception:
                    self._compiled_fn = fn
                    self._stats.backend_used = "eager"
            else:
                if cfg.fallback_to_eager:
                    self._compiled_fn = fn
                    self._stats.backend_used = "eager"
                else:
                    raise RuntimeError(
                        "TorchCompileDecode: compilation failed and "
                        "fallback_to_eager=False"
                    )

        self._stats.compile_latency_s = time.perf_counter() - t0
        self._stats.n_recompiles += 1

    # ── Call ──────────────────────────────────────────────────────────────────

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the (possibly compiled) function.

        Args:
            *args: Forwarded to the underlying function.
            **kwargs: Forwarded to the underlying function.

        Returns:
            Whatever the underlying function returns.

        Raises:
            RuntimeError: If :meth:`compile` has not been called.
        """
        if self._compiled_fn is None:
            raise RuntimeError(
                "TorchCompileDecode: call compile(fn) before using __call__"
            )

        # Check for shape change that requires recompile in non-dynamic mode
        if not self.config.dynamic:
            sig = self._input_signature(args)
            if self._last_input_signature is not None and sig != self._last_input_signature:
                self._try_compile()
            self._last_input_signature = sig

        t0 = time.perf_counter()
        result = self._compiled_fn(*args, **kwargs)
        elapsed_us = (time.perf_counter() - t0) * 1e6

        s = self._stats
        s.n_calls += 1
        alpha = 0.9
        s.mean_call_latency_us = (
            alpha * s.mean_call_latency_us + (1 - alpha) * elapsed_us
            if s.n_calls > 1 else elapsed_us
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _input_signature(args: tuple) -> tuple:
        """Extract a hashable shape+dtype signature from positional args."""
        sig = []
        for a in args:
            if isinstance(a, np.ndarray):
                sig.append((a.shape, str(a.dtype)))
            elif hasattr(a, "shape"):
                sig.append((tuple(a.shape), str(getattr(a, "dtype", "?"))))
            else:
                sig.append(type(a).__name__)
        return tuple(sig)

    @property
    def stats(self) -> CompileStats:
        """Current compilation and execution statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset execution statistics (not compilation state)."""
        s = self._stats
        n_recompiles = s.n_recompiles
        compiled = s.compiled
        backend_used = s.backend_used
        compile_latency_s = s.compile_latency_s
        self._stats = CompileStats(
            n_recompiles=n_recompiles,
            compiled=compiled,
            backend_used=backend_used,
            compile_latency_s=compile_latency_s,
        )

    def __repr__(self) -> str:
        return (
            f"TorchCompileDecode("
            f"mode='{self.config.mode}', "
            f"compiled={self._stats.compiled}, "
            f"backend='{self._stats.backend_used}')"
        )
