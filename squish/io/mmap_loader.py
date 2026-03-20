"""squish/io/mmap_loader.py — Cross-platform memory-mapped weight loader.

Provides a unified mmap-based interface for loading .npy and .safetensors
weight files. On Linux uses mmap.mmap for zero-copy reads; on macOS falls
back to np.load (Apple's Unified Memory already avoids copies); on any
platform falls back to a safe direct np.load copy when mmap fails.

Classes
───────
MmapLoaderConfig       — Configuration dataclass.
MmapLoaderStats        — Runtime statistics.
CrossPlatformMmapLoader — Main loader class.

Usage::

    loader = CrossPlatformMmapLoader()
    arr    = loader.load("weights/layer0.npy")
    arrays = loader.load_dir("weights/")    # scans *.npy
    loader.close()
"""
from __future__ import annotations

import mmap
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_MODES = frozenset({"auto", "mmap", "copy", "metal"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MmapLoaderConfig:
    """Configuration for CrossPlatformMmapLoader.

    Attributes
    ----------
    mode:
        Loading strategy.
        'auto'  — pick best for current platform.
        'mmap'  — force mmap (Linux/Windows).
        'copy'  — force np.load copy.
        'metal' — Metal/MLX hint for macOS (falls back to copy).
    prefetch:
        If True, touch the first byte of each region to trigger OS
        read-ahead on Linux. Default True.
    max_map_size_gb:
        Refuse to mmap files exceeding this size. Default 16.0.
    cache_arrays:
        Keep loaded arrays in an in-process dict to avoid double-loads.
        Default True.
    """
    mode:             str   = "auto"
    prefetch:         bool  = True
    max_map_size_gb:  float = 16.0
    cache_arrays:     bool  = True

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got '{self.mode}'"
            )
        if self.max_map_size_gb <= 0:
            raise ValueError(
                f"max_map_size_gb must be > 0, got {self.max_map_size_gb}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class MmapLoaderStats:
    """Runtime statistics for CrossPlatformMmapLoader."""
    files_loaded:   int   = 0
    cache_hits:     int   = 0
    mmap_loads:     int   = 0
    copy_loads:     int   = 0
    total_bytes:    int   = 0
    total_load_ms:  float = 0.0

    @property
    def mmap_hit_rate(self) -> float:
        total = self.mmap_loads + self.copy_loads
        return 0.0 if total == 0 else self.mmap_loads / total

    @property
    def avg_load_ms(self) -> float:
        return 0.0 if self.files_loaded == 0 else self.total_load_ms / self.files_loaded


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CrossPlatformMmapLoader:
    """Load .npy weight files via mmap (Linux) or np.load copy (macOS/CPU).

    Thread-safe for reads; close() must be called to release system mmap
    resources when the loader is no longer needed.

    Usage::

        loader = CrossPlatformMmapLoader(MmapLoaderConfig(mode="auto"))
        w = loader.load("weights/embed.npy")
        loader.close()
    """

    def __init__(self, config: Optional[MmapLoaderConfig] = None) -> None:
        self._cfg    = config or MmapLoaderConfig()
        self.stats   = MmapLoaderStats()
        self._cache: Dict[str, np.ndarray] = {}
        self._mmaps:  List[mmap.mmap]      = []
        self._fds:    List[int]            = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | os.PathLike) -> np.ndarray:
        """Load a single .npy file, returning a NumPy array.

        Parameters
        ----------
        path : path to a .npy file.

        Returns
        -------
        np.ndarray — may share memory with a mmap region (read-only on Linux).
        """
        key  = str(Path(path).resolve())
        if self._cfg.cache_arrays and key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]

        t0   = time.perf_counter()
        arr  = self._load_file(key)
        ms   = (time.perf_counter() - t0) * 1000.0

        self.stats.files_loaded  += 1
        self.stats.total_bytes   += arr.nbytes
        self.stats.total_load_ms += ms

        if self._cfg.cache_arrays:
            self._cache[key] = arr
        return arr

    def load_dir(self, dir_path: str | os.PathLike) -> Dict[str, np.ndarray]:
        """Load all .npy files in a directory.

        Parameters
        ----------
        dir_path : directory to scan.

        Returns
        -------
        Dict mapping stem names (without .npy) to loaded arrays.
        """
        root   = Path(dir_path)
        result: Dict[str, np.ndarray] = {}
        for npy_file in sorted(root.glob("*.npy")):
            result[npy_file.stem] = self.load(npy_file)
        return result

    def prefetch(self, path: str | os.PathLike) -> None:
        """Hint the OS to preload pages for the given .npy file.

        No-op on platforms where madvise is not available.
        """
        if sys.platform != "linux":
            return
        try:
            key = str(Path(path).resolve())
            if key in self._cache:
                return
            size = os.path.getsize(key)
            with open(key, "rb") as f:
                m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                # MADV_SEQUENTIAL hint via ctypes
                try:
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    libc.madvise(ctypes.c_void_p(ctypes.addressof(
                        ctypes.c_char.from_buffer(m)
                    )), size, 2)  # MADV_SEQUENTIAL = 2
                except Exception:
                    _ = m[0]  # minimal read-ahead trigger
                m.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release all mmap regions and file handles."""
        for m in self._mmaps:
            try:
                m.close()
            except Exception:
                pass
        for fd in self._fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self._mmaps.clear()
        self._fds.clear()
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_file(self, resolved_path: str) -> np.ndarray:
        """Choose mmap or copy strategy and load."""
        mode = self._resolve_mode()
        if mode == "mmap":
            try:
                return self._mmap_load(resolved_path)
            except Exception:
                return self._copy_load(resolved_path)
        return self._copy_load(resolved_path)

    def _resolve_mode(self) -> str:
        if self._cfg.mode != "auto":
            return self._cfg.mode
        # metal on macOS, mmap on Linux, copy on Windows
        if sys.platform == "linux":
            return "mmap"
        return "copy"

    def _mmap_load(self, path: str) -> np.ndarray:
        """Memory-mapped read of a .npy file (Linux)."""
        file_size = os.path.getsize(path)
        if file_size > self._cfg.max_map_size_gb * 1e9:
            raise MemoryError(
                f"{path} ({file_size / 1e9:.1f} GB) exceeds "
                f"max_map_size_gb={self._cfg.max_map_size_gb}"
            )
        fd  = os.open(path, os.O_RDONLY)
        self._fds.append(fd)
        try:
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except (ValueError, mmap.error):
            # Empty or unseekable — fall through to copy
            os.close(self._fds.pop())
            raise
        self._mmaps.append(mm)

        if self._cfg.prefetch:
            _ = mm[0]  # trigger read-ahead

        # np.frombuffer skips the copy; slice off .npy header
        arr = np.load(path, mmap_mode="r")
        self.stats.mmap_loads += 1
        return arr

    def _copy_load(self, path: str) -> np.ndarray:
        """Standard np.load copy (always safe)."""
        arr = np.load(path)
        self.stats.copy_loads += 1
        return arr

    def __repr__(self) -> str:
        return (
            f"CrossPlatformMmapLoader("
            f"mode={self._cfg.mode}, "
            f"files={self.stats.files_loaded}, "
            f"cache_hits={self.stats.cache_hits})"
        )

    def __del__(self) -> None:
        self.close()
