"""Mojo runtime bridge: ctypes-based dynamic loader with Rust + NumPy fallback.

This module handles detection and lazy loading of compiled Mojo shared libraries
(``libsquish_kernels.so`` / ``.dylib``).  When the ``magic`` Mojo toolchain is
not installed, or the compiled library is absent, all kernel lookups return
``None`` and callers fall back to Rust (via ``squish_quant``) or pure NumPy.

Typical backend resolution order for any Mojo wrapper:
1. Mojo (ctypes call into compiled ``.so``)
2. Rust (``squish_quant`` PyO3 module)
3. NumPy (pure-Python reference)
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

__all__ = ["MojoBridgeConfig", "MojoBridge"]

_log = logging.getLogger(__name__)

# Shared-library name stem; compiled by `magic run mojo build --emit shared …`
_LIB_STEM = "libsquish_kernels"

# Attempt to import Rust extension — used as secondary fallback
try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


def _default_lib_search_paths() -> list[Path]:
    """Return candidate directories for the compiled Mojo shared library."""
    mojo_dir = Path(__file__).parent
    return [
        mojo_dir / "kernels",
        mojo_dir,
        Path(os.environ.get("SQUISH_MOJO_LIB_DIR", mojo_dir / "kernels")),
    ]


def _lib_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"{_LIB_STEM}.dylib"
    return f"{_LIB_STEM}.so"


@dataclass
class MojoBridgeConfig:
    """Configuration for :class:`MojoBridge`.

    Attributes
    ----------
    lib_path:
        Explicit path to the compiled Mojo shared library.  When ``None``,
        the bridge searches :func:`_default_lib_search_paths`.
    rust_fallback:
        When ``True`` (default), callers can query whether Rust (``squish_quant``)
        is available via :meth:`MojoBridge.backend`.
    """

    lib_path: Optional[str] = None
    rust_fallback: bool = True


class MojoBridge:
    """ctypes bridge to compiled Mojo kernels.

    The bridge is a lightweight singleton-like helper.  Instantiate once and
    share across kernel wrappers.

    Parameters
    ----------
    config:
        Bridge configuration.  Defaults to :class:`MojoBridgeConfig`.
    """

    # Module-level availability flag — set to True only when the .so loads
    _mojo_available: bool = False

    def __init__(self, config: MojoBridgeConfig | None = None) -> None:
        self.config = config or MojoBridgeConfig()
        self._lib: ctypes.CDLL | None = None
        self._try_load()

    # ------------------------------------------------------------------ #
    #  Library loading                                                     #
    # ------------------------------------------------------------------ #

    def _try_load(self) -> None:
        """Attempt to load the compiled Mojo shared library."""
        candidate: Path | None = None

        if self.config.lib_path is not None:
            candidate = Path(self.config.lib_path)
        else:
            for search_dir in _default_lib_search_paths():
                p = search_dir / _lib_filename()
                if p.exists():
                    candidate = p
                    break

        if candidate is None or not candidate.exists():
            _log.debug(
                "Mojo shared library not found; falling back to Rust/NumPy. "
                "Build with: magic run mojo build --emit shared kernels/"
            )
            return

        try:
            self._lib = ctypes.CDLL(str(candidate))
            MojoBridge._mojo_available = True
            _log.info("Mojo kernel library loaded from %s", candidate)
        except OSError as exc:
            _log.warning("Failed to load Mojo library %s: %s", candidate, exc)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_kernel(self, name: str) -> Optional[Callable]:
        """Look up a kernel function from the compiled Mojo library.

        Parameters
        ----------
        name:
            C-symbol name of the kernel (e.g. ``"mojo_softmax_f32"``).

        Returns
        -------
        A callable ctypes function object, or ``None`` if the library is
        unavailable or the symbol was not found.
        """
        if self._lib is None:
            return None
        try:
            fn = getattr(self._lib, name)
            return fn
        except AttributeError:
            _log.debug("Symbol '%s' not found in Mojo library.", name)
            return None

    def is_available(self) -> bool:
        """Return ``True`` if the compiled Mojo library was loaded successfully."""
        return MojoBridge._mojo_available

    def backend(self) -> str:
        """Return the active backend name.

        Returns
        -------
        ``"mojo"`` when the shared library is loaded, ``"rust"`` when only the
        Rust extension is available, ``"numpy"`` otherwise.
        """
        if self.is_available():
            return "mojo"
        if self.config.rust_fallback and _RUST_AVAILABLE:
            return "rust"
        return "numpy"
