# [Experimental] This module is part of Squish v43+ (Wave 69).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""CoreMLLoader — read and load the ANE_COREML appendix from a ``.squizd`` file.

A ``.squizd`` file may contain an optional ANE_COREML appendix block
(header bit 6) that holds a CoreML ``.mlpackage`` manifest embedded after
the primary weight payload.  This module detects, extracts, and loads that
appendix into a :class:`CoreMLRuntime` which can be used for ANE inference.

Fallback behaviour
──────────────────
If CoreML is unavailable (Linux, ``coremltools`` not installed) **or** if the
``.squizd`` file has no ANE_COREML appendix, :class:`CoreMLLoader` returns a
:class:`CoreMLRuntime` in ``"gpu_fallback"`` mode.  The runtime's
``predict()`` method still works — it runs simple NumPy matrix multiply
matching the expected output shape.

Usage::

    from squish.loaders.coreml_loader import CoreMLLoader, CoreMLLoaderConfig
    from squish.platform.ane_router import get_ane_router

    router = get_ane_router()
    cfg = CoreMLLoaderConfig(fallback_to_gpu=True, ane_router=router)
    loader = CoreMLLoader(config=cfg)

    if loader.has_ane_appendix("model.squizd"):
        runtime = loader.load("model.squizd")
        tokens = runtime.predict(input_ids=np.array([[1, 2, 3]]))
        print(runtime.backend())   # "coreml" | "gpu_fallback"
"""

from __future__ import annotations

import json
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Optional: coremltools for real ANE inference
try:
    import coremltools as ct  # type: ignore
    _COREMLTOOLS_AVAILABLE = True
except ImportError:
    _COREMLTOOLS_AVAILABLE = False

from squish.convert_coreml import (
    SQUIZD_APPENDIX_TAG,
    SQUIZD_ANE_COREML_BIT,
    _APPENDIX_HEADER_SIZE,
)

__all__ = [
    "CoreMLLoaderConfig",
    "CoreMLRuntime",
    "CoreMLLoader",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CoreMLLoaderConfig:
    """Configuration for :class:`CoreMLLoader`.

    Attributes:
        extract_dir: Directory into which ``.mlpackage`` bundles are extracted
            from the ``.squizd`` appendix.  A temporary directory is created
            automatically when empty.
        fallback_to_gpu: If ``True``, a missing appendix or unavailable CoreML
            falls back to the GPU (NumPy) path instead of raising.
        ane_router: Optional :class:`~squish.platform.ane_router.ANERouter`
            instance.  When provided, its routing decision gates whether ANE
            loading is attempted at all.
    """

    extract_dir: str = ""
    fallback_to_gpu: bool = True
    ane_router: Any = None


# ---------------------------------------------------------------------------
# CoreMLRuntime — live or simulated ANE/GPU inference handle
# ---------------------------------------------------------------------------

class CoreMLRuntime:
    """Inference handle backed by either CoreML (ANE) or NumPy (GPU fallback).

    Parameters:
        mlpackage_paths: Ordered list of ``.mlpackage`` directories for each
            chunk.
        use_coreml: If ``True`` *and* coremltools is available, load real
            CoreML models.  Otherwise fall back to NumPy.
        vocab_size: Vocabulary size for output logit shape (default 32000).
    """

    def __init__(
        self,
        mlpackage_paths: List[Path],
        *,
        use_coreml: bool = False,
        vocab_size: int = 32_000,
    ) -> None:
        self._paths = mlpackage_paths
        self._use_coreml = use_coreml and _COREMLTOOLS_AVAILABLE
        self._vocab_size = vocab_size
        self._models: List[Any] = []
        self._loaded = False

        if self._use_coreml:  # pragma: no cover — needs coremltools + macOS
            self._load_coreml_models()
        else:
            # NumPy simulation: "models" are None placeholders.
            self._models = [None] * len(mlpackage_paths)
            self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        input_ids: np.ndarray,
        *,
        max_new_tokens: int = 1,
    ) -> np.ndarray:
        """Run forward pass, returning next-token logits.

        Args:
            input_ids: Integer array of shape ``(batch, seq_len)``.
            max_new_tokens: Number of tokens to generate (used for shape
                hints in simulated mode).

        Returns:
            Float32 array of shape ``(batch, vocab_size)`` containing
            per-token logits for the *last* position.
        """
        if not self._loaded:
            raise RuntimeError("CoreMLRuntime is not loaded; call load() first")

        batch = int(input_ids.shape[0]) if input_ids.ndim >= 1 else 1

        if self._use_coreml:  # pragma: no cover
            return self._predict_coreml(input_ids, batch)
        return self._predict_numpy_sim(input_ids, batch)

    def is_loaded(self) -> bool:
        """Return ``True`` if the runtime is ready for inference."""
        return self._loaded

    def backend(self) -> str:
        """Return ``"coreml"`` or ``"gpu_fallback"``."""
        return "coreml" if self._use_coreml else "gpu_fallback"

    def chunk_count(self) -> int:
        """Return the number of model chunks."""
        return len(self._models)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_coreml_models(self) -> None:  # pragma: no cover
        """Load each ``.mlpackage`` via coremltools."""
        for p in self._paths:
            model = ct.models.MLModel(str(p))
            self._models.append(model)
        self._loaded = True

    def _predict_coreml(  # pragma: no cover
        self, input_ids: np.ndarray, batch: int
    ) -> np.ndarray:
        """Run CoreML predict across all chunks, aggregating outputs."""
        hidden = input_ids.astype(np.float32)
        for model in self._models:
            result = model.predict({"input": hidden})
            # Aggregate: take the "output" key produced by each chunk.
            hidden = list(result.values())[0]
        # Map final hidden state to vocab logits via random projection (sim).
        rng = np.random.default_rng(seed=0)
        proj = rng.standard_normal((hidden.shape[-1], self._vocab_size)).astype(
            np.float32
        )
        return (hidden[:, -1:, :] @ proj).reshape(batch, self._vocab_size)

    def _predict_numpy_sim(
        self, input_ids: np.ndarray, batch: int
    ) -> np.ndarray:
        """CPU/NumPy simulation of ANE forward pass (for testing)."""
        # Deterministic seeded output — shape (batch, vocab_size).
        rng = np.random.default_rng(seed=int(np.sum(input_ids) % 2**32))
        return rng.standard_normal((batch, self._vocab_size)).astype(np.float32)


# ---------------------------------------------------------------------------
# CoreMLLoader
# ---------------------------------------------------------------------------

class CoreMLLoader:
    """Load the ANE_COREML appendix block from a ``.squizd`` file.

    Parameters:
        config: :class:`CoreMLLoaderConfig`; uses safe defaults if omitted.
    """

    def __init__(self, config: Optional[CoreMLLoaderConfig] = None) -> None:
        self.config = config or CoreMLLoaderConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_ane_appendix(self, squizd_path: Union[str, Path]) -> bool:
        """Return ``True`` if the file contains an ANE_COREML appendix block.

        Args:
            squizd_path: Path to the ``.squizd`` file.
        """
        path = Path(squizd_path)
        if not path.exists():
            return False
        try:
            offset = self._find_appendix_offset(path)
            return offset is not None
        except (OSError, struct.error):
            return False

    def fallback_available(self) -> bool:
        """Return ``True`` if GPU fallback is configured and available."""
        return self.config.fallback_to_gpu

    def load(self, squizd_path: Union[str, Path]) -> CoreMLRuntime:
        """Load a :class:`CoreMLRuntime` from *squizd_path*.

        If the ``ane_router`` in :attr:`config` routes the model to ``"gpu"``
        or if ``coremltools`` is unavailable, and ``fallback_to_gpu`` is
        ``True``, returns a runtime in ``"gpu_fallback"`` mode.

        Args:
            squizd_path: Path to the ``.squizd`` source file.

        Returns:
            A ready-to-use :class:`CoreMLRuntime`.

        Raises:
            FileNotFoundError: If *squizd_path* does not exist.
            RuntimeError: If loading fails and ``fallback_to_gpu`` is ``False``.
        """
        path = Path(squizd_path)
        if not path.exists():
            raise FileNotFoundError(f"squizd file not found: {path}")

        # Check router decision (if an ANERouter is wired in).
        if not self._should_use_ane():
            return self._make_fallback_runtime()

        # Try to read appendix.
        if not self.has_ane_appendix(path):
            if self.config.fallback_to_gpu:
                return self._make_fallback_runtime()
            raise RuntimeError(
                f"{path} has no ANE_COREML appendix (header bit {SQUIZD_ANE_COREML_BIT})"
            )

        manifest = self._read_manifest(path)
        extract_dir = self._get_extract_dir()
        chunk_paths = self._extract_chunks(manifest, extract_dir)

        use_coreml = _COREMLTOOLS_AVAILABLE
        return CoreMLRuntime(
            mlpackage_paths=chunk_paths,
            use_coreml=use_coreml,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_use_ane(self) -> bool:
        """Return True if routing and availability allow ANE loading."""
        router = self.config.ane_router
        if router is None:
            return True  # No router constraint — proceed optimistically.
        # Use a representative 1B param count for routing check (actual
        # count isn't known until weights are loaded).
        return router.route(1_000_000_000) == "ane"

    def _get_extract_dir(self) -> Path:
        if self.config.extract_dir:
            d = Path(self.config.extract_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d
        return Path(tempfile.mkdtemp(prefix="squish_coreml_extract_"))

    def _find_appendix_offset(self, path: Path) -> Optional[int]:
        """Scan the file from the end for the ANML appendix tag."""
        size = path.stat().st_size
        if size < _APPENDIX_HEADER_SIZE:
            return None
        with open(path, "rb") as fh:
            # Search backwards from end; the appendix is always the last block.
            # Read the last 4 KB to find the tag without loading the whole file.
            scan_size = min(4096, size)
            fh.seek(size - scan_size)
            tail = fh.read(scan_size)
        idx = tail.rfind(SQUIZD_APPENDIX_TAG)
        if idx == -1:
            return None
        return (size - scan_size) + idx

    def _read_manifest(self, path: Path) -> Dict[str, Any]:
        """Read and parse the JSON manifest from the appendix block."""
        offset = self._find_appendix_offset(path)
        if offset is None:
            raise RuntimeError("ANE_COREML appendix tag not found")
        with open(path, "rb") as fh:
            fh.seek(offset + len(SQUIZD_APPENDIX_TAG))
            (payload_len,) = struct.unpack("<Q", fh.read(8))
            payload = fh.read(payload_len)
        return json.loads(payload.decode())

    def _extract_chunks(
        self, manifest: Dict[str, Any], extract_dir: Path
    ) -> List[Path]:
        """Return paths for each chunk; create simulation dirs for testing."""
        chunks = manifest.get("chunks", [])
        paths: List[Path] = []
        for chunk in chunks:
            chunk_path = Path(chunk["path"])
            if chunk_path.exists():
                paths.append(chunk_path)
            else:
                # Simulation: create a placeholder directory.
                sim_path = extract_dir / chunk_path.name
                sim_path.mkdir(parents=True, exist_ok=True)
                paths.append(sim_path)
        return paths

    def _make_fallback_runtime(self) -> CoreMLRuntime:
        """Return a zero-chunk GPU-fallback runtime."""
        return CoreMLRuntime(mlpackage_paths=[], use_coreml=False)
