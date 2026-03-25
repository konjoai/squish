"""squish/serving/local_model_scanner.py — Scan for locally-installed models.

Detects models installed via Squish, Ollama, and LM Studio/Hugging Face and
presents them in a unified ``LocalModel`` format.

Public API
──────────
LocalModel          — dataclass for a single discovered model
LocalModelScanner   — scans all sources and merges results
"""
from __future__ import annotations

__all__ = ["LocalModel", "LocalModelScanner"]

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# LocalModel dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocalModel:
    """A locally-installed model discovered by the scanner.

    Attributes
    ----------
    name:       Canonical model name (e.g. ``qwen3:8b`` or ``llama3.1:8b``).
    path:       Absolute path to the model directory or file.
    source:     Origin scanner: ``"squish"``, ``"ollama"``, ``"lm_studio"``,
                or ``"gguf"``.
    size_bytes: Approximate on-disk size in bytes (0 if unknown).
    family:     Model family guess (e.g. ``"qwen"``, ``"llama"``).
    params:     Parameter count string (e.g. ``"7B"`` or ``""`` if unknown).
    """
    name:       str
    path:       Path
    source:     str
    size_bytes: int = 0
    family:     str = ""
    params:     str = ""

    def __post_init__(self) -> None:
        self.path = Path(self.path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Return total on-disk size of a directory tree in bytes."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


def _guess_family(name: str) -> str:
    name_l = name.lower()
    for family in ("qwen", "llama", "gemma", "phi", "mistral", "deepseek",
                   "mixtral", "falcon", "vicuna", "alpaca"):
        if family in name_l:
            return family
    return ""


def _guess_params(name: str) -> str:
    import re
    m = re.search(r"(\d+\.?\d*)b", name.lower())
    return f"{m.group(1)}B" if m else ""


# ---------------------------------------------------------------------------
# LocalModelScanner
# ---------------------------------------------------------------------------

class LocalModelScanner:
    """Scan multiple sources for locally-installed models.

    Parameters
    ----------
    squish_models_dir:
        Directory containing Squish-compressed models.
        Defaults to ``~/models``.
    ollama_manifests_dir:
        Root of the Ollama manifests tree.
        Defaults to ``~/.ollama/models/manifests/registry.ollama.ai/library``.
    lm_studio_dir:
        Root of the LM Studio / Hugging Face cache.
        Defaults to ``~/.cache/lm-studio/models``.
    """

    def __init__(
        self,
        squish_models_dir: Optional[Path] = None,
        ollama_manifests_dir: Optional[Path] = None,
        lm_studio_dir: Optional[Path] = None,
    ) -> None:
        home = Path.home()
        self._squish_dir  = squish_models_dir   or (home / "models")
        self._ollama_dir  = ollama_manifests_dir or (
            home / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
        )
        self._lm_studio_dir = lm_studio_dir or (home / ".cache" / "lm-studio" / "models")

    # ------------------------------------------------------------------
    # Individual scanners
    # ------------------------------------------------------------------

    def scan_squish(self) -> list[LocalModel]:
        """Return models installed by Squish (``~/models/`` subdirs)."""
        if not self._squish_dir.is_dir():
            return []
        models: list[LocalModel] = []
        for d in sorted(self._squish_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            # Convert dir name to canonical id (e.g. "Qwen3-8B-bf16" → "qwen3:8b")
            name = _dir_to_canonical(d.name)
            models.append(LocalModel(
                name=name,
                path=d,
                source="squish",
                size_bytes=_dir_size(d),
                family=_guess_family(d.name),
                params=_guess_params(d.name),
            ))
        return models

    def scan_ollama(self) -> list[LocalModel]:
        """Return models found in the Ollama manifest directory."""
        if not self._ollama_dir.is_dir():
            return []
        models: list[LocalModel] = []
        for model_dir in sorted(self._ollama_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for tag_file in model_dir.iterdir():
                if not tag_file.is_file():
                    continue
                tag  = tag_file.name
                name = f"{model_name}:{tag}"
                size = 0
                try:
                    manifest = json.loads(tag_file.read_text(encoding="utf-8"))
                    for layer in manifest.get("layers", []):
                        size += layer.get("size", 0)
                except Exception:
                    pass
                models.append(LocalModel(
                    name=name,
                    path=tag_file,
                    source="ollama",
                    size_bytes=size,
                    family=_guess_family(model_name),
                    params=_guess_params(model_name),
                ))
        return models

    def scan_lm_studio(self) -> list[LocalModel]:
        """Return GGUF / safetensors models found in the LM Studio cache."""
        if not self._lm_studio_dir.is_dir():
            return []
        models: list[LocalModel] = []
        for gguf in self._lm_studio_dir.rglob("*.gguf"):
            models.append(LocalModel(
                name=gguf.stem.lower(),
                path=gguf,
                source="gguf",
                size_bytes=gguf.stat().st_size if gguf.exists() else 0,
                family=_guess_family(gguf.stem),
                params=_guess_params(gguf.stem),
            ))
        return models

    def find_all(self) -> list[LocalModel]:
        """Return merged, deduplicated results from all scanners.

        Deduplication is by canonical ``name`` — the first occurrence wins
        (squish > ollama > lm_studio priority order).
        """
        seen: set[str] = set()
        merged: list[LocalModel] = []
        for model in (
            self.scan_squish() + self.scan_ollama() + self.scan_lm_studio()
        ):
            if model.name not in seen:
                seen.add(model.name)
                merged.append(model)
        return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dir_to_canonical(dir_name: str) -> str:
    """Convert a Squish model directory name to a canonical ``family:size`` id.

    Examples::

        "Qwen3-8B-bf16"            → "qwen3:8b"
        "Llama-3.1-8B-Instruct-bf16" → "llama3.1:8b"
    """
    import re
    name = dir_name.lower()
    # Strip common suffixes
    for suffix in ("-bf16", "-int4", "-int2", "-int8", "-instruct", "-chat",
                   "-squished", "-squish", "-mlx", "-gguf"):
        name = name.replace(suffix, "")
    # Replace last hyphen before a size pattern with colon
    name = re.sub(r"-(\d+\.?\d*b)(\b|$)", r":\1", name)
    # Clean up remaining hyphens between version/name parts → keep single hyphen
    name = re.sub(r"-+", "-", name).strip("-")
    return name
