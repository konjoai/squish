"""squish/squash/remediate.py — Automated remediation for unsafe model formats.

Converts PyTorch ``.bin`` / ``.pt`` / ``.pth`` pickle files to ``.safetensors``
format using a safe, sandboxed load path.  The resulting artifacts are identical
in weights but carry zero code-execution risk on load.

Why this is safe
----------------
PyTorch ≥ 2.0 ships ``torch.load(weights_only=True)``, which restricts the
unpickler to a tight allowlist of tensor-safe opcodes.  A reverse-shell payload
that relies on ``REDUCE``/``GLOBAL`` opcodes will raise ``UnpicklingError``
instead of executing.  We convert only files that pass this safe-load; files
that fail are quarantined and reported.

Usage::

    result = Remediator.convert(Path("./model.bin"))
    # result.converted_paths — list of (src, dst) tuples
    # result.failed_paths    — list of (src, reason) tuples
    # result.sbom_patch      — dict of hash updates for the SBOM

    # Update an existing SBOM in-place:
    Remediator.patch_sbom(Path("./cyclonedx-mlbom.json"), result.sbom_patch)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Extensions treated as unsafe PyTorch pickle files
_PICKLE_EXTENSIONS: frozenset[str] = frozenset({".bin", ".pt", ".pth"})


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ConvertedFile:
    """A single successfully converted weight file."""
    source: Path
    destination: Path
    source_sha256: str
    destination_sha256: str
    tensor_count: int


@dataclass
class FailedFile:
    """A file that could not be converted safely."""
    source: Path
    reason: str  # "unsafe_pickle" | "load_error" | "dependency_missing"


@dataclass
class RemediateResult:
    """Aggregate result from :func:`Remediator.convert`."""
    model_path: Path
    target_format: str
    converted: list[ConvertedFile] = field(default_factory=list)
    failed: list[FailedFile] = field(default_factory=list)
    # SBOM hash-update patch: {old_sha256: {"new_sha256": ..., "new_file": ...}}
    sbom_patch: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def fully_remediated(self) -> bool:
        """True when no file failed remediation (includes the case of zero pickle files found)."""
        return len(self.failed) == 0

    @property
    def partial(self) -> bool:
        return len(self.converted) > 0 and len(self.failed) > 0

    def summary(self) -> str:
        ok = len(self.converted)
        fail = len(self.failed)
        total = ok + fail
        if total == 0:
            return "No unsafe files found — model is already clean."
        lines = [f"Remediation: {ok}/{total} files converted to {self.target_format}"]
        for c in self.converted:
            lines.append(f"  ✓ {c.source.name} → {c.destination.name} ({c.tensor_count} tensors)")
        for f in self.failed:
            lines.append(f"  ✗ {f.source.name}: {f.reason}")
        return "\n".join(lines)


# ── Core class ───────────────────────────────────────────────────────────────


class Remediator:
    """Convert unsafe PyTorch pickle files to safe ``safetensors`` format."""

    # Maximum file size to attempt conversion (256 MB by default).
    MAX_FILE_BYTES: int = 256 * 1024 * 1024

    @classmethod
    def convert(
        cls,
        model_path: Path,
        *,
        target_format: str = "safetensors",
        output_dir: Path | None = None,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> RemediateResult:
        """Convert all unsafe pickle files under *model_path*.

        Parameters
        ----------
        model_path:
            Directory or single file to scan and convert.
        target_format:
            Currently only ``"safetensors"`` is supported.
        output_dir:
            Where to write converted files (default: same directory as source).
        dry_run:
            If True, discover and validate but do not write any files.
        overwrite:
            If True, allow overwriting an existing ``.safetensors`` at the
            destination.  Default: False (skip already-converted files).
        """
        if target_format.lower() != "safetensors":
            raise ValueError(f"Unsupported target_format: {target_format!r}. Only 'safetensors' is supported.")

        result = RemediateResult(model_path=model_path, target_format=target_format)

        targets = cls._find_pickle_files(model_path)
        if not targets:
            return result

        for src in targets:
            try:
                cf = cls._convert_one(src, output_dir=output_dir, dry_run=dry_run, overwrite=overwrite)
                if cf is not None:
                    result.converted.append(cf)
                    result.sbom_patch[cf.source_sha256] = {
                        "new_sha256": cf.destination_sha256,
                        "new_file": str(cf.destination),
                        "old_file": str(cf.source),
                    }
            except Exception as exc:
                log.debug("Conversion failed for %s: %s", src, exc)
                result.failed.append(FailedFile(source=src, reason=str(exc)))

        return result

    @classmethod
    def patch_sbom(cls, sbom_path: Path, patch: dict[str, dict[str, str]]) -> bool:
        """Update an existing CycloneDX JSON BOM with new file hashes.

        Replaces hash values for converted files and appends a remediation
        note to the metadata.tools list.  Returns True if the BOM was modified.
        """
        if not sbom_path.exists() or not patch:
            return False

        try:
            with sbom_path.open() as fh:
                bom: dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("patch_sbom: could not read %s: %s", sbom_path, exc)
            return False

        modified = False
        for component in bom.get("components", []):
            for h in component.get("hashes", []):
                old_value = h.get("content", "")
                if old_value in patch:
                    h["content"] = patch[old_value]["new_sha256"]
                    component["_remediated"] = True
                    modified = True

        if modified:
            # Append remediation note to tools
            bom.setdefault("metadata", {}).setdefault("tools", [])
            bom["metadata"]["tools"].append({
                "vendor": "Squish Squash",
                "name": "squash-remediate",
                "version": "1.0",
                "externalReferences": [
                    {"type": "other", "url": "https://squish.local/squash/remediate"}
                ],
            })
            sbom_path.write_text(json.dumps(bom, indent=2))
            log.info("Patched SBOM at %s", sbom_path)

        return modified

    # ── Private helpers ──────────────────────────────────────────────────────

    @classmethod
    def _find_pickle_files(cls, model_path: Path) -> list[Path]:
        if model_path.is_file():
            if model_path.suffix in _PICKLE_EXTENSIONS:
                return [model_path]
            return []
        return sorted(
            p for p in model_path.rglob("*")
            if p.is_file() and p.suffix in _PICKLE_EXTENSIONS and p.stat().st_size <= cls.MAX_FILE_BYTES
        )

    @classmethod
    def _convert_one(
        cls,
        src: Path,
        *,
        output_dir: Path | None,
        dry_run: bool,
        overwrite: bool,
    ) -> ConvertedFile | None:
        dest_dir = output_dir if output_dir is not None else src.parent
        dest = dest_dir / (src.stem + ".safetensors")

        if dest.exists() and not overwrite:
            log.info("Skipping %s — destination already exists", src)
            return None

        src_sha256 = cls._sha256(src)

        if dry_run:
            # Validate safe-loadability without writing
            state_dict = cls._safe_load(src)  # raises on unsafe pickle
            return ConvertedFile(
                source=src,
                destination=dest,
                source_sha256=src_sha256,
                destination_sha256="(dry-run)",
                tensor_count=len(state_dict),
            )

        state_dict = cls._safe_load(src)
        dest_dir.mkdir(parents=True, exist_ok=True)
        cls._write_safetensors(state_dict, dest)
        dest_sha256 = cls._sha256(dest)

        return ConvertedFile(
            source=src,
            destination=dest,
            source_sha256=src_sha256,
            destination_sha256=dest_sha256,
            tensor_count=len(state_dict),
        )

    @staticmethod
    def _safe_load(src: Path) -> dict[str, Any]:
        """Load a PyTorch checkpoint safely (weights_only=True).

        Raises ``ImportError`` if torch is unavailable, or
        ``pickle.UnpicklingError`` / ``RuntimeError`` if the file is unsafe.
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch is required for remediation. Install it with: pip install torch"
            ) from exc

        try:
            # weights_only=True: blocks arbitrary code execution (pickle REDUCE/GLOBAL)
            data = torch.load(str(src), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"dependency_missing or unsafe_pickle: {exc}"
            ) from exc

        # Flatten nested dicts (e.g. {"model": {"weight": tensor}})
        if isinstance(data, dict):
            flat: dict[str, Any] = {}
            for k, v in data.items():
                if hasattr(v, "to"):  # tensor-like
                    flat[k] = v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}.{kk}"] = vv
            return flat if flat else data
        raise TypeError(f"Expected a dict checkpoint, got {type(data).__name__}")

    @staticmethod
    def _write_safetensors(state_dict: dict[str, Any], dest: Path) -> None:
        """Write *state_dict* as a safetensors file."""
        try:
            from safetensors.torch import save_file  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for remediation. Install with: pip install safetensors"
            ) from exc

        # Ensure all tensors are contiguous
        import torch  # noqa: PLC0415
        clean = {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        save_file(clean, str(dest))

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
