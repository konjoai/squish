"""squish/quant/super_weight_registry.py

Super Weight Registry — JSON-backed persistence and lookup for super weight
coordinate maps produced by :mod:`squish.quant.super_weight_calibrator`.

The registry stores one JSON file per model.  During asymmetric ternary
compression (:mod:`squish.convert`), the registry is consulted for each tensor
to determine which columns must be preserved at FP16 precision.

File format (schema v1)::

    {
      "schema_version": 1,
      "model_name": "Qwen3-8B",
      "model_dir": "/Users/…/Qwen3-8B-bf16",
      "threshold": 100.0,
      "calibrated_at": "2026-03-14T12:00:00",
      "super_weights": [
        {
          "tensor_name": "model.layers.15.mlp.down_proj",
          "row": 1234,
          "col": 3968,
          "value": 2.3451,
          "ratio": 487.2,
          "original_shape": [4096, 11008]
        },
        …
      ]
    }

Usage::

    from squish.quant.super_weight_registry import (
        SuperWeightRegistry,
        save_registry,
        load_registry,
    )

    # After running the calibrator:
    from squish.quant.super_weight_calibrator import calibrate_from_dir
    coords = calibrate_from_dir("~/models/Qwen3-8B-bf16", verbose=True)

    registry = SuperWeightRegistry.from_coords(
        coords,
        model_dir="~/models/Qwen3-8B-bf16",
        threshold=100.0,
    )
    save_registry(registry, Path("~/.squish/super_weights/Qwen3-8B.json"))

    # During compression:
    registry = load_registry(Path("~/.squish/super_weights/Qwen3-8B.json"))
    protected_cols = registry.protected_columns("model.layers.15.mlp.down_proj")
    # protected_cols → [3968]
"""

from __future__ import annotations

__all__ = [
    "SuperWeightRegistry",
    "save_registry",
    "load_registry",
]

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# SuperWeightRegistry
# ---------------------------------------------------------------------------

@dataclass
class SuperWeightRegistry:
    """Immutable registry of super weight coordinates for one model.

    Use :meth:`from_coords` to build from a
    :class:`~squish.quant.super_weight_calibrator.SuperWeightCoord` list, or
    :func:`load_registry` to read back from disk.

    Attributes:
        schema_version: File format version (always ``1`` for this release).
        model_name:     Human-readable model name derived from the directory
                        basename (e.g. ``"Qwen3-8B-bf16"``).
        model_dir:      Absolute filesystem path to the source model directory.
        threshold:      The outlier-ratio threshold used during calibration.
        calibrated_at:  UTC ISO-8601 timestamp of when the registry was created.
        super_weights:  List of dicts, each with keys ``tensor_name``, ``row``,
                        ``col``, ``value``, ``ratio``, ``original_shape``.
    """

    schema_version: int
    model_name:     str
    model_dir:      str
    threshold:      float
    calibrated_at:  str
    super_weights:  list[dict]   = field(default_factory=list)

    # ----------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------

    @classmethod
    def from_coords(
        cls,
        coords,   # list[SuperWeightCoord] — imported lazily to avoid circular
        model_dir: str | Path,
        threshold: float = 100.0,
    ) -> SuperWeightRegistry:
        """Build a registry from a list of calibration coordinates.

        Args:
            coords:    Output of
                :func:`~squish.quant.super_weight_calibrator.calibrate_from_dir`.
            model_dir: Path to the model directory (used to derive model name).
            threshold: The threshold that was used during calibration.

        Returns:
            A fully populated :class:`SuperWeightRegistry`.
        """
        model_dir = Path(model_dir).expanduser().resolve()
        sw_list = [
            {
                "tensor_name":    c.tensor_name,
                "row":            c.row,
                "col":            c.col,
                "value":          float(c.value),
                "ratio":          float(c.ratio),
                "original_shape": list(c.original_shape),
            }
            for c in coords
        ]
        return cls(
            schema_version=_SCHEMA_VERSION,
            model_name=model_dir.name,
            model_dir=str(model_dir),
            threshold=threshold,
            calibrated_at=datetime.now(timezone.utc).isoformat(),
            super_weights=sw_list,
        )

    # ----------------------------------------------------------------
    # Lookup helpers
    # ----------------------------------------------------------------

    def protected_columns(self, tensor_name: str) -> list[int]:
        """Return the list of column indices that must be preserved in FP16
        for the given tensor.

        A column is protected if it contains at least one super weight.
        Duplicates are removed and the list is sorted ascending.

        Args:
            tensor_name: Full dot-notation tensor name.

        Returns:
            Sorted list of unique column indices.  Empty list if no super
            weights are registered for this tensor.
        """
        cols = {
            sw["col"]
            for sw in self.super_weights
            if sw["tensor_name"] == tensor_name
        }
        return sorted(cols)

    def protected_mask(
        self, tensor_name: str, shape: tuple
    ) -> np.ndarray:
        """Return a boolean numpy mask of shape *shape* where ``True`` marks
        positions that must be preserved at FP16.

        The mask is computed by marking entire columns containing super weights.
        When *shape* is multidimensional, it is first viewed as
        ``(n_rows, n_cols)`` where ``n_cols = shape[-1]``.

        Args:
            tensor_name: Full dot-notation tensor name.
            shape:       Shape of the weight tensor (any number of dimensions).

        Returns:
            Boolean numpy array with the same shape; ``True`` = protected.
        """
        import numpy as np

        cols = self.protected_columns(tensor_name)
        mask = np.zeros(shape, dtype=bool)
        if not cols:
            return mask

        # Work on the last dimension (n_cols after reshape to 2-D)
        # Broadcasting: mark full columns by index
        n_cols = shape[-1]
        for col in cols:
            if 0 <= col < n_cols:
                mask[..., col] = True
        return mask

    def has_tensor(self, tensor_name: str) -> bool:
        """Return True if any super weight is registered for the given tensor name."""
        return any(sw["tensor_name"] == tensor_name for sw in self.super_weights)

    def summary(self) -> str:
        """Return a human-readable one-liner summary."""
        n_tensors = len({sw["tensor_name"] for sw in self.super_weights})
        return (
            f"SuperWeightRegistry({self.model_name}): "
            f"{len(self.super_weights)} coords across {n_tensors} tensor(s), "
            f"threshold={self.threshold}"
        )

    def __len__(self) -> int:
        return len(self.super_weights)


# ---------------------------------------------------------------------------
# save_registry / load_registry
# ---------------------------------------------------------------------------

def save_registry(registry: SuperWeightRegistry, path: str | Path) -> None:
    """Serialise a :class:`SuperWeightRegistry` to a JSON file.

    The parent directory is created if it does not exist.

    Args:
        registry: Registry to serialise.
        path:     Destination file path (``*.json``).
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": registry.schema_version,
        "model_name":     registry.model_name,
        "model_dir":      registry.model_dir,
        "threshold":      registry.threshold,
        "calibrated_at":  registry.calibrated_at,
        "super_weights":  registry.super_weights,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_registry(path: str | Path) -> SuperWeightRegistry:
    """Load a :class:`SuperWeightRegistry` from a JSON file.

    Args:
        path: Path to a registry file previously written by :func:`save_registry`.

    Returns:
        A :class:`SuperWeightRegistry` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not a valid Squish super-weight registry.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Super weight registry not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported registry schema version: "
            f"{data.get('schema_version')} (expected {_SCHEMA_VERSION})"
        )

    return SuperWeightRegistry(
        schema_version=data["schema_version"],
        model_name=data["model_name"],
        model_dir=data["model_dir"],
        threshold=float(data["threshold"]),
        calibrated_at=data["calibrated_at"],
        super_weights=data.get("super_weights", []),
    )
