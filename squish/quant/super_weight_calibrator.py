"""squish/quant/super_weight_calibrator.py

Super Weight Calibrator — statistically identifies the coordinates of
"super weights" inside a transformer model's weight tensors.

Background
----------
Frantar et al. (arXiv:2411.07191) showed that within models of tens of billions
of parameters a small number of scalar weight elements — sometimes a single one —
exercise disproportionately large influence over linguistic coherence.  Degrading
just one such element can drive zero-shot accuracy to random-guess levels.

These "super weights" are identified by an unusually high ratio of the element's
absolute value to the mean absolute value of its row (the "outlier ratio"):

    ratio_ij = |w_ij| / (mean(|row_i|) + eps)

Elements whose ratio exceeds a configurable threshold are recorded as super
weights that must be preserved in FP16 during subsequent ternary quantization.

The calibrator operates on CPU numpy arrays loaded directly from safetensors
shards — no MLX, no Metal, no model architecture instantiation required.  This
makes it safe to run on any model regardless of size.

Usage::

    from squish.quant.super_weight_calibrator import (
        SuperWeightConfig,
        SuperWeightCalibrator,
        calibrate_from_dir,
    )

    coords = calibrate_from_dir(
        model_dir="~/models/Qwen3-8B-bf16",
        threshold=100.0,
        verbose=True,
    )
    print(f"Found {len(coords)} super weight(s)")
    for c in coords[:5]:
        print(c)
"""

from __future__ import annotations

__all__ = [
    "SuperWeightCoord",
    "SuperWeightConfig",
    "SuperWeightCalibrator",
    "calibrate_from_dir",
]

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# SuperWeightCoord — a single identified super weight
# ---------------------------------------------------------------------------

@dataclass(order=True, frozen=True)
class SuperWeightCoord:
    """The exact location and metadata of one identified super weight.

    Attributes:
        tensor_name: Full dot-notation tensor name as it appears in the model's
            safetensors files, e.g. ``"model.layers.15.mlp.down_proj"``.
        row: Row index within the 2-D view of the tensor (after reshaping to
            ``(n_rows, n_cols)``).
        col: Column index within the 2-D view.  This determines the activation
            channel that must be preserved; when using asymmetric ternary
            compression, the entire column ``col`` of this tensor is stored
            at FP16 precision rather than in the ternary {-1, 0, +1} alphabet.
        value: The original float32 value of the weight element.
        ratio: The outlier ratio ``|value| / (mean(|row|) + eps)`` that caused
            this element to be identified.
        original_shape: Shape of the tensor *before* 2-D reshaping, stored as a
            tuple for reference when mapping coordinates back to native layout.
    """

    tensor_name:    str
    row:            int
    col:            int
    value:          float
    ratio:          float
    original_shape: tuple = field(compare=False)

    @property
    def coord_key(self) -> str:
        """Human-readable coordinate string, e.g. ``"model.layers.15.mlp.down_proj[1234,3968]"``."""
        return f"{self.tensor_name}[{self.row},{self.col}]"


# ---------------------------------------------------------------------------
# SuperWeightConfig
# ---------------------------------------------------------------------------

@dataclass
class SuperWeightConfig:
    """Configuration for super-weight detection.

    Attributes:
        threshold: Minimum outlier ratio for an element to be classified as a
            super weight in a multi-row (2-D) tensor.  The paper reports ratios
            of several hundred for the most critical super weights in LLaMA-7B.
            Empirically, 20 is a practical lower bound for most models; raise to
            100+ to capture only the most extreme outliers.
        threshold_1d: Minimum outlier ratio for single-row tensors (1-D weights
            such as layer-norm scale vectors and attention biases, which are
            reshaped to ``(1, n_cols)`` for scanning).  These tensors have no
            row-averaging effect, so genuine outliers appear at lower ratios
            than in large weight matrices.  A value of ``5.0`` captures elements
            that are ≥ 5× the mean absolute value of the tensor — sufficient to
            protect the extreme-value channels that cause coherence collapse when
            ternary-quantised.
        max_per_tensor: Maximum number of super weight coordinates recorded per
            multi-row tensor.  When a tensor has many elements above ``threshold``
            (rare), only the top ``max_per_tensor`` by ratio are kept.  Set to 0
            for no limit.
        max_per_tensor_1d: Maximum number of super weight coordinates recorded
            per 1-D tensor.  Defaults to ``0`` (no limit) because 1-D tensors
            are small — protecting all of their outliers adds negligible storage.
        skip_patterns: Tensor name substrings to skip during scanning.
            Embedding tables and LM head are rarely the source of super weights
            and are very large, so they are skipped by default.
        min_2d_cols: Tensors whose second dimension (after reshaping to 2-D) is
            smaller than this value are skipped.  Does not apply to 1-D tensors,
            which are always scanned using ``threshold_1d``.
    """

    threshold:      float       = 100.0
    threshold_1d:   float       = 5.0
    max_per_tensor: int         = 8
    max_per_tensor_1d: int      = 0   # 0 = no limit; all 1-D outliers are kept
    skip_patterns:  list[str]   = field(default_factory=lambda: [
        "embed_tokens",
        "lm_head",
        "embed_positions",
    ])
    min_2d_cols:    int         = 64


# ---------------------------------------------------------------------------
# SuperWeightCalibrator
# ---------------------------------------------------------------------------

class SuperWeightCalibrator:
    """Identifies super weight coordinates from a flat mapping of weight tensors.

    The calibrator expects pre-loaded float32 numpy arrays in a dict keyed by
    the original tensor name (dot-notation).  It does *not* load files itself;
    use :func:`calibrate_from_dir` for filesystem-based workflows.

    Args:
        config: :class:`SuperWeightConfig` controlling detection thresholds.
    """

    def __init__(self, config: SuperWeightConfig | None = None) -> None:
        self.config = config or SuperWeightConfig()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def scan_weights(
        self, weights: dict[str, np.ndarray]
    ) -> list[SuperWeightCoord]:
        """Scan all weight tensors and return detected super weight coordinates.

        Args:
            weights: Dict mapping tensor name to float32 numpy array of any shape.

        Returns:
            List of :class:`SuperWeightCoord`, sorted by ratio descending.
        """
        cfg = self.config
        coords: list[SuperWeightCoord] = []

        for name, arr in weights.items():
            # Skip patterns
            if any(pat in name for pat in cfg.skip_patterns):
                continue

            # Must be at least 1-D
            arr_f32 = np.asarray(arr, dtype=np.float32)
            if arr_f32.ndim == 0:
                continue

            original_shape = tuple(arr_f32.shape)

            # Reshape to 2-D: (n_rows, n_cols)
            if arr_f32.ndim == 1:
                flat = arr_f32.reshape(1, -1)
            else:
                flat = arr_f32.reshape(-1, arr_f32.shape[-1])

            n_rows, n_cols = flat.shape

            # 1-D tensors (reshaped to a single row) are always scanned with
            # threshold_1d — their outlier ratios are naturally lower because
            # there is no cross-row averaging to amplify the relative magnitude.
            is_1d = (arr_f32.ndim == 1)

            # For multi-row tensors, enforce minimum column count guard
            if not is_1d and n_cols < cfg.min_2d_cols:
                continue

            found = self._find_super_weights(name, flat, original_shape, is_1d=is_1d)
            coords.extend(found)

        # Sort globally by ratio descending
        coords.sort(key=lambda c: c.ratio, reverse=True)
        return coords

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _find_super_weights(
        self,
        name: str,
        flat: np.ndarray,
        original_shape: tuple,
        is_1d: bool = False,
    ) -> list[SuperWeightCoord]:
        """Find super weight candidates within a single 2-D weight matrix.

        Computes the per-row outlier ratio matrix:

            ratio_ij = |w_ij| / (mean(|row_i|) + eps)

        and returns all positions whose ratio exceeds the configured threshold.
        For single-row (1-D) tensors ``is_1d=True`` selects the lower
        ``config.threshold_1d`` instead of ``config.threshold``, because 1-D
        tensors have no cross-row averaging effect and their genuine outliers
        naturally appear at smaller ratios.

        Args:
            name:           Tensor name (for inclusion in each :class:`SuperWeightCoord`).
            flat:           Float32 array of shape ``(n_rows, n_cols)``.
            original_shape: Original shape before flattening to 2-D.
            is_1d:          True when the original tensor was 1-D (reshaped to
                            ``(1, n_cols)``).  Selects ``threshold_1d``.

        Returns:
            List of ``SuperWeightCoord`` for this tensor, sorted by ratio descending.
            At most ``config.max_per_tensor`` entries if that limit is non-zero.
        """
        cfg = self.config
        threshold = cfg.threshold_1d if is_1d else cfg.threshold

        abs_flat = np.abs(flat)

        # Per-row mean absolute value with epsilon guard against all-zero rows
        row_mean = abs_flat.mean(axis=1) + 1e-9   # (n_rows,)

        # Outlier ratio matrix: (n_rows, n_cols)
        ratio_matrix = abs_flat / row_mean[:, np.newaxis]

        # Mask positions above threshold
        above = ratio_matrix > threshold
        if not above.any():
            return []

        # Get coordinates of all above-threshold elements
        rows_idx, cols_idx = np.where(above)  # each is a 1-D array of indices
        ratios  = ratio_matrix[rows_idx, cols_idx]
        values  = flat[rows_idx, cols_idx]

        # Sort by ratio descending
        order = np.argsort(ratios)[::-1]
        rows_idx = rows_idx[order]
        cols_idx = cols_idx[order]
        ratios   = ratios[order]
        values   = values[order]

        # Apply per-tensor limit (1-D tensors use max_per_tensor_1d, default unlimited)
        limit = cfg.max_per_tensor_1d if is_1d else cfg.max_per_tensor
        if limit > 0:
            rows_idx = rows_idx[:limit]
            cols_idx = cols_idx[:limit]
            ratios   = ratios[:limit]
            values   = values[:limit]

        coords = [
            SuperWeightCoord(
                tensor_name=name,
                row=int(r),
                col=int(c),
                value=float(v),
                ratio=float(rt),
                original_shape=original_shape,
            )
            for r, c, v, rt in zip(rows_idx, cols_idx, values, ratios)
        ]
        return coords


# ---------------------------------------------------------------------------
# calibrate_from_dir — filesystem entry point
# ---------------------------------------------------------------------------

def calibrate_from_dir(
    model_dir: str | Path,
    config: SuperWeightConfig | None = None,
    verbose: bool = False,
) -> list[SuperWeightCoord]:
    """Scan all safetensors shards in *model_dir* for super weights.

    This is the primary public entry point for the calibration workflow.
    It loads each shard sequentially, scans the weights, and releases the
    shard from RAM before moving to the next one.

    Args:
        model_dir: Path to the directory containing ``*.safetensors`` files and
            ``config.json``.  BF16 and FP16 models are both supported.
        config:    :class:`SuperWeightConfig` controlling detection thresholds.
            Defaults to ``SuperWeightConfig()`` (threshold=100).
        verbose:   Print per-shard progress to stdout.

    Returns:
        List of :class:`SuperWeightCoord`, sorted globally by ratio descending.

    Raises:
        FileNotFoundError: If no ``.safetensors`` files are found in *model_dir*.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}"
        )

    cfg = config or SuperWeightConfig()
    calibrator = SuperWeightCalibrator(cfg)
    all_coords: list[SuperWeightCoord] = []

    for shard_idx, shard_path in enumerate(shards, 1):
        if verbose:
            print(f"  [{shard_idx}/{len(shards)}] Scanning {shard_path.name} …")

        shard_weights = _load_shard_f32(shard_path)
        coords = calibrator.scan_weights(shard_weights)
        all_coords.extend(coords)

        if verbose and coords:
            print(f"    Found {len(coords)} super weight(s) in this shard")
            for c in coords[:3]:
                print(f"      {c.coord_key}  ratio={c.ratio:.1f}  value={c.value:.4f}")

        del shard_weights

    # Global sort by ratio descending across all shards
    all_coords.sort(key=lambda c: c.ratio, reverse=True)

    if verbose:
        print(f"\n  Total super weights found: {len(all_coords)}")
        if all_coords:
            top = all_coords[0]
            print(f"  Highest ratio: {top.coord_key}  ratio={top.ratio:.1f}")

    return all_coords


# ---------------------------------------------------------------------------
# Internal shard loader (no MLX, CPU numpy only)
# ---------------------------------------------------------------------------

def _load_shard_f32(shard_path: Path) -> dict[str, np.ndarray]:
    """Load one safetensors shard as float32 numpy arrays (CPU only, no Metal).

    Tries ``safetensors.numpy`` first because it is zero-copy for F32/F16.
    Falls back to ``mlx.core`` on CPU for BF16 tensors that
    ``safetensors.numpy`` cannot parse.
    """
    try:
        from safetensors.numpy import load_file as _st_load
        raw = _st_load(str(shard_path))
        return {name: arr.astype(np.float32) for name, arr in raw.items()}
    except Exception:
        pass

    import mlx.core as mx
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)  # type: ignore[arg-type]
    try:
        raw = mx.load(str(shard_path))  # type: ignore[assignment]
        return {
            name: np.array(arr.astype(mx.float32))  # type: ignore[call-overload]
            for name, arr in raw.items()
        }
    finally:
        mx.set_default_device(prev)
