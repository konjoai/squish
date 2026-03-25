"""
AdaptiveKVTC — Per-Layer Calibrated KVTC with Auto-Rank Selection.

Extends KVTCManager (squish/kv/kvtc.py) with:
  1.  Automatic principal-component rank selection per layer based on an
      *explained variance ratio* target (e.g. 0.95 = retain 95% variance).
  2.  A target compression ratio that drives rank selection downward when
      memory pressure is high.
  3.  A `compression_summary()` report showing per-layer rank, compression
      ratio, and explained variance.

Typical usage::

    manager = AdaptiveKVTCManager(AdaptiveKVTCConfig(target_compression=15.0))
    ranks = manager.auto_calibrate(all_kv_samples)  # auto-selects ranks
    enc_k, enc_v = manager.encode_layer(0, k, v)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from squish.kv.kvtc import KVTCConfig, KVTCEncoded, KVTCLayer, KVTCManager, KVTCStats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveKVTCConfig:
    """Configuration for adaptive per-layer KVTC."""
    target_compression: float = 10.0   # desired compression ratio
    explained_variance_target: float = 0.95  # min explained variance to keep
    max_rank: int = 128
    min_rank: int = 4
    quant_bits: int = 8
    calibration_batches: int = 32

    def __post_init__(self) -> None:
        if self.target_compression < 1.0:
            raise ValueError(
                f"target_compression must be >= 1.0, got {self.target_compression}"
            )
        if not (0.0 < self.explained_variance_target <= 1.0):
            raise ValueError(
                f"explained_variance_target must be in (0, 1], "
                f"got {self.explained_variance_target}"
            )
        if self.min_rank < 1:
            raise ValueError(f"min_rank must be >= 1, got {self.min_rank}")
        if self.max_rank < self.min_rank:
            raise ValueError(
                f"max_rank ({self.max_rank}) must be >= min_rank ({self.min_rank})"
            )

    def to_kvtc_config(self, rank: int) -> KVTCConfig:
        return KVTCConfig(rank=rank, quant_bits=self.quant_bits)


# ---------------------------------------------------------------------------
# Per-layer adaptive coder
# ---------------------------------------------------------------------------

class AdaptiveKVTCLayer(KVTCLayer):
    """KVTC layer that can auto-select its PCA rank."""

    def __init__(self, cfg: AdaptiveKVTCConfig) -> None:
        # We start with max_rank; actual rank is refined in calibrate_and_tune
        super().__init__(KVTCConfig(rank=cfg.max_rank, quant_bits=cfg.quant_bits))
        self._adaptive_cfg = cfg
        self._selected_rank: Optional[int] = None
        self._singular_values: Optional[np.ndarray] = None

    def calibrate_and_tune(
        self,
        kv_samples: np.ndarray,
        target_rank: Optional[int] = None,
    ) -> int:
        """Calibrate and select the minimum rank that satisfies the EV target.

        Args:
            kv_samples:  (n_samples, d_kv) calibration vectors.
            target_rank: if given, override the auto-selection logic.

        Returns:
            The selected rank (also stored as self._selected_rank).
        """
        if kv_samples.ndim != 2:
            raise ValueError(
                f"kv_samples must be 2-D (n_samples, d_kv), got {kv_samples.shape}"
            )
        n, d = kv_samples.shape
        max_rank = min(self._adaptive_cfg.max_rank, min(n, d))

        centered = kv_samples.astype(np.float32) - kv_samples.mean(axis=0)
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        self._singular_values = s

        if target_rank is not None:
            rank = int(np.clip(target_rank, self._adaptive_cfg.min_rank, max_rank))
        else:
            # Select rank for explained_variance_target
            ev_cumsum = np.cumsum(s**2) / (s**2).sum()
            above = np.where(ev_cumsum >= self._adaptive_cfg.explained_variance_target)[0]
            rank_ev = int(above[0]) + 1 if len(above) > 0 else len(s)

            # Also respect compression target: rank ≤ d / target_compression
            approx_max = max(
                self._adaptive_cfg.min_rank,
                int(d / self._adaptive_cfg.target_compression),
            )
            rank = int(np.clip(min(rank_ev, approx_max), self._adaptive_cfg.min_rank, max_rank))

        self._selected_rank = rank
        # Re-build with correct rank
        self.config = KVTCConfig(rank=rank, quant_bits=self._adaptive_cfg.quant_bits)
        self._components = Vt[:rank]
        self._mean = kv_samples.mean(axis=0)
        self._calibrated = True
        return rank

    def explained_variance_ratio(self, rank: int) -> float:
        """Fraction of variance explained by the first `rank` components."""
        if self._singular_values is None:
            raise RuntimeError("Must call calibrate_and_tune() first")
        sv = self._singular_values
        total = (sv**2).sum()
        if total == 0:
            return 0.0
        return float((sv[:rank]**2).sum() / total)

    @property
    def selected_rank(self) -> Optional[int]:
        return self._selected_rank


# ---------------------------------------------------------------------------
# Multi-layer adaptive manager
# ---------------------------------------------------------------------------

@dataclass
class LayerRankInfo:
    """Per-layer rank selection result."""
    layer_idx: int
    rank: int
    d_kv: int
    explained_variance: float
    estimated_compression: float


class AdaptiveKVTCManager(KVTCManager):
    """KVTC manager that auto-calibrates rank per layer.

    Extends KVTCManager; all encode_layer / decode_layer calls work unchanged.
    """

    def __init__(self, config: AdaptiveKVTCConfig, n_layers: int) -> None:
        # Bootstrap with a dummy KVTCConfig; we override coders below
        dummy_kvtc = KVTCConfig(rank=config.max_rank, quant_bits=config.quant_bits)
        super().__init__(dummy_kvtc, n_layers)
        self._adaptive_cfg = config
        # Replace standard KVTCLayer instances with adaptive ones
        self._k_coders = {i: AdaptiveKVTCLayer(config) for i in range(n_layers)}
        self._v_coders = {i: AdaptiveKVTCLayer(config) for i in range(n_layers)}
        self._rank_info: Dict[int, Tuple[LayerRankInfo, LayerRankInfo]] = {}

    def auto_calibrate(
        self,
        all_kv_samples: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, int]:
        """Calibrate all layers and auto-select per-layer ranks.

        Args:
            all_kv_samples: dict layer_idx → (k_samples, v_samples), each
                            shaped (n_samples, d_kv).

        Returns:
            Dict layer_idx → selected_rank (averaged over K and V).
        """
        result: Dict[int, int] = {}
        for layer_idx, (k_samp, v_samp) in all_kv_samples.items():
            k_coder = self._k_coders[layer_idx]
            v_coder = self._v_coders[layer_idx]

            k_rank = k_coder.calibrate_and_tune(k_samp)
            v_rank = v_coder.calibrate_and_tune(v_samp)
            avg_rank = (k_rank + v_rank) // 2

            d_kv = k_samp.shape[1]
            ev_k = k_coder.explained_variance_ratio(k_rank)
            ev_v = v_coder.explained_variance_ratio(v_rank)
            ev = (ev_k + ev_v) / 2.0
            compression = d_kv / max(avg_rank, 1)

            self._rank_info[layer_idx] = (
                LayerRankInfo(layer_idx, k_rank, d_kv, ev_k, d_kv / max(k_rank, 1)),
                LayerRankInfo(layer_idx, v_rank, d_kv, ev_v, d_kv / max(v_rank, 1)),
            )
            self.stats.calibration_calls += 1
            result[layer_idx] = avg_rank

        return result

    def compression_summary(self) -> Dict[str, float]:
        """Return aggregate compression statistics across all calibrated layers.

        Returns:
            Dict with 'mean_rank', 'mean_compression', 'mean_explained_variance'.
        """
        if not self._rank_info:
            return {}
        ranks = []
        compressions = []
        evs = []
        for k_info, v_info in self._rank_info.values():
            ranks.append((k_info.rank + v_info.rank) / 2)
            compressions.append((k_info.estimated_compression + v_info.estimated_compression) / 2)
            evs.append((k_info.explained_variance + v_info.explained_variance) / 2)
        return {
            "mean_rank": float(np.mean(ranks)),
            "mean_compression": float(np.mean(compressions)),
            "mean_explained_variance": float(np.mean(evs)),
            "n_layers": float(len(self._rank_info)),
        }

    def rank_info(self, layer_idx: int) -> Optional[Tuple[LayerRankInfo, LayerRankInfo]]:
        """Return (k_info, v_info) for a layer, or None if not yet calibrated."""
        return self._rank_info.get(layer_idx)

    def __repr__(self) -> str:
        summary = self.compression_summary()
        if summary:
            return (
                f"AdaptiveKVTCManager(layers={self.n_layers}, "
                f"mean_rank={summary['mean_rank']:.1f}, "
                f"mean_compression={summary['mean_compression']:.1f}x, "
                f"mean_ev={summary['mean_explained_variance']:.1%})"
            )
        return f"AdaptiveKVTCManager(layers={self.n_layers}, uncalibrated)"
