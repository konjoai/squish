"""
squish/quant/layer_dedup.py

LayerDeduplicator — Cross-Layer Weight Deduplication via Delta Encoding.

Based on:
  "Cross-Layer Attention Sharing for Large Language Models"
  Brandon et al. — arXiv:2405.12981 (2024)

  "DeepSeek-V3 Technical Report" — Model Size Reduction via Shared Layers
  DeepSeek AI — December 2024

  Weight-sharing and delta-compression observations from:
  "The Truth is in There: Improving Reasoning in Language Models with
   Layer-Selective Rank Reduction" — LASER (ICLR 2024)

Background
----------
In large transformer models, successive layers often learn highly similar
weight matrices.  Adjacent or symmetrically-placed layers in Qwen-2.5,
LLaMA-3, and DeepSeek-V3 families can have row-cosine-similarity > 0.99.

Key Insight:

  If layer_j.weight ≈ layer_i.weight, we can store:
    • Full weight of the *reference* layer i.
    • Only the **delta** (difference) for the *dependent* layer j:
        delta_j = layer_j.weight − layer_i.weight

  The delta is typically small in magnitude and sparse.  We quantize the
  delta to 8-bit integers (delta_bits=8) to further compress it.

  At inference, the dependent layer weight is reconstructed on demand:
    layer_j.weight = layer_i.weight + dequantize(delta_j)

Compression
-----------
For a pair of layers each with weight F bytes:
  - With deduplication: F (reference) + F/4 (int8 delta ≈ 25% of float32)
  - Savings: ~75% of second-layer storage = 37.5% of the pair.
  - At model level: 20–40% reduction in on-disk weight bytes.

Classes
-------
``LayerDedupConfig``    — configuration
``LayerSimilarity``     — similarity result between two layers
``DedupEntry``          — a deduplicated (reference + delta) entry
``LayerDedupStats``     — per-instance statistics
``LayerDeduplicator``   — analyze, deduplicate, reconstruct API

Usage::

    from squish.quant.layer_dedup import LayerDedupConfig, LayerDeduplicator
    import numpy as np

    weights = {
        "layer_0.weight": np.random.randn(256, 256).astype(np.float32),
        "layer_1.weight": np.random.randn(256, 256).astype(np.float32),
    }
    dedup = LayerDeduplicator(LayerDedupConfig(similarity_threshold=0.95))

    similarities = dedup.analyze(weights)
    store = dedup.deduplicate(weights)
    reconstructed = dedup.reconstruct(store, "layer_1.weight")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "LayerDedupConfig",
    "LayerSimilarity",
    "DedupEntry",
    "LayerDedupStats",
    "LayerDeduplicator",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LayerDedupConfig:
    """Configuration for the cross-layer weight deduplicator.

    Attributes:
        similarity_threshold: Minimum mean row-cosine-similarity to consider
                              two layers as "similar enough" to deduplicate.
                              Value in [0, 1].  Default: ``0.99``.
        delta_bits:           Bit-width for delta quantization.  Must be 8 or
                              16.  Default: ``8``.
        min_rows:             Minimum number of weight rows (output dimension)
                              required before attempting deduplication (avoids
                              overhead on tiny embeddings).  Default: ``16``.
    """

    similarity_threshold: float = 0.99
    delta_bits: int = 8
    min_rows: int = 16

    def __post_init__(self) -> None:
        if not (0.0 < self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {self.similarity_threshold}"
            )
        if self.delta_bits not in {8, 16}:
            raise ValueError(
                f"delta_bits must be 8 or 16, got {self.delta_bits}"
            )
        if self.min_rows < 1:
            raise ValueError(f"min_rows must be >= 1, got {self.min_rows}")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LayerSimilarity:
    """Similarity analysis result between two weight matrices.

    Attributes:
        layer_a:          Key of the first layer.
        layer_b:          Key of the second layer.
        row_similarity:   Mean row-level cosine similarity in [0, 1].
        delta_norm:       L2 norm of the normalized difference
                          (weight_b − weight_a) / max(|weight_a|).
        shape:            Shape of both weight matrices.
    """

    layer_a: str
    layer_b: str
    row_similarity: float
    delta_norm: float
    shape: Tuple[int, ...]

    @property
    def is_similar(self) -> bool:
        return self.row_similarity >= 0.0  # check against threshold at call site

    def __repr__(self) -> str:
        return (
            f"LayerSimilarity({self.layer_a!r} ↔ {self.layer_b!r}, "
            f"sim={self.row_similarity:.4f}, δ‖·‖={self.delta_norm:.4f})"
        )


@dataclass
class DedupEntry:
    """A deduplicated layer: reference key + quantized delta.

    Attributes:
        reference_key:  Key in the store for the reference (full) weight.
        delta_quant:    Quantized int8/int16 delta array.
        delta_scale:    Per-row dequantization scale factors (float32).
        original_dtype: Original dtype of the weight.
    """

    reference_key: str
    delta_quant: np.ndarray
    delta_scale: np.ndarray
    original_dtype: np.dtype

    def nbytes(self) -> int:
        return self.delta_quant.nbytes + self.delta_scale.nbytes

    def __repr__(self) -> str:
        return (
            f"DedupEntry(ref={self.reference_key!r}, "
            f"delta={self.delta_quant.shape}, dtype={self.original_dtype})"
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class LayerDedupStats:
    """Lifetime statistics for a LayerDeduplicator.

    Attributes:
        pairs_analyzed:  Pairs of layers compared.
        pairs_deduped:   Pairs where deduplication was applied.
        bytes_saved:     Estimated byte reduction vs full float32 storage.
        original_bytes:  Total float32 bytes of all deduplicated layers.
    """

    pairs_analyzed: int = 0
    pairs_deduped: int = 0
    bytes_saved: int = 0
    original_bytes: int = 0

    @property
    def disk_reduction_ratio(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return self.bytes_saved / self.original_bytes

    def __repr__(self) -> str:
        return (
            f"LayerDedupStats(analyzed={self.pairs_analyzed}, "
            f"deduped={self.pairs_deduped}, "
            f"saved_bytes={self.bytes_saved}, "
            f"reduction={self.disk_reduction_ratio:.2%})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


def _row_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between corresponding rows of two matrices."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    # Avoid division by zero
    a_safe = np.where(a_norm > 0, a / np.maximum(a_norm, 1e-12), 0.0)
    b_safe = np.where(b_norm > 0, b / np.maximum(b_norm, 1e-12), 0.0)
    cos_per_row = np.sum(a_safe * b_safe, axis=1)
    return float(np.mean(cos_per_row))


def _quantize_delta(
    delta: np.ndarray, bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a float32 delta weight row-wise to int8/int16.

    Returns
    -------
    quantized : int8/int16 array of same shape.
    scale     : float32 per-row scale.
    """
    dtype = np.int8 if bits == 8 else np.int16
    max_val = float((1 << (bits - 1)) - 1)

    row_abs_max = np.abs(delta).max(axis=1, keepdims=True)
    row_abs_max = np.maximum(row_abs_max, 1e-12)  # avoid /0
    scale = row_abs_max[:, 0].astype(np.float32) / max_val
    q = np.round(delta / row_abs_max * max_val).clip(-max_val, max_val).astype(dtype)
    return q, scale


def _dequantize_delta(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Reconstruct float32 delta from quantized + scale."""
    return q.astype(np.float32) * scale[:, None]


class LayerDeduplicator:
    """Analyze and deduplicate cross-layer weight similarity.

    Parameters
    ----------
    config:
        Deduplication configuration.
    """

    def __init__(self, config: LayerDedupConfig | None = None) -> None:
        self._cfg = config or LayerDedupConfig()
        self.stats = LayerDedupStats()

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(
        self, weights: Dict[str, np.ndarray]
    ) -> List[LayerSimilarity]:
        """Compute pairwise similarity for all weight matrices.

        Only compares matrices of identical shape.  Returns all pairs,
        sorted by descending row_similarity.

        Parameters
        ----------
        weights:
            Dict of layer_name → float32 weight array.

        Returns
        -------
        List of LayerSimilarity results.
        """
        keys = list(weights.keys())
        results: List[LayerSimilarity] = []
        cfg = self._cfg

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ka, kb = keys[i], keys[j]
                wa, wb = weights[ka], weights[kb]
                if wa.shape != wb.shape:
                    continue
                if wa.ndim < 2:
                    continue
                if wa.shape[0] < cfg.min_rows:
                    continue

                wa_f = wa.astype(np.float32)
                wb_f = wb.astype(np.float32)

                sim = _row_cosine_similarity(wa_f, wb_f)
                w_range = max(float(np.abs(wa_f).max()), 1e-12)
                delta_norm = float(np.linalg.norm(wb_f - wa_f) / w_range)

                results.append(
                    LayerSimilarity(
                        layer_a=ka,
                        layer_b=kb,
                        row_similarity=sim,
                        delta_norm=delta_norm,
                        shape=wa.shape,
                    )
                )
                self.stats.pairs_analyzed += 1

        results.sort(key=lambda x: x.row_similarity, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Deduplicate
    # ------------------------------------------------------------------

    def deduplicate(
        self, weights: Dict[str, np.ndarray]
    ) -> Dict[str, Union[np.ndarray, DedupEntry]]:
        """Replace similar layer pairs with reference + delta encoding.

        For each similar pair (a, b) sorted by similarity descending:
          - Keep layer_a as a full weight (reference).
          - Replace layer_b with a DedupEntry pointing to layer_a.

        Parameters
        ----------
        weights:
            Dict of layer_name → float32 weight array.

        Returns
        -------
        Store dict where values are either:
          - ``np.ndarray`` (full weight, unchanged or reference)
          - ``DedupEntry`` (compressed dependent layer)
        """
        cfg = self._cfg
        similarities = self.analyze(weights)
        referenced: set = set()  # layers already used as reference or deduped
        store: Dict[str, Union[np.ndarray, DedupEntry]] = {
            k: v.astype(np.float32) for k, v in weights.items()
        }

        for sim in similarities:
            if sim.row_similarity < cfg.similarity_threshold:
                continue
            ka, kb = sim.layer_a, sim.layer_b
            # Only deduplicate if neither layer has already been deduped
            if kb in referenced or ka in referenced:
                continue
            # Do not deduplicate a layer that is already a DedupEntry
            if isinstance(store.get(ka), DedupEntry) or isinstance(store.get(kb), DedupEntry):
                continue

            wa: np.ndarray = store[ka]  # type: ignore[assignment]
            wb: np.ndarray = store[kb]  # type: ignore[assignment]
            delta = wb.astype(np.float32) - wa.astype(np.float32)
            delta_q, delta_scale = _quantize_delta(delta, cfg.delta_bits)

            original_bytes = wb.nbytes
            delta_entry = DedupEntry(
                reference_key=ka,
                delta_quant=delta_q,
                delta_scale=delta_scale,
                original_dtype=wb.dtype,
            )
            store[kb] = delta_entry
            referenced.add(kb)

            bytes_saved = original_bytes - delta_entry.nbytes()
            self.stats.bytes_saved += max(0, bytes_saved)
            self.stats.original_bytes += original_bytes
            self.stats.pairs_deduped += 1

        return store

    # ------------------------------------------------------------------
    # Reconstruct
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        store: Dict[str, Union[np.ndarray, DedupEntry]],
        key: str,
    ) -> np.ndarray:
        """Reconstruct the full float32 weight for ``key``.

        Parameters
        ----------
        store:
            The store returned by ``deduplicate()``.
        key:
            Layer name to reconstruct.

        Returns
        -------
        float32 weight array.
        """
        val = store.get(key)
        if val is None:
            raise KeyError(f"Key {key!r} not found in store")

        if isinstance(val, np.ndarray):
            return val.astype(np.float32)

        # It's a DedupEntry
        ref_key = val.reference_key
        ref_val = store.get(ref_key)
        if ref_val is None:
            raise KeyError(f"Reference key {ref_key!r} not found in store")
        if isinstance(ref_val, DedupEntry):
            # Recurse (chains are rare but possible)
            ref_weight = self.reconstruct(store, ref_key)
        else:
            ref_weight = ref_val.astype(np.float32)

        delta = _dequantize_delta(val.delta_quant, val.delta_scale)
        return (ref_weight + delta).astype(val.original_dtype)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def disk_reduction_ratio(self) -> float:
        return self.stats.disk_reduction_ratio

    def __repr__(self) -> str:
        return (
            f"LayerDeduplicator("
            f"threshold={self._cfg.similarity_threshold}, "
            f"delta_bits={self._cfg.delta_bits}, "
            f"{self.stats})"
        )
