# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""KVCompress — Online KV cache quantisation + pruning during generation.

As generation proceeds, older KV entries are less attended.  This module
compresses old KV entries using INT8 symmetric quantisation and
magnitude-based pruning, reducing memory pressure at long context.

Inspired by:
    Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
    of Large Language Models", NeurIPS 2023. https://arxiv.org/abs/2306.14048

Usage::

    from squish.kv_compress import KVCompressor, KVCompressConfig
    import numpy as np

    cfg  = KVCompressConfig(compress_after=256, quant_bits=8, prune_ratio=0.1,
                            n_heads=8, head_dim=64)
    comp = KVCompressor(cfg)

    keys   = np.random.randn(8, 300, 64).astype(np.float32)
    values = np.random.randn(8, 300, 64).astype(np.float32)
    entry  = comp.compress(keys, values)
    k_back, v_back = comp.decompress(entry)
    print(f"prune_rate={comp.stats.prune_rate:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "KVCompressConfig",
    "CompressedKVEntry",
    "KVCompressor",
    "CompressStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KVCompressConfig:
    """Configuration for online KV cache compression.

    Attributes:
        compress_after: Number of tokens already in the cache before
                        compression is triggered.  This value is informational
                        for callers; the :class:`KVCompressor` itself
                        compresses whatever slice it receives.
        quant_bits:     Bit-width for INT quantisation.  Currently only 8 is
                        fully supported; the implementation scales to
                        ``quant_bits`` bits symmetrically.
        prune_ratio:    Fraction of token positions to drop per head based on
                        L2 magnitude of the key vector.  Must be in
                        ``[0.0, 1.0)``.
        n_heads:        Number of KV heads in the model.
        head_dim:       Head dimension (key / value vector length per head).
    """

    compress_after: int = 256
    quant_bits: int = 8
    prune_ratio: float = 0.1
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.compress_after < 0:
            raise ValueError(
                f"compress_after must be >= 0; got {self.compress_after}"
            )
        if self.quant_bits not in (4, 8):
            raise ValueError(
                f"quant_bits must be 4 or 8; got {self.quant_bits}"
            )
        if not (0.0 <= self.prune_ratio < 1.0):
            raise ValueError(
                f"prune_ratio must be in [0, 1); got {self.prune_ratio}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")

    @property
    def quant_max(self) -> int:
        """Maximum representable integer value for symmetric quantisation."""
        return (1 << (self.quant_bits - 1)) - 1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CompressedKVEntry:
    """A compressed representation of a KV cache slice.

    Attributes:
        keys_q:   Quantised key tensor, shape ``(n_heads, n_kept, head_dim)``,
                  dtype uint8 (packed as signed integers offset by quant_max).
        values_q: Quantised value tensor, same shape and dtype as ``keys_q``.
        scale_k:  Scalar dequantisation scale for keys.
        scale_v:  Scalar dequantisation scale for values.
        mask:     Boolean keep-mask, shape ``(n_heads, seq_len)``, indicating
                  which original token positions are retained after pruning.
                  ``True`` = retained, ``False`` = pruned.
    """

    keys_q: np.ndarray
    values_q: np.ndarray
    scale_k: float
    scale_v: float
    mask: np.ndarray


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CompressStats:
    """Cumulative compression statistics.

    Attributes:
        n_compress_calls: Total number of :meth:`~KVCompressor.compress` calls.
        total_pruned:     Total token positions dropped across all calls.
        total_tokens:     Total token positions presented for compression.
    """

    n_compress_calls: int = 0
    total_pruned: int = 0
    total_tokens: int = 0

    @property
    def prune_rate(self) -> float:
        """Fraction of token positions pruned across all compression calls.

        Returns 0.0 when no tokens have been processed.
        """
        if self.total_tokens == 0:
            return 0.0
        return self.total_pruned / self.total_tokens


# ---------------------------------------------------------------------------
# KVCompressor
# ---------------------------------------------------------------------------


class KVCompressor:
    """Online KV cache quantiser and magnitude-based pruner.

    Compresses a KV slice via two passes:

    1. **Pruning** — For each head, token positions whose key L2 norm falls
       below the ``prune_ratio`` quantile are discarded.
    2. **Quantisation** — The surviving positions are INT8-quantised with a
       per-tensor symmetric scale derived from the maximum absolute value.

    Args:
        config: :class:`KVCompressConfig` instance.
    """

    def __init__(self, config: KVCompressConfig) -> None:
        self._config = config
        self._stats = CompressStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> CompressedKVEntry:
        """Quantise and prune a KV slice.

        Args:
            keys:   Key tensor, shape ``(n_heads, seq_len, head_dim)``,
                    float32.
            values: Value tensor, same shape as ``keys``, float32.

        Returns:
            :class:`CompressedKVEntry` containing the quantised and pruned
            key/value data along with the keep-mask and dequant scales.

        Raises:
            ValueError: if ``keys`` and ``values`` shapes are incompatible
                        with the configured ``n_heads`` and ``head_dim``.
        """
        self._validate_input(keys, values)

        cfg = self._config
        n_heads, seq_len, head_dim = keys.shape

        # Step 1 — Magnitude-based pruning per head.
        mask = self._build_prune_mask(keys)  # (n_heads, seq_len)
        n_kept = int(mask[0].sum())  # same count for all heads by construction

        keys_kept = np.empty((n_heads, n_kept, head_dim), dtype=np.float32)
        vals_kept = np.empty((n_heads, n_kept, head_dim), dtype=np.float32)
        for h in range(n_heads):
            keys_kept[h] = keys[h, mask[h]]
            vals_kept[h] = values[h, mask[h]]

        # Step 2 — Symmetric INT quantisation.
        keys_q, scale_k = self._quantize(keys_kept)
        vals_q, scale_v = self._quantize(vals_kept)

        n_pruned = seq_len * n_heads - n_kept * n_heads
        self._stats.n_compress_calls += 1
        self._stats.total_tokens += seq_len * n_heads
        self._stats.total_pruned += n_pruned

        return CompressedKVEntry(
            keys_q=keys_q,
            values_q=vals_q,
            scale_k=scale_k,
            scale_v=scale_v,
            mask=mask,
        )

    def decompress(
        self,
        entry: CompressedKVEntry,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dequantise a :class:`CompressedKVEntry` back to float32.

        Note:
            Pruned positions are not restored; the returned tensors have the
            same shape as the *kept* (post-pruning) slices, not the original
            ``seq_len``.

        Args:
            entry: A :class:`CompressedKVEntry` produced by
                   :meth:`compress`.

        Returns:
            ``(keys_fp32, values_fp32)``, each of shape
            ``(n_heads, n_kept, head_dim)``, float32.
        """
        cfg = self._config
        q_max = cfg.quant_max

        keys_fp32 = (entry.keys_q.astype(np.float32) - q_max) * entry.scale_k
        vals_fp32 = (entry.values_q.astype(np.float32) - q_max) * entry.scale_v
        return keys_fp32, vals_fp32

    @property
    def stats(self) -> CompressStats:
        """Cumulative compression statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset cumulative statistics to zero."""
        self._stats = CompressStats()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prune_mask(self, keys: np.ndarray) -> np.ndarray:
        """Build a per-head boolean keep-mask based on key L2 norms.

        Positions whose per-head L2 norm falls below the ``prune_ratio``
        quantile are marked False (pruned).  All heads retain the same
        number of positions to enable packed storage.

        Args:
            keys: Shape ``(n_heads, seq_len, head_dim)``, float32.

        Returns:
            Boolean mask of shape ``(n_heads, seq_len)``.
        """
        cfg = self._config
        n_heads, seq_len, _ = keys.shape

        norms = np.linalg.norm(keys, axis=-1)  # (n_heads, seq_len)

        # Compute the threshold as the prune_ratio quantile of norms,
        # averaged across heads so all heads keep the same number of positions.
        if cfg.prune_ratio == 0.0:
            return np.ones((n_heads, seq_len), dtype=bool)

        threshold = float(np.quantile(norms, cfg.prune_ratio))
        mask = norms >= threshold  # (n_heads, seq_len)

        # Guarantee at least one position is kept per head.
        for h in range(n_heads):
            if not mask[h].any():  # pragma: no cover
                best = int(np.argmax(norms[h]))
                mask[h, best] = True

        return mask

    def _quantize(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Quantise a float32 tensor to uint8 using symmetric INT scaling.

        The scale is derived from the maximum absolute value of ``x``.
        Quantised values are stored offset by ``quant_max`` so that the
        range ``[-quant_max, +quant_max]`` maps to ``[0, 2*quant_max]``.

        Args:
            x: Float32 array of any shape.

        Returns:
            ``(quantised, scale)`` where ``quantised`` is uint8 and
            ``scale`` is the dequantisation scalar.
        """
        q_max = self._config.quant_max
        abs_max = float(np.max(np.abs(x)))
        if abs_max < 1e-30:
            scale = 1.0
        else:
            scale = abs_max / q_max

        x_scaled = x / scale
        x_clipped = np.clip(np.round(x_scaled), -q_max, q_max)
        # Offset by q_max to make unsigned (uint8 friendly).
        x_uint8 = (x_clipped + q_max).astype(np.uint8)
        return x_uint8, scale

    def _validate_input(
        self,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> None:
        cfg = self._config
        if keys.ndim != 3:
            raise ValueError(
                f"keys must be 3-D (n_heads, seq_len, head_dim); "
                f"got shape {keys.shape}"
            )
        if values.shape != keys.shape:
            raise ValueError(
                f"values shape {values.shape} must match keys shape "
                f"{keys.shape}"
            )
        if keys.shape[0] != cfg.n_heads:
            raise ValueError(
                f"keys.shape[0]={keys.shape[0]} does not match "
                f"config n_heads={cfg.n_heads}"
            )
        if keys.shape[2] != cfg.head_dim:
            raise ValueError(
                f"keys.shape[2]={keys.shape[2]} does not match "
                f"config head_dim={cfg.head_dim}"
            )
