"""squish/quant/delta_zip.py

DeltaZipAdapter — Delta compression for fine-tuned model weights supporting
multi-tenant LoRA adapter serving (Yao et al., MLSys 2025 / arXiv:2312.05215).

Reference
---------
"DeltaZip: Multi-Tenant Delta Compression for Large Language Models."
Yao et al., MLSys 2025 / arXiv:2312.05215.

Algorithm
---------
DeltaZip stores adapters as a quantised *delta* over a shared base model:

1.  Compute ``delta = adapted_weight - base_weight``.
2.  Quantise delta with symmetric uniform quantisation (quant_bits bits).
3.  Store codes as int8 (or int4 packed) + per-block float32 scale.
4.  At inference, reconstruct delta, add base weight → merged weight.

The XOR framing: because delta values cluster around zero (small updates),
entropy-coding them achieves much higher compression than coding the full
weights directly.

This simulation:
* Supports ``quant_bits`` ∈ {2, 4, 8}.
* Uses block-wise symmetric quantisation; block_size quantised separately.
* Provides ``merge(adapter_id, base_weight)`` for zero-copy inference.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* Scales (float16) + codes (int8) stored per block.
* Compression ratio = bytes(stored) / bytes(original fp32).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "DeltaZipConfig",
    "DeltaCompressedAdapter",
    "DeltaZipAdapter",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class DeltaZipConfig:
    """Configuration for :class:`DeltaZipAdapter`.

    Attributes:
        quant_bits: Quantisation precision for the delta (2, 4, or 8).
        block_size: Number of elements per quantisation block.
        symmetric: If True, symmetric quantisation (zero_point = 0).
        seed: RNG seed (unused in production path; for test reproducibility).
    """

    quant_bits: int = 8
    block_size: int = 64
    symmetric: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.quant_bits not in (2, 4, 8):
            raise ValueError(
                f"quant_bits must be 2, 4, or 8; got {self.quant_bits}"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be ≥ 1; got {self.block_size}")


# ── Compressed adapter ────────────────────────────────────────────────────────


class DeltaCompressedAdapter:
    """Quantised delta representation of a single adapter.

    Attributes:
        codes: int8 array of quantised delta values; shape ``(n_elements,)``.
        scales: float16 array of per-block scales; shape ``(n_blocks,)``.
        original_shape: Original ``(rows, cols)`` of adapted weight.
        pad: Number of zero-pad elements appended before quantisation.
        quant_bits: Bits used for quantisation.
    """

    def __init__(
        self,
        codes: np.ndarray,
        scales: np.ndarray,
        original_shape: Tuple[int, ...],
        pad: int,
        quant_bits: int,
    ) -> None:
        self.codes = codes
        self.scales = scales
        self.original_shape = original_shape
        self.pad = pad
        self.quant_bits = quant_bits

    def nbytes(self) -> int:
        """Bytes used by codes + scales."""
        return self.codes.nbytes + self.scales.nbytes


# ── Adapter store ─────────────────────────────────────────────────────────────


class DeltaZipAdapter:
    """Store and serve LoRA adapters as compressed delta tensors.

    Example::

        cfg = DeltaZipConfig(quant_bits=8, block_size=64)
        store = DeltaZipAdapter(cfg)

        base = np.random.randn(256, 256).astype(np.float32)
        fine = base + 0.01 * np.random.randn(256, 256).astype(np.float32)
        store.compress_delta("lora_a", base, fine)
        merged = store.merge("lora_a", base)  # close to fine
    """

    def __init__(self, config: Optional[DeltaZipConfig] = None) -> None:
        self.config = config or DeltaZipConfig()
        self._adapters: Dict[str, DeltaCompressedAdapter] = {}

    # ── Compression ───────────────────────────────────────────────────────────

    def compress_delta(
        self,
        adapter_id: str,
        base_weight: np.ndarray,
        adapted_weight: np.ndarray,
    ) -> DeltaCompressedAdapter:
        """Compute and compress ``adapted - base`` for ``adapter_id``.

        Args:
            adapter_id: Unique string identifier for this adapter.
            base_weight: Base model weight, any shape castable to float32.
            adapted_weight: Fine-tuned weight; must match base_weight's shape.

        Returns:
            The :class:`DeltaCompressedAdapter` (also stored internally).

        Raises:
            ValueError: If shapes do not match.
        """
        base_weight = np.asarray(base_weight, dtype=np.float32)
        adapted_weight = np.asarray(adapted_weight, dtype=np.float32)
        if base_weight.shape != adapted_weight.shape:
            raise ValueError(
                f"Shape mismatch: base {base_weight.shape} vs adapted {adapted_weight.shape}"
            )
        orig_shape = base_weight.shape
        delta = (adapted_weight - base_weight).ravel()
        codes, scales, pad = self._quantize(delta)
        compressed = DeltaCompressedAdapter(
            codes=codes,
            scales=scales,
            original_shape=orig_shape,
            pad=pad,
            quant_bits=self.config.quant_bits,
        )
        self._adapters[adapter_id] = compressed
        return compressed

    # ── Decompression ─────────────────────────────────────────────────────────

    def decompress_delta(self, adapter_id: str) -> np.ndarray:
        """Reconstruct float32 delta tensor for ``adapter_id``.

        Returns:
            Delta array with shape matching the original weight.

        Raises:
            KeyError: If adapter_id not found.
        """
        compressed = self._get(adapter_id)
        delta_flat = self._dequantize(compressed.codes, compressed.scales)
        if compressed.pad > 0:
            delta_flat = delta_flat[: -compressed.pad]
        return delta_flat.reshape(compressed.original_shape)

    # ── Merge ─────────────────────────────────────────────────────────────────

    def merge(self, adapter_id: str, base_weight: np.ndarray) -> np.ndarray:
        """Return the merged (adapted) weight for inference.

        Args:
            adapter_id: Stored adapter identifier.
            base_weight: Base model weight (must match original shape).

        Returns:
            ``base_weight + decompress_delta(adapter_id)`` in float32.

        Raises:
            KeyError: If adapter_id not found.
            ValueError: If base_weight shape mismatches stored original shape.
        """
        compressed = self._get(adapter_id)
        base_weight = np.asarray(base_weight, dtype=np.float32)
        if base_weight.shape != compressed.original_shape:
            raise ValueError(
                f"Base weight shape {base_weight.shape} does not match "
                f"stored shape {compressed.original_shape}."
            )
        delta = self.decompress_delta(adapter_id)
        return (base_weight + delta).astype(np.float32)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def n_adapters(self) -> int:
        """Number of stored adapters."""
        return len(self._adapters)

    def memory_bytes(self) -> int:
        """Total bytes used by all stored compressed adapters."""
        return sum(c.nbytes() for c in self._adapters.values())

    def compression_ratio(self, adapter_id: str) -> float:
        """Bytes(compressed) / bytes(original FP32 weight).

        Values < 1 indicate compression.

        Raises:
            KeyError: If adapter_id not found.
        """
        compressed = self._get(adapter_id)
        n_elements = int(np.prod(compressed.original_shape))
        original_bytes = n_elements * 4  # float32 = 4 bytes
        return compressed.nbytes() / original_bytes

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get(self, adapter_id: str) -> DeltaCompressedAdapter:
        if adapter_id not in self._adapters:
            raise KeyError(f"Adapter '{adapter_id}' not in DeltaZipAdapter store.")
        return self._adapters[adapter_id]

    def _quantize(
        self, delta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Block-wise symmetric quantisation of a 1-D float32 delta.

        Returns:
            codes: int8 array of quantised values.
            scales: float16 array of per-block scales.
            pad: number of padding elements appended.
        """
        bs = self.config.block_size
        levels = 2 ** self.config.quant_bits
        half_levels = levels // 2 - 1  # e.g. 127 for 8-bit symmetric

        n = len(delta)
        pad = (bs - n % bs) % bs
        if pad > 0:
            delta = np.concatenate([delta, np.zeros(pad, dtype=np.float32)])
        n_padded = len(delta)
        n_blocks = n_padded // bs

        blocks = delta.reshape(n_blocks, bs)
        abs_max = np.abs(blocks).max(axis=-1)  # (n_blocks,)
        scales = (abs_max / half_levels).astype(np.float16)   # float16 scales
        # Avoid division by zero for zero-blocks
        safe_scales = scales.astype(np.float32)
        safe_scales = np.where(safe_scales == 0.0, 1.0, safe_scales)

        codes = np.clip(
            np.round(blocks / safe_scales[:, None]), -half_levels, half_levels
        ).astype(np.int8)

        return codes.ravel(), scales, pad

    def _dequantize(self, codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
        bs = self.config.block_size
        scales_f32 = scales.astype(np.float32)
        n_blocks = len(scales_f32)
        blocks = codes.reshape(n_blocks, bs).astype(np.float32)
        return (blocks * scales_f32[:, None]).ravel()

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"DeltaZipAdapter(quant_bits={cfg.quant_bits}, "
            f"block_size={cfg.block_size}, n_adapters={self.n_adapters()})"
        )
