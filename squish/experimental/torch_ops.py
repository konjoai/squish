"""squish/torch_ops.py — INT4 / INT8 dequantization for the PyTorch/CUDA backend.

This module mirrors the Rust-backed dequantisation ops in ``squish_quant_rs``
but operates on PyTorch tensors so that they can stay on GPU (CUDA/ROCm) without
an unnecessary CPU round-trip.

Public API
----------
    dequantize_int4_asymmetric_torch(packed, scales, zero_points,
                                     group_size, device) → torch.Tensor
        Unpack nibble-packed INT4 weights and dequantise to float32 on *device*.

    dequantize_int4_torch(packed, scales,
                          group_size, device) → torch.Tensor
        Symmetric INT4 dequantisation (legacy format, kept for compatibility).

    loaded_weight_to_torch(arr, device) → torch.Tensor
        Convert a float32 numpy array (from ``loader_utils._dequantize_npy``) to
        a torch tensor on *device*.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def _import_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "squish.torch_ops requires PyTorch. "
            "Install with: pip install torch"
        ) from exc


def dequantize_int4_asymmetric_torch(
    packed: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    group_size: int,
    device: str = "cpu",
) -> "torch.Tensor":
    """Dequantise nibble-packed asymmetric INT4 weights to float32.

    Parameters
    ----------
    packed:
        ``uint8`` array of shape ``(n_rows, n_cols // 2)`` — two nibbles per byte,
        low nibble first (little-endian order).
    scales:
        ``float32`` array of shape ``(n_rows, n_groups)`` — per-group scale.
    zero_points:
        ``float32`` array of shape ``(n_rows, n_groups)`` — per-group zero-point.
    group_size:
        Number of columns covered by each scale / zero-point group.
    device:
        Target device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(n_rows, n_rows, n_cols)`` on *device*.
    """
    torch = _import_torch()

    n_rows = packed.shape[0]
    n_cols = packed.shape[1] * 2

    # Unpack nibbles on CPU (numpy is faster than torch for byte manipulation)
    packed_np = np.ascontiguousarray(packed, dtype=np.uint8)
    lo = (packed_np & 0x0F).astype(np.float32)  # (n_rows, n_cols//2)
    hi = ((packed_np >> 4) & 0x0F).astype(np.float32)
    # Interleave: column 0 = lo[0], column 1 = hi[0], column 2 = lo[1], ...
    interleaved = np.empty((n_rows, n_cols), dtype=np.float32)
    interleaved[:, 0::2] = lo
    interleaved[:, 1::2] = hi

    n_groups = n_cols // group_size
    scales_e    = np.repeat(scales,     group_size, axis=1)  # (n_rows, n_cols)
    zero_pts_e  = np.repeat(zero_points, group_size, axis=1)

    # dequant formula:  w = scale * (q - zero_point)
    result = scales_e * (interleaved - zero_pts_e)

    return torch.from_numpy(result).to(device=device, dtype=torch.float32)


def dequantize_int4_torch(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    device: str = "cpu",
) -> "torch.Tensor":
    """Dequantise nibble-packed symmetric INT4 weights to float32 (legacy).

    Parameters
    ----------
    packed:
        ``uint8`` array of shape ``(n_rows, n_cols // 2)``.
    scales:
        ``float32`` array of shape ``(n_rows, n_groups)`` — per-group scale.
    group_size:
        Number of columns per group.
    device:
        Target device string.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(n_rows, n_cols)`` on *device*.
    """
    torch = _import_torch()

    n_rows = packed.shape[0]
    n_cols = packed.shape[1] * 2

    packed_np = np.ascontiguousarray(packed, dtype=np.uint8)
    lo = (packed_np & 0x0F).astype(np.int8)
    hi = ((packed_np >> 4) & 0x0F).astype(np.int8)
    # Convert 4-bit unsigned (0–15) to signed (–8…7)
    lo = (lo.astype(np.int8) - 8).astype(np.float32)
    hi = (hi.astype(np.int8) - 8).astype(np.float32)

    interleaved = np.empty((n_rows, n_cols), dtype=np.float32)
    interleaved[:, 0::2] = lo
    interleaved[:, 1::2] = hi

    scales_e = np.repeat(scales, group_size, axis=1)
    result = scales_e * interleaved

    return torch.from_numpy(result).to(device=device, dtype=torch.float32)


def loaded_weight_to_torch(
    arr: np.ndarray,
    device: str = "cpu",
    dtype: str = "float16",
) -> "torch.Tensor":
    """Convert a float32 numpy array to a torch tensor on *device*.

    Used to convert weights returned by ``loader_utils._dequantize_npy`` (which
    always returns float32) into the dtype expected by the HuggingFace model.

    Parameters
    ----------
    arr:
        Float32 numpy array (any shape).
    device:
        Target device string.
    dtype:
        Target dtype: ``"float16"`` (default), ``"bfloat16"``, or ``"float32"``.

    Returns
    -------
    torch.Tensor
        Tensor on *device* with the requested dtype.
    """
    torch = _import_torch()
    _dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tgt = _dtype_map.get(dtype, torch.float16)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(
        device=device, dtype=tgt
    )
