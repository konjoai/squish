"""squish/kv/k8v4_codec.py — K8V4 quantized codec for on-disk KV-cache persistence.

The disk KV cache (:class:`squish.kv.prompt_kv_cache.PromptKVStore`) stores prefilled
key/value tensors as float16 ``.npy`` files so a repeated prompt can skip prefill on
restart.  At ~2 bytes per element that dominates the cache's disk footprint and the
restore-time I/O.

**K8V4** quantizes that on-disk state — *keys to INT8, values to INT4* — using
group-wise asymmetric affine quantization along ``head_dim``.  The bit allocation is
deliberate and was validated empirically on an M3 (quantize → restore → greedy decode
vs the float16 baseline):

==========  ==================  ============================================
Scheme      40-token match      Verdict
==========  ==================  ============================================
K8V8        40/40               lossless (but only ~2x vs f16)
**K8V4**    **40/40**           **lossless, ~2.7x — shipped**
K4V4        2/40                broken — INT4 *keys* destroy decode at token 2
==========  ==================  ============================================

Keys carry far more decode-critical precision than values, so they stay at 8 bits;
values tolerate 4 bits with no greedy-decode divergence.  INT4 keys are *never*
emitted by this codec — :data:`K_BITS` is fixed at 8.

The codec is pure NumPy (no MLX), so it is importable on every platform and unit-
testable without a model, per the project's MLX-gating rule.  Callers convert MLX
arrays to NumPy before serialization and back after restore.

Disk layout (per layer, one ``.npz`` each):
  ``q``       packed quantized codes (uint8; for 4-bit, two nibbles per byte)
  ``scale``   per-group affine scale  (float32, shape ``(..., n_groups)``)
  ``zero``    per-group affine offset (float32, same shape as ``scale``)
  ``bits``    int8 scalar — quantization bit width (8 or 4)
  ``gsize``   int32 scalar — group size used along the last axis
  ``packed``  int8 scalar — 1 if ``q`` is nibble-packed, else 0
  ``shape``   int32 vector — original tensor shape (for unpack/reshape)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed bit allocation. K_BITS is 8 and is *not* configurable: INT4 keys broke
# greedy decode at token 2 in validation. V_BITS is 4 — the validated lossless point.
K_BITS = 8
V_BITS = 4
DEFAULT_GROUP_SIZE = 64


def _effective_group(dim: int, group_size: int) -> int:
    """Largest group ``<= group_size`` that evenly divides ``dim``.

    Group-wise quant needs ``dim % g == 0``; when the requested size doesn't
    divide ``head_dim`` we fall back to the whole axis as one group rather than
    silently mis-quantizing a ragged tail.
    """
    if group_size <= 0 or dim % group_size != 0:
        return dim
    return group_size


def quantize_array(arr: np.ndarray, bits: int, group_size: int = DEFAULT_GROUP_SIZE) -> dict:
    """Group-wise asymmetric affine quantization of ``arr`` along its last axis.

    Parameters
    ----------
    arr : np.ndarray
        Float tensor of shape ``(..., D)`` (e.g. ``(1, n_heads, seq, head_dim)``).
    bits : int
        Quantization width (8 for keys, 4 for values).
    group_size : int
        Group length along ``D``; clamped to a divisor of ``D``.

    Returns
    -------
    dict
        npz-ready payload (see module docstring for the schema).
    """
    if bits not in (4, 8):
        raise ValueError(f"K8V4 codec supports 4- or 8-bit only, got {bits}")
    shape = arr.shape
    dim = shape[-1]
    g = _effective_group(dim, group_size)
    n_groups = dim // g

    a = arr.reshape(-1, n_groups, g).astype(np.float32)
    lo = a.min(axis=-1, keepdims=True)
    hi = a.max(axis=-1, keepdims=True)
    qmax = (1 << bits) - 1
    scale = (hi - lo) / qmax
    # Constant groups (hi == lo) would divide by zero; use unit scale so every
    # code maps back to ``lo`` exactly.
    scale = np.where(scale == 0.0, 1.0, scale)
    codes = np.rint((a - lo) / scale)
    codes = np.clip(codes, 0, qmax).astype(np.uint8).reshape(shape)

    packed = 0
    if bits == 4 and dim % 2 == 0:
        # Nibble-pack pairs along the last axis: low nibble = even index.
        flat = codes.reshape(-1, dim)
        codes = (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)
        codes = codes.reshape(*shape[:-1], dim // 2)
        packed = 1

    return {
        "q": codes,
        "scale": scale.reshape(*shape[:-1], n_groups).astype(np.float32),
        "zero": lo.reshape(*shape[:-1], n_groups).astype(np.float32),
        "bits": np.array(bits, dtype=np.int8),
        "gsize": np.array(g, dtype=np.int32),
        "packed": np.array(packed, dtype=np.int8),
        "shape": np.array(shape, dtype=np.int32),
    }


def dequantize_array(payload: dict, dtype=np.float16) -> np.ndarray:
    """Reconstruct the float tensor quantized by :func:`quantize_array`."""
    shape = tuple(int(x) for x in payload["shape"])
    dim = shape[-1]
    g = int(payload["gsize"])
    n_groups = dim // g
    codes = np.asarray(payload["q"])

    if int(payload["packed"]):
        flat = codes.reshape(-1, dim // 2)
        unpacked = np.empty((flat.shape[0], dim), dtype=np.uint8)
        unpacked[:, 0::2] = flat & 0x0F
        unpacked[:, 1::2] = (flat >> 4) & 0x0F
        codes = unpacked

    qf = codes.reshape(-1, n_groups, g).astype(np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32).reshape(-1, n_groups, 1)
    zero = np.asarray(payload["zero"], dtype=np.float32).reshape(-1, n_groups, 1)
    out = qf * scale + zero
    return out.reshape(shape).astype(dtype)


def save_quantized(
    path: Path, arr: np.ndarray, bits: int, group_size: int = DEFAULT_GROUP_SIZE
) -> None:
    """Quantize ``arr`` and write it to ``path`` (``.npz``, compressed)."""
    np.savez_compressed(str(path), **quantize_array(arr, bits, group_size))


def load_quantized(path: Path, dtype=np.float16) -> np.ndarray:
    """Load and dequantize a ``.npz`` written by :func:`save_quantized`.

    A truncated/corrupt archive surfaces as :class:`ValueError` so callers can
    treat it as a clean cache miss (the same handling as a bad ``.npy``).
    """
    import zipfile

    try:
        with np.load(str(path), allow_pickle=False) as data:
            return dequantize_array({k: data[k] for k in data.files}, dtype)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"corrupt K8V4 archive {path}: {exc}") from exc


def load_layer_auto(entry_dir: Path, kind: str, index: int, dtype=np.float16) -> np.ndarray:
    """Load layer ``index`` of ``kind`` (``"k"``/``"v"``), auto-detecting format.

    Prefers the quantized ``{kind}_{index}.npz`` and falls back to the legacy
    float16 ``{kind}_{index}.npy`` so a single store can hold both formats and
    old caches keep loading after the codec lands.
    """
    npz = entry_dir / f"{kind}_{index}.npz"
    if npz.exists():
        return load_quantized(npz, dtype)
    npy = entry_dir / f"{kind}_{index}.npy"
    return np.load(str(npy)).astype(dtype)


def compression_ratio(arr: np.ndarray, bits: int, group_size: int = DEFAULT_GROUP_SIZE) -> float:
    """Raw (pre-``savez`` compression) bytes saved vs a float16 baseline.

    Counts the quantized codes plus the per-group scale/zero overhead against
    ``2 * arr.size`` float16 bytes — a lower bound on the on-disk win (zlib
    shrinks the codes further).
    """
    payload = quantize_array(arr, bits, group_size)
    quant_bytes = payload["q"].nbytes + payload["scale"].nbytes + payload["zero"].nbytes
    fp16_bytes = 2 * arr.size
    return fp16_bytes / quant_bytes if quant_bytes else float("inf")
