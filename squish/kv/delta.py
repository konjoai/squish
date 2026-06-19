"""
squish/kv/delta.py — KV cache delta / diff compression.

Long-running iterative generation (chain-of-thought, agent loops, RAG with
context refresh) repeatedly mutates a KV cache.  Snapshotting the whole
cache after every step is wasteful — only a small suffix of tokens changes
between consecutive states.

:class:`KVCacheDelta` encodes the difference between a *base* snapshot and
a *target* snapshot as:

  base_len   — how many leading tokens are unchanged
  new_keys   — keys appended after the unchanged prefix
  new_values — values appended after the unchanged prefix
  truncate   — how many trailing base tokens to drop before appending

The delta size is therefore O(|target| − |unchanged prefix|), regardless of
the prefix length.  Reconstruction is exact when the prefix is bit-identical;
the diff routine uses a fast byte-level scan to find the longest common
prefix so prefix detection is O(min(len_a, len_b) × n_heads × head_dim)
bytes compared (a few MB for a 32K-context layer, micro-seconds in NumPy).

Wire format (when ``encode_bytes()`` is used)
---------------------------------------------
A delta serialises to bytes with a tiny header:

  magic        b"SQDLT\\x01"               (6 bytes)
  base_len     uint32                       (4 bytes, little-endian)
  truncate     uint32                       (4 bytes, little-endian)
  n_new        uint32                       (4 bytes, little-endian)
  n_heads      uint16                       (2 bytes, little-endian)
  head_dim     uint16                       (2 bytes, little-endian)
  dtype        1-char ASCII ('H' for fp16,
                             'F' for fp32)  (1 byte)
  padding      (1 byte to 4-align payload)
  new_keys     n_new × n_heads × head_dim × itemsize bytes
  new_values   same

The format is self-describing; decoding requires no out-of-band metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


_MAGIC = b"SQDLT\x01"
_HEADER_LEN = 24                 # 6 + 4 + 4 + 4 + 2 + 2 + 1 + 1 (padding)
_DTYPE_BY_CHAR = {"H": np.float16, "F": np.float32}
_CHAR_BY_DTYPE = {np.dtype("float16"): "H", np.dtype("float32"): "F"}


@dataclass
class KVCacheDelta:
    """Encoded diff between two stacks of token-shaped K/V tensors.

    Attributes
    ----------
    base_len   : tokens kept from the base.
    truncate   : tokens removed from the *end* of the base before appending.
    new_keys   : (n_new, n_heads, head_dim) array of appended keys.
    new_values : same shape as ``new_keys``.

    The base is implicit — the delta is only meaningful relative to a known
    base snapshot.  :func:`apply` performs the merge.
    """
    base_len: int
    truncate: int
    new_keys: np.ndarray
    new_values: np.ndarray

    # ── derived ──────────────────────────────────────────────────────────

    @property
    def n_new(self) -> int:
        return int(self.new_keys.shape[0])

    @property
    def is_empty(self) -> bool:
        """True when the delta represents zero change."""
        return self.truncate == 0 and self.n_new == 0

    def size_bytes(self) -> int:
        """In-memory footprint of the delta tensors."""
        return int(self.new_keys.nbytes + self.new_values.nbytes)

    # ── factory + apply ──────────────────────────────────────────────────

    @classmethod
    def compute(
        cls,
        base_keys:  np.ndarray,
        base_values: np.ndarray,
        target_keys:  np.ndarray,
        target_values: np.ndarray,
        atol: float = 0.0,
    ) -> "KVCacheDelta":
        """Compute the delta from base → target.

        Arrays must be ``(n_tokens, n_heads, head_dim)`` with identical
        ``(n_heads, head_dim)`` between base and target.  ``n_tokens`` may
        differ.

        Parameters
        ----------
        atol : absolute tolerance for prefix-match comparison (0 = exact).
            Useful when the base and target carry fp16 round-off.
        """
        _validate_kv_stack(base_keys,   base_values,   name="base")
        _validate_kv_stack(target_keys, target_values, name="target")

        if base_keys.shape[1:] != target_keys.shape[1:]:
            raise ValueError(
                f"shape mismatch: base {base_keys.shape[1:]} vs "
                f"target {target_keys.shape[1:]}"
            )

        base_n   = int(base_keys.shape[0])
        target_n = int(target_keys.shape[0])
        prefix_max = min(base_n, target_n)

        # Find the longest common prefix (LCP) — token-by-token equality.
        # For small dims this is fast; for big dims we still pay once per
        # snapshot pair, not per token.
        common = 0
        if prefix_max > 0:
            if atol == 0.0:
                # vectorised equality over the common-prefix slice
                eq_k = np.all(
                    base_keys[:prefix_max] == target_keys[:prefix_max],
                    axis=(1, 2),
                )
                eq_v = np.all(
                    base_values[:prefix_max] == target_values[:prefix_max],
                    axis=(1, 2),
                )
                eq = eq_k & eq_v                       # (prefix_max,)
            else:
                eq_k = np.all(
                    np.abs(base_keys[:prefix_max] - target_keys[:prefix_max])
                    <= atol, axis=(1, 2),
                )
                eq_v = np.all(
                    np.abs(base_values[:prefix_max] - target_values[:prefix_max])
                    <= atol, axis=(1, 2),
                )
                eq = eq_k & eq_v
            # LCP = index of first False, or prefix_max if all-True
            if eq.all():
                common = int(prefix_max)
            else:
                common = int(np.argmin(eq))

        truncate = base_n - common
        new_k = np.array(target_keys[common:],   copy=True)
        new_v = np.array(target_values[common:], copy=True)
        return cls(
            base_len=common, truncate=truncate,
            new_keys=new_k, new_values=new_v,
        )

    def apply(
        self,
        base_keys: np.ndarray, base_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the target snapshot from a base.

        Returns ``(target_keys, target_values)`` — fresh arrays, the base is
        not mutated.
        """
        _validate_kv_stack(base_keys, base_values, name="base")
        base_n = int(base_keys.shape[0])
        if self.base_len + self.truncate != base_n:
            raise ValueError(
                f"delta does not match base: base_len + truncate = "
                f"{self.base_len + self.truncate}, but base has {base_n} tokens"
            )
        if self.n_new > 0 and self.new_keys.shape[1:] != base_keys.shape[1:]:
            raise ValueError(
                f"delta payload shape {self.new_keys.shape[1:]} does not "
                f"match base {base_keys.shape[1:]}"
            )

        kept_k = base_keys[: self.base_len]
        kept_v = base_values[: self.base_len]

        if self.n_new == 0:
            return np.array(kept_k, copy=True), np.array(kept_v, copy=True)

        target_k = np.concatenate([kept_k, self.new_keys], axis=0)
        target_v = np.concatenate([kept_v, self.new_values], axis=0)
        return target_k, target_v

    # ── wire format ──────────────────────────────────────────────────────

    def encode_bytes(self) -> bytes:
        """Serialise to the compact wire format (see module docstring)."""
        # Use new_keys dtype as the wire dtype.  Fallback to fp16 if empty.
        if self.n_new > 0:
            dt = np.dtype(self.new_keys.dtype)
            if dt not in _CHAR_BY_DTYPE:
                raise ValueError(f"unsupported dtype for wire format: {dt}")
            ch = _CHAR_BY_DTYPE[dt]
            n_heads  = int(self.new_keys.shape[1])
            head_dim = int(self.new_keys.shape[2])
        else:
            ch = "H"
            n_heads  = 0
            head_dim = 0

        header = (
            _MAGIC
            + int(self.base_len).to_bytes(4, "little", signed=False)
            + int(self.truncate).to_bytes(4, "little", signed=False)
            + int(self.n_new).to_bytes(4, "little", signed=False)
            + int(n_heads).to_bytes(2, "little", signed=False)
            + int(head_dim).to_bytes(2, "little", signed=False)
            + ch.encode("ascii")
            + b"\x00"     # padding to 24 bytes
        )
        if len(header) != _HEADER_LEN:  # pragma: no cover - header is a fixed 24-byte layout; this guards against an accidental format edit
            raise RuntimeError(f"header length {len(header)} != {_HEADER_LEN}")
        if self.n_new == 0:
            return header
        # Ensure contiguous bytes for downstream encoders (e.g. zstd).
        kbuf = np.ascontiguousarray(self.new_keys).tobytes()
        vbuf = np.ascontiguousarray(self.new_values).tobytes()
        return header + kbuf + vbuf

    @classmethod
    def decode_bytes(cls, buf: bytes) -> "KVCacheDelta":
        """Inverse of :meth:`encode_bytes`."""
        if len(buf) < _HEADER_LEN:
            raise ValueError(f"buffer too short: {len(buf)} bytes")
        if buf[:6] != _MAGIC:
            raise ValueError(f"bad magic: {buf[:6]!r}")
        base_len = int.from_bytes(buf[6:10],  "little")
        truncate = int.from_bytes(buf[10:14], "little")
        n_new    = int.from_bytes(buf[14:18], "little")
        n_heads  = int.from_bytes(buf[18:20], "little")
        head_dim = int.from_bytes(buf[20:22], "little")
        ch       = chr(buf[22])
        if ch not in _DTYPE_BY_CHAR:
            raise ValueError(f"unsupported dtype char: {ch!r}")
        dtype = _DTYPE_BY_CHAR[ch]

        if n_new == 0:
            empty = np.zeros((0, 0, 0), dtype=dtype)
            return cls(
                base_len=base_len, truncate=truncate,
                new_keys=empty, new_values=empty,
            )

        itemsize = np.dtype(dtype).itemsize
        per_tensor = n_new * n_heads * head_dim * itemsize
        expected = _HEADER_LEN + 2 * per_tensor
        if len(buf) != expected:
            raise ValueError(
                f"buffer length {len(buf)} != expected {expected} "
                f"(n_new={n_new}, n_heads={n_heads}, head_dim={head_dim})"
            )
        shape = (n_new, n_heads, head_dim)
        new_k = np.frombuffer(
            buf, dtype=dtype, count=n_new * n_heads * head_dim,
            offset=_HEADER_LEN,
        ).reshape(shape).copy()
        new_v = np.frombuffer(
            buf, dtype=dtype, count=n_new * n_heads * head_dim,
            offset=_HEADER_LEN + per_tensor,
        ).reshape(shape).copy()
        return cls(
            base_len=base_len, truncate=truncate,
            new_keys=new_k, new_values=new_v,
        )


def _validate_kv_stack(
    keys: np.ndarray, values: np.ndarray, *, name: str,
) -> None:
    if keys.ndim != 3:
        raise ValueError(
            f"{name}_keys must be 3-D (n_tokens, n_heads, head_dim); "
            f"got shape {keys.shape}"
        )
    if keys.shape != values.shape:
        raise ValueError(
            f"{name}_keys shape {keys.shape} != {name}_values shape {values.shape}"
        )


def delta_from_layer_caches(
    base_layers:   Sequence[tuple[np.ndarray, np.ndarray]],
    target_layers: Sequence[tuple[np.ndarray, np.ndarray]],
    atol: float = 0.0,
) -> list[KVCacheDelta]:
    """Compute one :class:`KVCacheDelta` per layer.

    ``base_layers[i]`` / ``target_layers[i]`` are ``(keys, values)`` tuples of
    shape ``(n_tokens, n_heads, head_dim)``.
    """
    if len(base_layers) != len(target_layers):
        raise ValueError(
            f"layer count mismatch: base={len(base_layers)}, "
            f"target={len(target_layers)}"
        )
    return [
        KVCacheDelta.compute(b_k, b_v, t_k, t_v, atol=atol)
        for (b_k, b_v), (t_k, t_v) in zip(base_layers, target_layers, strict=True)
    ]
