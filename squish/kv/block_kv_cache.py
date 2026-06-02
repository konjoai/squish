"""squish/kv/block_kv_cache.py — block-level paged KV cache (v5).

Splits a prompt into fixed-size blocks (default 64 tokens) and caches the
KV state of every block independently.  A new prompt that shares a prefix
with a previously seen prompt re-uses the cached blocks and only re-prefills
the non-matching suffix.

Two storage tiers:

* **Hot tier** — in-RAM dict of ``block_hash → (keys, values)``.  Capped at
  ``hot_max_bytes`` (default 2 GiB); LRU eviction to cold tier.
* **Cold tier** — per-block numpy arrays under ``cache_dir/<hash[:2]>/<hash>.npz``.
  Survives server restarts.  LRU eviction (oldest atime first) when total
  on-disk usage exceeds ``cold_max_bytes`` (default 8 GiB).

Hash chain
----------
Each block's hash depends on its content AND the hash of the previous block,
so a "matched prefix" means the actual token-id sequence matches, not just
that some block earlier matches an unrelated cached block.

::

    block_0_hash = sha256(block_0_token_ids).hexdigest()
    block_i_hash = sha256(block_{i-1}_hash + block_i_token_ids).hexdigest()

Lookup walks the chain forward, taking the longest contiguous matching
prefix of blocks.

Wire-up
-------
``server.py`` calls ``BlockKVCache.lookup_prefix(input_ids)`` BEFORE prefill.
On a non-empty match, it restores the matched blocks' KV state into a fresh
``mlx_lm`` prompt cache and prefills only the suffix.  On generation
completion it calls ``BlockKVCache.store_blocks(input_ids, cache)`` to
record any new blocks (idempotent — known hashes are skipped).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_BLOCK_SIZE   = 64
_DEFAULT_CACHE_DIR    = Path.home() / ".cache" / "squish" / "blocks"
_DEFAULT_HOT_MAX      = 2 * 1024 * 1024 * 1024   # 2 GiB
_DEFAULT_COLD_MAX     = 8 * 1024 * 1024 * 1024   # 8 GiB
_CACHE_VERSION        = 2  # bumped in v5.1 for last_logit field


@dataclass
class BlockEntry:
    """One cached block's KV state + (v5.1) the block's last-position logit.

    The ``last_logit`` is the model's prediction for the token at position
    ``(absolute_block_end_index)`` — i.e. the token that would come
    immediately AFTER this block.  When this is the LAST cached block of a
    prompt and ``matched_tokens == prompt_tokens`` (no trailing partial
    block), the caller can sample from this logit directly to emit the
    first model-response token without running any forward pass.

    Stored as float32; size = vocab_size * 4 bytes (~600 KB for Qwen2.5).
    """
    hash:        str
    n_layers:    int
    n_tokens:    int           # number of prompt tokens covered by this block
    keys:        list[np.ndarray]   # one (1, n_heads, n_tokens, head_dim) per layer
    values:      list[np.ndarray]
    nbytes:      int           # total byte size for hot-tier accounting
    last_logit:  "np.ndarray | None" = None  # v5.1: (vocab_size,) float32
    last_used:   float = field(default_factory=time.time)


@dataclass
class PrefixMatch:
    """Result of a prefix lookup."""
    matched_blocks: list[BlockEntry]
    matched_tokens: int       # = sum(b.n_tokens for b in matched_blocks)


class BlockKVCache:
    """Two-tier (hot RAM + cold disk) block-level KV cache.

    Parameters
    ----------
    cache_dir : str | Path
        Cold-tier directory.  Created automatically.
    block_size : int
        Token count per block (default 64).  Fixed for the lifetime of the
        store — change requires clearing the cache.
    hot_max_bytes : int
        Soft limit on RAM dict size.  Excess entries evict to cold tier.
    cold_max_bytes : int
        Soft limit on cold-tier disk usage.  Excess entries are removed (LRU).
    model_key : str
        Opaque key identifying the model.  Blocks from a different model_key
        are rejected on lookup.
    """

    def __init__(
        self,
        cache_dir: "str | Path" = _DEFAULT_CACHE_DIR,
        block_size: int = _DEFAULT_BLOCK_SIZE,
        hot_max_bytes: int = _DEFAULT_HOT_MAX,
        cold_max_bytes: int = _DEFAULT_COLD_MAX,
        model_key: str = "",
    ) -> None:
        self._dir            = Path(cache_dir)
        self._block_size     = max(1, int(block_size))
        self._hot_max_bytes  = int(hot_max_bytes)
        self._cold_max_bytes = int(cold_max_bytes)
        self._model_key      = model_key
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            self._dir.chmod(0o700)
        except OSError:
            pass
        # Hot tier: OrderedDict for O(1) LRU
        self._hot: "OrderedDict[str, BlockEntry]" = OrderedDict()
        self._hot_bytes: int = 0
        # Persist block_size + model_key in a manifest so we can detect a
        # mismatch on restart and clear if needed.
        self._manifest_path = self._dir / "manifest.json"
        self._validate_or_init_manifest()

    # ── Hashing ───────────────────────────────────────────────────────────────

    def block_size(self) -> int:
        return self._block_size

    def split_blocks(self, input_ids: "list[int]") -> "list[list[int]]":
        """Split input_ids into block_size chunks (last block may be short)."""
        bs = self._block_size
        return [input_ids[i : i + bs] for i in range(0, len(input_ids), bs)]

    def chain_hash(self, input_ids: "list[int]") -> "list[str]":
        """Return per-block hashes with chained dependency on preceding blocks.

        Each block's hash mixes the previous block's hash so a matched prefix
        of N blocks proves the FIRST N blocks of the prompt match exactly.
        """
        blocks = self.split_blocks(input_ids)
        hashes: list[str] = []
        prev = self._model_key.encode() if self._model_key else b""
        for block in blocks:
            raw = np.array(block, dtype=np.int32).tobytes()
            h = hashlib.sha256(prev + raw).hexdigest()
            hashes.append(h)
            prev = h.encode()
        return hashes

    # ── Lookup (read) ─────────────────────────────────────────────────────────

    def lookup_prefix(self, input_ids: "list[int]") -> "PrefixMatch":
        """Walk the chained hashes; collect contiguous matched blocks.

        Returns the longest contiguous matching prefix. Stops at the first
        block hash not present in either tier (so the caller knows exactly
        how many tokens of the prompt are already cached).
        """
        # If we have fewer than 1 full block, the cache cannot help — we don't
        # support partial-block caching (the KV slice shape wouldn't align
        # with the block_size).  Caller will prefill the whole prompt.
        if len(input_ids) < self._block_size:
            return PrefixMatch(matched_blocks=[], matched_tokens=0)

        hashes = self.chain_hash(input_ids)
        # Walk hash chain, taking the longest contiguous prefix that exists.
        # The final block of the prompt may be short — we don't cache short
        # blocks, so we only consider full blocks.
        n_full = len(input_ids) // self._block_size
        blocks: list[BlockEntry] = []
        matched_tokens = 0
        for i in range(n_full):
            h = hashes[i]
            entry = self._get_block(h)
            if entry is None:
                break
            blocks.append(entry)
            matched_tokens += entry.n_tokens
        return PrefixMatch(matched_blocks=blocks, matched_tokens=matched_tokens)

    def _get_block(self, h: str) -> "BlockEntry | None":
        # Hot first
        if h in self._hot:
            entry = self._hot[h]
            entry.last_used = time.time()
            self._hot.move_to_end(h)  # LRU bump
            return entry
        # Cold fallback
        return self._read_cold(h)

    def _read_cold(self, h: str) -> "BlockEntry | None":
        entry_path = self._cold_path(h)
        if not entry_path.exists():
            return None
        try:
            data = np.load(str(entry_path), allow_pickle=False)
            n_layers = int(data["n_layers"])
            keys = [data[f"k_{i}"] for i in range(n_layers)]
            values = [data[f"v_{i}"] for i in range(n_layers)]
            n_tokens = int(data["n_tokens"])
            # v5.1: optional per-block last logit (legacy v5 files don't have it)
            last_logit: "np.ndarray | None" = None
            if "last_logit" in data.files:
                last_logit = data["last_logit"].astype(np.float32)
            extra_bytes = last_logit.nbytes if last_logit is not None else 0
            entry = BlockEntry(
                hash=h,
                n_layers=n_layers,
                n_tokens=n_tokens,
                keys=keys,
                values=values,
                last_logit=last_logit,
                nbytes=sum(k.nbytes for k in keys) + sum(v.nbytes for v in values) + extra_bytes,
            )
            try:
                os.utime(entry_path, None)
            except OSError:
                pass
            self._add_to_hot(entry)
            return entry
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("[block-kv-cache] cold read FAILED for %s: %s", h[:16], exc)
            return None

    # ── Store (write) ─────────────────────────────────────────────────────────

    def store_blocks(
        self,
        input_ids: "list[int]",
        per_block_keys:   "list[list[Any]]",   # outer = blocks, inner = layers
        per_block_values: "list[list[Any]]",
        per_block_last_logits: "list[Any] | None" = None,  # v5.1: one per block
    ) -> None:
        """Persist any new full blocks from the prompt's prefill.

        Parameters
        ----------
        input_ids : list[int]
            Full token-id list of the prompt.  The first ``n // block_size``
            blocks are eligible to store.
        per_block_keys, per_block_values : list of list of arrays
            Outer dim = blocks, inner = layers.  Each inner element is a
            ``(1, n_heads, block_size, head_dim)`` tensor sliced from the
            mlx_lm prompt cache.
        per_block_last_logits : list of arrays, optional
            v5.1: per-block last-position logit (vocab_size,).  Same outer
            length as ``per_block_keys``.  When present, a full-prefix-match
            lookup can sample the first response token directly from the
            stored logit, skipping the suffix forward pass entirely.

        Caller is responsible for slicing the prompt-cache's per-layer KV
        tensors.  This method only handles hashing, hot/cold writes, and
        idempotency (existing hashes are skipped).
        """
        hashes = self.chain_hash(input_ids)
        n_full = len(input_ids) // self._block_size
        n_with_logits = len(per_block_last_logits) if per_block_last_logits else 0
        for i in range(min(n_full, len(per_block_keys))):
            h = hashes[i]
            if self._has_block(h):
                continue
            block_keys = per_block_keys[i]
            block_vals = per_block_values[i]
            if not block_keys:
                continue
            keys_np = [_to_numpy_f16(k) for k in block_keys]
            vals_np = [_to_numpy_f16(v) for v in block_vals]
            logit_np: "np.ndarray | None" = None
            if i < n_with_logits and per_block_last_logits[i] is not None:
                try:
                    logit_np = _to_numpy_f32(per_block_last_logits[i])
                except (TypeError, ValueError, RuntimeError) as exc:
                    logger.warning(
                        "[block-kv-cache] block %d logit conversion failed (%s) — "
                        "stored without logit", i, exc,
                    )
                    logit_np = None
            extra_bytes = logit_np.nbytes if logit_np is not None else 0
            entry = BlockEntry(
                hash=h,
                n_layers=len(keys_np),
                n_tokens=self._block_size,
                keys=keys_np,
                values=vals_np,
                last_logit=logit_np,
                nbytes=sum(k.nbytes for k in keys_np) + sum(v.nbytes for v in vals_np) + extra_bytes,
            )
            self._add_to_hot(entry)
            self._write_cold(entry)

    def _has_block(self, h: str) -> bool:
        return h in self._hot or self._cold_path(h).exists()

    def _add_to_hot(self, entry: "BlockEntry") -> None:
        if entry.hash in self._hot:
            self._hot.move_to_end(entry.hash)
            return
        self._hot[entry.hash] = entry
        self._hot_bytes += entry.nbytes
        # Evict from hot down to budget; cold tier already has the data
        # because we always write cold after add_to_hot OR the entry came
        # from cold tier (which still has the file).
        while self._hot_bytes > self._hot_max_bytes and len(self._hot) > 1:
            _h, evicted = self._hot.popitem(last=False)
            self._hot_bytes -= evicted.nbytes

    def _write_cold(self, entry: "BlockEntry") -> None:
        entry_path = self._cold_path(entry.hash)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = entry_path.with_suffix(".tmp.npz")
        try:
            arrays: dict[str, np.ndarray] = {
                "n_layers": np.array(entry.n_layers, dtype=np.int32),
                "n_tokens": np.array(entry.n_tokens, dtype=np.int32),
            }
            for i, (k, v) in enumerate(zip(entry.keys, entry.values, strict=True)):
                arrays[f"k_{i}"] = k
                arrays[f"v_{i}"] = v
            # v5.1: persist per-block last logit when present
            if entry.last_logit is not None:
                arrays["last_logit"] = entry.last_logit
            np.savez(str(tmp), **arrays)
            os.replace(str(tmp), str(entry_path))
        except (OSError, ValueError) as exc:
            logger.warning("[block-kv-cache] cold write failed (%s)", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return
        # Async-ish cold eviction (1 in 32 writes)
        import random
        if random.random() < 1 / 32:
            self._evict_cold_lru()

    def _cold_path(self, h: str) -> Path:
        return self._dir / h[:2] / f"{h}.npz"

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _evict_cold_lru(self) -> int:
        total = 0
        entries: list[tuple[float, Path, int]] = []
        for sub in self._dir.iterdir():
            if not sub.is_dir():
                continue
            for entry_path in sub.iterdir():
                if entry_path.suffix != ".npz":
                    continue
                try:
                    stat = entry_path.stat()
                except OSError:
                    continue
                total += stat.st_size
                entries.append((stat.st_atime, entry_path, stat.st_size))
        if total <= self._cold_max_bytes:
            return 0
        entries.sort(key=lambda e: e[0])  # oldest atime first
        evicted = 0
        for _atime, path, sz in entries:
            if total <= self._cold_max_bytes:
                break
            try:
                path.unlink(missing_ok=True)
                total -= sz
                evicted += 1
            except OSError:
                pass
        return evicted

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "block_size":     self._block_size,
            "hot_entries":    len(self._hot),
            "hot_bytes":      self._hot_bytes,
            "hot_max_bytes":  self._hot_max_bytes,
            "cold_max_bytes": self._cold_max_bytes,
            "model_key":      self._model_key,
        }

    def clear(self) -> None:
        """Wipe both tiers (test/admin use only)."""
        self._hot.clear()
        self._hot_bytes = 0
        for sub in self._dir.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    try:
                        f.unlink()
                    except OSError:
                        pass
                try:
                    sub.rmdir()
                except OSError:
                    pass

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _validate_or_init_manifest(self) -> None:
        if not self._manifest_path.exists():
            self._write_manifest()
            return
        try:
            m = json.loads(self._manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("[block-kv-cache] manifest corrupt — recreating cache")
            self.clear()
            self._write_manifest()
            return
        if m.get("version") != _CACHE_VERSION or m.get("block_size") != self._block_size:
            logger.warning(
                "[block-kv-cache] manifest mismatch "
                "(version=%s, block_size=%s) — clearing cache",
                m.get("version"), m.get("block_size"),
            )
            self.clear()
            self._write_manifest()

    def _write_manifest(self) -> None:
        m = {
            "version":    _CACHE_VERSION,
            "block_size": self._block_size,
            "model_key":  self._model_key,
            "created":    time.time(),
        }
        try:
            self._manifest_path.write_text(json.dumps(m))
        except OSError as exc:
            logger.warning("[block-kv-cache] manifest write failed (%s)", exc)


# ── mlx ↔ numpy helpers ───────────────────────────────────────────────────────

def _to_numpy_f16(arr) -> np.ndarray:
    """Convert mlx or numpy array to float16 numpy.

    Routes mlx bfloat16 through float32 because numpy has no native bf16
    (same fix as squish.kv.prompt_kv_cache._to_numpy).
    """
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float16)
    try:
        import mlx.core as mx
        mx.eval(arr)
        try:
            return np.array(arr, dtype=np.float16)
        except (TypeError, RuntimeError):
            return np.array(arr.astype(mx.float32), dtype=np.float16)
    except ImportError:
        raise TypeError(
            f"Cannot convert {type(arr)} to numpy — mlx not available"
        ) from None


def _to_numpy_f32(arr) -> np.ndarray:
    """Convert mlx or numpy array to float32 numpy (for the per-block logit).

    Float32 keeps the logit precise enough for stable sampling — the
    logit is only used once per cache hit, not throughout decode.
    """
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32)
    try:
        import mlx.core as mx
        mx.eval(arr)
        try:
            return np.array(arr, dtype=np.float32)
        except (TypeError, RuntimeError):
            return np.array(arr.astype(mx.float32), dtype=np.float32)
    except ImportError:
        raise TypeError(
            f"Cannot convert {type(arr)} to numpy — mlx not available"
        ) from None


# ── Slice/concat helpers used by server.py ────────────────────────────────────

def per_block_last_logits_from_full_logits(
    full_logits,  # mx.array, shape [1, n_tokens, vocab]
    n_blocks: int,
    block_size: int,
) -> "list[Any]":
    """Extract the last-position logit of each block from a full-prompt
    forward pass.

    The full forward pass on ``prompt_ids[:n_blocks*block_size]`` yields a
    tensor of shape ``[1, n_blocks*block_size, vocab]``.  Block ``i``'s last
    position is index ``(i + 1) * block_size - 1``; its logit predicts the
    token that would come AFTER the block.  We slice those indices out and
    return one ``(vocab,)`` array per block.

    Returns mlx arrays; conversion to numpy happens inside
    ``BlockKVCache.store_blocks`` via ``_to_numpy_f32``.
    """
    out: list[Any] = []
    for i in range(n_blocks):
        idx = (i + 1) * block_size - 1
        out.append(full_logits[0, idx])
    return out


def slice_cache_into_blocks(
    cache,
    block_size: int,
    n_blocks: int,
    n_layers: int,
) -> "tuple[list[list[Any]], list[list[Any]]]":
    """Slice an mlx_lm prompt cache's per-layer KV tensors into block-sized chunks.

    Each layer cache has shape (1, n_heads, total_tokens, head_dim) after
    prefill of ``n_blocks * block_size`` tokens.  We return two lists shaped
    [n_blocks][n_layers].

    The returned arrays are mlx-views; they share memory with the cache.
    Callers should copy via ``_to_numpy_f16`` before persisting.
    """
    per_block_keys: list[list[Any]] = []
    per_block_values: list[list[Any]] = []
    for b in range(n_blocks):
        block_keys: list[Any] = []
        block_vals: list[Any] = []
        start = b * block_size
        end   = start + block_size
        for layer in range(n_layers):
            layer_cache = cache[layer]
            k_full = getattr(layer_cache, "keys", None)
            v_full = getattr(layer_cache, "values", None)
            if k_full is None or v_full is None:
                return [], []
            block_keys.append(k_full[..., start:end, :])
            block_vals.append(v_full[..., start:end, :])
        per_block_keys.append(block_keys)
        per_block_values.append(block_vals)
    return per_block_keys, per_block_values


def restore_blocks_to_cache(
    cache,
    matched_blocks: "list[BlockEntry]",
) -> "tuple[int, int] | None":
    """Concatenate per-block KV slices into a fresh mlx_lm prompt cache.

    Returns ``(n_layers, n_tokens_restored)`` on success, ``None`` if the
    cache type is unsupported or layer counts mismatch.
    """
    if not matched_blocks:
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None
    n_layers = matched_blocks[0].n_layers
    if len(cache) != n_layers:
        logger.warning(
            "[block-kv-cache] layer mismatch (cache=%d entry=%d)",
            len(cache), n_layers,
        )
        return None
    n_tokens = sum(b.n_tokens for b in matched_blocks)
    for layer_idx in range(n_layers):
        # Concatenate all blocks' keys/values for this layer along the seq dim
        ks = [mx.array(b.keys[layer_idx])   for b in matched_blocks]
        vs = [mx.array(b.values[layer_idx]) for b in matched_blocks]
        keys_full = mx.concatenate(ks, axis=2)
        vals_full = mx.concatenate(vs, axis=2)
        layer_cache = cache[layer_idx]
        if not (hasattr(layer_cache, "keys") and hasattr(layer_cache, "values")):
            return None
        layer_cache.keys   = keys_full
        layer_cache.values = vals_full
        if hasattr(layer_cache, "offset"):
            layer_cache.offset = n_tokens
    return (n_layers, n_tokens)
