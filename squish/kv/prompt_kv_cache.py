"""squish/kv/prompt_kv_cache.py — disk-backed KV cache keyed by prompt prefix hash.

For repeated prompts (the squish use case — commit messages, PR descriptions,
code review prompts) this skips prefill entirely.  The first request runs full
prefill and saves the KV state; subsequent requests with the same prompt prefix
load the saved state and skip to decode.

Algorithm
---------
1. Hash the prompt prefix (everything before the user's variable input) using
   SHA-256 (first 32 hex chars = 128 bits — collision probability negligible).
2. On inference, look up ~/.cache/squish/kv_cache/<hash>/ for saved KV state.
3. If found and valid, deserialise the KV arrays; skip prefill.
4. If not found, run prefill, serialise KV state to the cache dir.
5. LRU eviction: when total on-disk size exceeds ``max_bytes`` (default 1 GB),
   remove the least recently accessed entries until under budget.

Implementation notes
--------------------
MLX arrays are evaluated to numpy and saved as .npy files.  On load they are
re-wrapped as mlx arrays.  This is a fast path: numpy → mlx conversion is a
zero-copy memory view on Apple Silicon (Metal shared memory).

The cache stores per-layer KV arrays (keys + values) for the prompt prefix.
The offset into the model's KVCache is also stored so the subsequent decode
step can position the cache correctly.

Security
--------
Cache entries are keyed by SHA-256 of the full prompt (not just a prefix) to
avoid poisoning attacks.  Cache directories are mode 0o700.

Thread safety
-------------
Concurrent reads are safe (each request opens its own file handles).
Concurrent writes use a per-hash file lock (lockfile alongside the .npy files).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "squish" / "kv_cache"
_DEFAULT_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
_CACHE_VERSION     = 1


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class KVCacheEntry:
    """Deserialised KV cache entry for one prompt."""
    prompt_hash: str
    n_layers:    int
    offset:      int                          # token count prefilled
    keys:        list[np.ndarray]             # one (1, n_heads, seq, head_dim) per layer
    values:      list[np.ndarray]             # same shape
    # v4.2: post-prefill logit so callers can sample the first generated token
    # without running another forward pass.  Float32, shape (vocab_size,).
    # Optional for backwards compatibility with v4.1 entries.
    last_logit:  "np.ndarray | None" = None
    model_key:   str = ""                     # model identifier (for multi-model safety)
    saved_at:    float = field(default_factory=time.time)


class PromptKVStore:
    """Disk-backed KV cache store.

    Parameters
    ----------
    cache_dir : Path | str
        Root directory for cache entries.  Created automatically.
    max_bytes : int
        Soft limit on total on-disk cache size in bytes.  When exceeded, LRU
        entries are evicted until the total is under budget.
    model_key : str
        Opaque key identifying the model (path hash or name).  Used to
        invalidate entries from a different model.
    """

    def __init__(
        self,
        cache_dir: "Path | str" = _DEFAULT_CACHE_DIR,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        model_key: str = "",
    ) -> None:
        self._dir      = Path(cache_dir)
        self._max_bytes = max_bytes
        self._model_key = model_key
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dir.chmod(0o700)

    # ── Key ───────────────────────────────────────────────────────────────────

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Return a 32-hex-char SHA-256 prefix for *prompt*."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, prompt: str, lazy_kv: bool = False) -> "KVCacheEntry | None":
        """Load a cached KV state for *prompt*.

        Returns None if not cached or the entry is stale/corrupt.

        Parameters
        ----------
        lazy_kv : bool, default False
            v4.2 fast-path: when True, only the metadata and ``last_logit`` are
            read up-front (small).  The per-layer ``keys``/``values`` arrays
            are populated lazily by ``restore_kv_state`` via
            ``entry._lazy_kv_dir`` so a caller that emits the first token from
            ``last_logit`` can yield BEFORE paying the ~5 ms of npy disk I/O
            plus the ~100 ms numpy→mlx copies on KV restore.
        """
        h    = self.hash_prompt(prompt)
        d    = self._entry_dir(h)
        meta = d / "meta.json"
        if not meta.exists():
            return None

        try:
            with open(meta) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("corrupt meta for hash %s — evicting", h)
            self._remove_entry(d)
            return None

        # Version + model key check
        if m.get("version") != _CACHE_VERSION:
            self._remove_entry(d)
            return None
        if self._model_key and m.get("model_key") != self._model_key:
            return None

        # v4.2: optional last_logit (small — vocab-sized)
        last_logit: "np.ndarray | None" = None
        logit_path = d / "last_logit.npy"
        if logit_path.exists():
            try:
                last_logit = np.load(str(logit_path)).astype(np.float32)
            except (OSError, ValueError):
                last_logit = None  # missing/corrupt logit — fall back to forward pass

        # Per-layer KV arrays: eager by default, deferred when lazy_kv=True
        keys: "list[np.ndarray] | list[None]"
        values: "list[np.ndarray] | list[None]"
        if lazy_kv:
            keys   = [None] * m["n_layers"]
            values = [None] * m["n_layers"]
        else:
            try:
                keys   = [np.load(str(d / f"k_{i}.npy")) for i in range(m["n_layers"])]
                values = [np.load(str(d / f"v_{i}.npy")) for i in range(m["n_layers"])]
            except (OSError, ValueError):
                logger.warning("corrupt npy arrays for hash %s — evicting", h)
                self._remove_entry(d)
                return None

        # Touch access time for LRU
        _touch(meta)

        entry = KVCacheEntry(
            prompt_hash = h,
            n_layers    = m["n_layers"],
            offset      = m["offset"],
            keys        = keys,
            values      = values,
            last_logit  = last_logit,
            model_key   = m.get("model_key", ""),
            saved_at    = m.get("saved_at", 0.0),
        )
        if lazy_kv:
            # Stash the dir so restore_kv_state can demand-load the npy files.
            entry._lazy_kv_dir = d  # type: ignore[attr-defined]
        return entry

    # ── Write ─────────────────────────────────────────────────────────────────

    def put(
        self,
        prompt:  str,
        keys:    "list[Any]",   # mlx arrays or numpy arrays
        values:  "list[Any]",
        offset:  int,
        last_logit: "Any | None" = None,
    ) -> None:
        """Save the KV state for *prompt* to disk.

        Parameters
        ----------
        prompt : str
            The full prompt string (used as the cache key).
        keys : list
            Per-layer key arrays (mlx or numpy, shape (1, n_heads, seq, head_dim)).
        values : list
            Per-layer value arrays (same shape as keys).
        offset : int
            Number of tokens that were prefilled (= len(tokenized prompt)).
        last_logit : numpy or mlx array | None
            (v4.2) Post-prefill final-position logit vector, shape (vocab_size,).
            When present, ``get`` returns it on the entry so callers can sample
            the first generated token without running another forward pass.
        """
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")

        h  = self.hash_prompt(prompt)
        d  = self._entry_dir(h)
        # Create directory first, then acquire file lock
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
        lock = d / ".lock"

        # Acquire a file lock (best-effort; avoids two writers racing)
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            return  # another writer is already saving this entry

        try:
            n_layers = len(keys)
            for i, (k, v) in enumerate(zip(keys, values, strict=True)):
                k_np = _to_numpy(k)
                v_np = _to_numpy(v)
                np.save(str(d / f"k_{i}.npy"), k_np)
                np.save(str(d / f"v_{i}.npy"), v_np)

            # v4.2: persist the post-prefill logit alongside the KV state.
            # Stored as float32 for sampling stability (vocab dim only — small).
            has_logit = False
            if last_logit is not None:
                try:
                    logit_np = _to_numpy(last_logit).astype(np.float32)
                    np.save(str(d / "last_logit.npy"), logit_np)
                    has_logit = True
                except (TypeError, ValueError, RuntimeError) as exc:
                    logger.warning(
                        "[prompt-kv-cache] last_logit save failed (%s) — "
                        "entry stored without it", exc,
                    )

            meta = {
                "version":   _CACHE_VERSION,
                "n_layers":  n_layers,
                "offset":    offset,
                "model_key": self._model_key,
                "saved_at":  time.time(),
                "prompt_len": len(prompt),
                "has_logit": has_logit,
            }
            (d / "meta.json").write_text(json.dumps(meta))

            logger.debug("KV cached: hash=%s offset=%d layers=%d logit=%s",
                         h, offset, n_layers, has_logit)
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

        # Async-style eviction: run only occasionally (1 in 20 writes)
        import random
        if random.random() < 0.05:
            self._evict_lru()

    # ── Invalidation ─────────────────────────────────────────────────────────

    def invalidate(self, prompt: str) -> bool:
        """Remove the cached entry for *prompt*.  Returns True if removed."""
        h = self.hash_prompt(prompt)
        d = self._entry_dir(h)
        if d.exists():
            self._remove_entry(d)
            return True
        return False

    def clear(self) -> int:
        """Remove all entries.  Returns count of removed entries."""
        count = 0
        for d in self._dir.iterdir():
            if d.is_dir():
                self._remove_entry(d)
                count += 1
        return count

    # ── Size / stats ──────────────────────────────────────────────────────────

    def total_bytes(self) -> int:
        """Return total on-disk size of the cache in bytes."""
        total = 0
        for p in self._dir.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def entry_count(self) -> int:
        """Return number of cached entries."""
        return sum(1 for d in self._dir.iterdir() if d.is_dir())

    # ── LRU eviction ─────────────────────────────────────────────────────────

    def _evict_lru(self) -> int:
        """Evict entries until total_bytes() <= max_bytes.  Returns eviction count."""
        if self.total_bytes() <= self._max_bytes:
            return 0

        entries = []
        for d in self._dir.iterdir():
            if not d.is_dir():
                continue
            meta = d / "meta.json"
            if not meta.exists():
                continue
            try:
                atime = meta.stat().st_atime
            except OSError:
                atime = 0.0
            entries.append((atime, d))

        # Sort oldest access first
        entries.sort(key=lambda x: x[0])
        evicted = 0
        for _, d in entries:
            if self.total_bytes() <= self._max_bytes:
                break
            self._remove_entry(d)
            evicted += 1

        if evicted:
            logger.info("KV cache LRU evicted %d entries", evicted)
        return evicted

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _entry_dir(self, h: str) -> Path:
        return self._dir / h

    def _remove_entry(self, d: Path) -> None:
        import shutil
        try:
            shutil.rmtree(d)
        except Exception:
            pass


# ── mlx / numpy conversion helpers ────────────────────────────────────────────

def _to_numpy(arr) -> np.ndarray:
    """Convert an mlx array or numpy array to float16 numpy.

    Routes mlx bfloat16 arrays through float32 first because numpy has no
    native bf16 dtype — the direct buffer cast raises
    "PEP 3118 buffer format string B does not match dtype B item size 1".

    Raises ``TypeError`` for anything that isn't an ndarray or an mlx.core
    array — preventing dict / list / scalar values from silently entering
    the conversion path and producing confusing downstream errors.
    """
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float16)
    # mlx array — evaluate, then cast to a numpy-supported dtype before copy
    try:
        import mlx.core as mx
    except ImportError:
        raise TypeError(
            f"Cannot convert {type(arr).__name__} to numpy — mlx not available"
        ) from None
    if not isinstance(arr, mx.array):
        raise TypeError(
            f"_to_numpy expected np.ndarray or mlx.core.array, got {type(arr).__name__}"
        )
    mx.eval(arr)
    # bf16 has no numpy equivalent; numpy raises RuntimeError on the buffer
    # protocol mismatch ("Item size 2 ... format string B item size 1").
    # Cast to f32 first for those dtypes.
    try:
        return np.array(arr, dtype=np.float16)
    except (TypeError, RuntimeError):
        return np.array(arr.astype(mx.float32), dtype=np.float16)


def _touch(p: Path) -> None:
    """Update access time of *p* without modifying content."""
    try:
        now = time.time()
        os.utime(p, (now, p.stat().st_mtime))
    except OSError:
        pass


# ── mlx_lm KV state capture helpers ───────────────────────────────────────────

def infer_kv_dtype(model) -> "Any":
    """Return the model's compute dtype (first floating parameter), or float16.

    KV state is stored on disk as float16 (halves cache size) but must be
    restored in the model's *native* KV dtype (bfloat16 for INT4 MLX builds).
    Restoring as float16 makes attention read a mismatched-dtype cache — both a
    slow mixed-dtype attention path AND, once mlx_lm's KVCache reallocs, an
    ``float16 ⊕ bfloat16 → float32`` promotion that doubles per-step KV
    bandwidth.  Restoring in the native dtype avoids both.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    for _name, p in tree_flatten(model.parameters()):
        if isinstance(p, mx.array) and p.dtype in (mx.float16, mx.bfloat16, mx.float32):
            return p.dtype
    return mx.float16


def capture_kv_state(cache) -> "tuple[list, list, int] | None":
    """Extract (keys, values, offset) from an mlx_lm KV cache object.

    Returns None if the cache type is not supported.
    """
    if cache is None:
        return None
    try:
        # mlx_lm KVCache / PromptCache list
        if isinstance(cache, list) and len(cache) > 0:
            keys   = []
            values = []
            offset = getattr(cache[0], "offset", 0)
            for layer_cache in cache:
                k = getattr(layer_cache, "keys", None)
                v = getattr(layer_cache, "values", None)
                if k is None or v is None:
                    return None
                keys.append(k)
                values.append(v)
            return keys, values, int(offset)
    except Exception:
        pass
    return None


def restore_kv_state(cache, entry: KVCacheEntry, target_dtype=None) -> bool:
    """Write a cached KV entry back into an mlx_lm cache object.

    Returns True on success, False on type mismatch or error.

    v4.2: if ``entry`` was produced by ``get(..., lazy_kv=True)``, the per-layer
    npy files are loaded on demand from ``entry._lazy_kv_dir`` here instead of
    in ``get()``.  Callers that emit a first token from ``last_logit`` can
    defer this call until after the yield so the KV restore (~100 ms) stays
    off the TTFT critical path.

    Parameters
    ----------
    target_dtype : mlx.core.Dtype | None
        When supplied (the server passes the model's compute dtype via
        ``infer_kv_dtype``), each restored K/V array is cast to it.  KV is kept
        as float16 *on disk* but must be restored in the model's native dtype —
        a float16 cache decodes ~1.4x slower (mixed-dtype attention) and
        promotes to float32 on the first realloc.  Defaults to None
        (legacy behaviour: restore in the on-disk float16 dtype).
    """
    if cache is None:
        return False
    try:
        if not isinstance(cache, list):
            return False
        if len(cache) != entry.n_layers:
            logger.warning(
                "KV restore: layer count mismatch (cache=%d entry=%d)",
                len(cache), entry.n_layers,
            )
            return False

        try:
            import mlx.core as mx
            _mx = mx
        except ImportError:
            return False

        # v4.2 lazy-load: populate keys/values from disk now if get(lazy_kv=True)
        lazy_dir = getattr(entry, "_lazy_kv_dir", None)
        if lazy_dir is not None and entry.keys and entry.keys[0] is None:
            try:
                entry.keys = [
                    np.load(str(lazy_dir / f"k_{i}.npy"))
                    for i in range(entry.n_layers)
                ]
                entry.values = [
                    np.load(str(lazy_dir / f"v_{i}.npy"))
                    for i in range(entry.n_layers)
                ]
            except (OSError, ValueError):
                logger.warning("KV restore: lazy-load failed for %s", lazy_dir)
                return False

        for i, layer_cache in enumerate(cache):
            k_mlx = _mx.array(entry.keys[i])
            v_mlx = _mx.array(entry.values[i])
            if target_dtype is not None:
                k_mlx = k_mlx.astype(target_dtype)
                v_mlx = v_mlx.astype(target_dtype)
            # Write into the layer cache's key/value slots
            # mlx_lm KVCache exposes .keys and .values as writable attributes
            # (or via update_and_fetch / direct assignment depending on version).
            if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                layer_cache.keys   = k_mlx
                layer_cache.values = v_mlx
            else:
                return False
            if hasattr(layer_cache, "offset"):
                layer_cache.offset = entry.offset
        return True
    except (AttributeError, RuntimeError, ValueError):
        logger.warning("KV restore failed", exc_info=True)
        return False
