"""Behavioral coverage for the lazy-KV fast path, write/lock handling, LRU
eviction, error handlers, and mlx↔numpy capture/restore helpers of
``squish.kv.prompt_kv_cache`` left untested by the baseline suite.

MLX-only paths are guarded with ``pytest.importorskip("mlx.core")`` so they run
on the macOS + MLX coverage runner and skip cleanly on Linux. mlx-absent
branches are forced with ``monkeypatch.setitem(sys.modules, "mlx.core", None)``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from squish.kv import prompt_kv_cache as pkv
from squish.kv.prompt_kv_cache import (
    KVCacheEntry,
    PromptKVStore,
    _to_numpy,
    _touch,
    capture_kv_state,
    infer_kv_dtype,
    restore_kv_state,
)


def _make_kv(n_layers=2, seq=4, heads=2, head_dim=8):
    shape = (1, heads, seq, head_dim)
    keys = [np.ones(shape, np.float16) * i for i in range(n_layers)]
    values = [np.ones(shape, np.float16) * (i + 10) for i in range(n_layers)]
    return keys, values


def _store(tmp_path, **kw):
    return PromptKVStore(cache_dir=tmp_path, **kw)


# ── get(): last_logit + lazy_kv ─────────────────────────────────────────────


def test_get_loads_last_logit(tmp_path):
    store = _store(tmp_path)
    k, v = _make_kv()
    logit = np.arange(7, dtype=np.float32)
    store.put("p", k, v, offset=4, last_logit=logit)
    entry = store.get("p")
    assert entry is not None and entry.last_logit is not None
    np.testing.assert_array_equal(entry.last_logit, logit)


def test_get_tolerates_corrupt_last_logit(tmp_path):
    store = _store(tmp_path)
    k, v = _make_kv()
    store.put("p", k, v, offset=4, last_logit=np.arange(7, dtype=np.float32))
    # Corrupt only the logit file → entry still returned, logit dropped (162-163).
    (store._entry_dir(store.hash_prompt("p")) / "last_logit.npy").write_bytes(b"junk")
    entry = store.get("p")
    assert entry is not None and entry.last_logit is None


def test_get_lazy_kv_defers_array_load(tmp_path):
    store = _store(tmp_path)
    k, v = _make_kv(n_layers=3)
    store.put("p", k, v, offset=4)
    entry = store.get("p", lazy_kv=True)
    assert entry is not None
    # Arrays are placeholders (169-170) and the lazy dir is stashed (195).
    assert entry.keys == [None, None, None]
    assert entry.values == [None, None, None]
    assert entry._lazy_kv_dir == store._entry_dir(store.hash_prompt("p"))


# ── put(): validation, lock, logit save, unlink ─────────────────────────────


def test_put_length_mismatch_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="same length"):
        store.put("p", [np.ones((1, 1, 1, 1), np.float16)], [], offset=0)


def test_put_skips_when_lock_already_held(tmp_path):
    store = _store(tmp_path)
    h = store.hash_prompt("p")
    d = store._entry_dir(h)
    d.mkdir(parents=True, exist_ok=True)
    (d / ".lock").touch()  # pre-existing lock → writer bails out (239-240)
    k, v = _make_kv()
    store.put("p", k, v, offset=4)
    assert not (d / "meta.json").exists()  # nothing written


def test_put_with_bad_logit_stores_without_it(tmp_path, monkeypatch):
    store = _store(tmp_path)
    k, v = _make_kv()
    # Make only the logit conversion fail (258-259); KV save still succeeds.
    real = pkv._to_numpy

    def flaky(arr):
        if arr is SENTINEL:
            raise ValueError("bad logit")
        return real(arr)

    SENTINEL = object()
    monkeypatch.setattr(pkv, "_to_numpy", flaky)
    store.put("p", k, v, offset=4, last_logit=SENTINEL)
    entry = store.get("p")
    assert entry is not None and entry.last_logit is None
    meta = json.loads((store._entry_dir(store.hash_prompt("p")) / "meta.json").read_text())
    assert meta["has_logit"] is False


def test_put_lock_unlink_missing_is_swallowed(tmp_path, monkeypatch):
    store = _store(tmp_path)
    k, v = _make_kv()
    real_unlink = Path.unlink

    def unlink_gone(self, *a, **k):
        if self.name == ".lock":
            raise FileNotFoundError(self)
        return real_unlink(self, *a, **k)

    # Lock vanishing before cleanup must be swallowed (280-281).
    monkeypatch.setattr(Path, "unlink", unlink_gone)
    store.put("p", k, v, offset=4)
    assert store.get("p") is not None


def test_put_triggers_random_eviction(tmp_path, monkeypatch):
    store = _store(tmp_path)
    calls = {"n": 0}
    monkeypatch.setattr(store, "_evict_lru", lambda: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr("random.random", lambda: 0.0)  # always < 0.05 (line 285)
    k, v = _make_kv()
    store.put("p", k, v, offset=4)
    assert calls["n"] == 1


# ── clear / total_bytes / entry_count ───────────────────────────────────────


def test_clear_skips_non_dir_entries(tmp_path):
    store = _store(tmp_path)
    k, v = _make_kv()
    store.put("p", k, v, offset=4)
    (tmp_path / "stray.txt").write_text("x")  # non-dir entry skipped (303→302)
    removed = store.clear()
    assert removed == 1
    assert (tmp_path / "stray.txt").exists()


def test_total_bytes_swallows_stat_errors(tmp_path, monkeypatch):
    store = _store(tmp_path)
    k, v = _make_kv()
    store.put("p", k, v, offset=4)
    files = list(store._entry_dir(store.hash_prompt("p")).glob("*"))
    # Feed a fixed file list (avoids rglob's own stat), force is_file True, and
    # make the size read raise → the OSError is swallowed (316-317), total == 0.
    monkeypatch.setattr(Path, "rglob", lambda self, pat: iter(files))
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(
        Path, "stat", lambda self, *a, **k: (_ for _ in ()).throw(OSError("stat fail")),
    )
    assert store.total_bytes() == 0


# ── LRU eviction ────────────────────────────────────────────────────────────


def test_evict_lru_noop_under_budget(tmp_path):
    store = _store(tmp_path, max_bytes=10**12)
    k, v = _make_kv()
    store.put("p", k, v, offset=4)
    assert store._evict_lru() == 0  # under budget early return (329-330)


def test_evict_lru_removes_oldest_until_under_budget(tmp_path):
    store = _store(tmp_path, max_bytes=10**12)
    for i in range(3):
        k, v = _make_kv()
        store.put(f"prompt-{i}", k, v, offset=4)
    # Age the entries so order is deterministic (oldest = prompt-0).
    for i in range(3):
        meta = store._entry_dir(store.hash_prompt(f"prompt-{i}")) / "meta.json"
        os.utime(meta, (i + 1, i + 1))
    one = store.total_bytes() // 3
    store._max_bytes = one * 2  # room for ~2 entries → evict the oldest
    evicted = store._evict_lru()
    assert evicted >= 1
    assert store.get("prompt-0") is None  # oldest gone
    assert store.entry_count() < 3


def test_evict_lru_skips_nondir_and_metaless(tmp_path):
    # Put under a generous budget so put()'s 1-in-20 random eviction can't wipe
    # the entry, then lower the budget to force eviction deterministically.
    store = _store(tmp_path, max_bytes=10**12)
    k, v = _make_kv()
    store.put("real", k, v, offset=4)
    (tmp_path / "loose.txt").write_text("x")  # non-dir → skip (335)
    (tmp_path / "emptydir").mkdir()           # dir w/o meta → skip (338)
    store._max_bytes = 0
    evicted = store._evict_lru()
    assert evicted == 1  # only the real entry was eligible and removed
    assert store.get("real") is None


def test_evict_lru_swallows_atime_stat_error(tmp_path, monkeypatch):
    # Generous budget during put() (avoids the random-eviction race), then 0.
    store = _store(tmp_path, max_bytes=10**12)
    k, v = _make_kv()
    store.put("real", k, v, offset=4)
    store._max_bytes = 0
    # Bypass total_bytes (keeps its own stats clean) and force the per-entry
    # atime read to fail so the 0.0 fallback runs (341-342). 1 > max_bytes(0)
    # so the early-return budget check is passed.
    monkeypatch.setattr(store, "total_bytes", lambda: 1)
    real_stat = Path.stat
    seen: dict[str, int] = {}

    def flaky_stat(self, *a, **k):
        if self.name == "meta.json":
            seen[str(self)] = seen.get(str(self), 0) + 1
            if seen[str(self)] >= 2:  # let exists() pass, fail the atime read
                raise OSError("no atime")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert store._evict_lru() == 1  # entry still evicted despite atime failure


def test_evict_lru_no_eligible_entries_returns_zero(tmp_path):
    store = _store(tmp_path, max_bytes=0)
    (tmp_path / "loose.txt").write_text("x")  # over budget but no dir entries
    # Passes the budget check, finds nothing to evict → evicted stays 0 (354→356).
    assert store._evict_lru() == 0


def test_remove_entry_swallows_rmtree_errors(tmp_path, monkeypatch):
    store = _store(tmp_path)
    k, v = _make_kv()
    store.put("p", k, v, offset=4)
    d = store._entry_dir(store.hash_prompt("p"))
    monkeypatch.setattr("shutil.rmtree",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("locked")))
    # rmtree failing is logged and swallowed (367-368) — invalidate returns True.
    assert store.invalidate("p") is True
    assert d.exists()  # not actually removed


# ── _to_numpy / _touch helpers ──────────────────────────────────────────────


def test_to_numpy_numpy_passthrough():
    out = _to_numpy(np.ones((2, 3), np.float32))
    assert out.dtype == np.float16 and out.shape == (2, 3)


def test_to_numpy_rejects_non_array_without_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(TypeError, match="mlx not available"):
        _to_numpy([1, 2, 3])  # non-ndarray, mlx import fails (389-390)


def test_touch_swallows_errors(tmp_path, monkeypatch):
    p = tmp_path / "x"
    p.write_text("hi")
    monkeypatch.setattr(os, "utime", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    _touch(p)  # OSError swallowed (412) — no raise


# ── capture_kv_state ────────────────────────────────────────────────────────


class _Layer:
    def __init__(self, keys, values, offset=0):
        self.keys = keys
        self.values = values
        self.offset = offset


def test_capture_kv_state_none_and_empty():
    assert capture_kv_state(None) is None
    assert capture_kv_state([]) is None  # empty list → falls through to None


def test_capture_kv_state_extracts_layers():
    k0, v0 = np.ones((1, 2, 4, 8), np.float16), np.zeros((1, 2, 4, 8), np.float16)
    cache = [_Layer(k0, v0, offset=4), _Layer(k0, v0, offset=4)]
    out = capture_kv_state(cache)
    assert out is not None
    keys, values, offset = out
    assert offset == 4 and len(keys) == 2


def test_capture_kv_state_missing_kv_returns_none():
    cache = [_Layer(None, None)]  # k/v None → None (453)
    assert capture_kv_state(cache) is None


def test_capture_kv_state_swallows_conversion_error():
    # A non-numeric offset makes int(offset) raise ValueError → caught (457-458).
    cache = [_Layer(np.ones((1, 1, 1, 1), np.float16),
                    np.ones((1, 1, 1, 1), np.float16), offset="not-an-int")]
    assert capture_kv_state(cache) is None


# ── MLX-only: _to_numpy mlx branch, infer_kv_dtype, restore_kv_state ─────────


def test_to_numpy_mlx_and_bf16_branches():
    mx = pytest.importorskip("mlx.core")
    out = _to_numpy(mx.ones((2, 2), dtype=mx.float32))
    assert out.dtype == np.float16
    out_bf = _to_numpy(mx.ones((2, 2), dtype=mx.bfloat16))  # bf16 fallback (403-404)
    assert out_bf.dtype == np.float16


def test_to_numpy_rejects_wrong_mlx_type():
    pytest.importorskip("mlx.core")
    with pytest.raises(TypeError, match="expected np.ndarray or mlx"):
        _to_numpy({"not": "an array"})  # non-ndarray, mlx present (393-396)


def test_infer_kv_dtype_returns_param_dtype():
    mx = pytest.importorskip("mlx.core")

    class _Model:
        def parameters(self):
            return {"w": mx.ones((2, 2), dtype=mx.bfloat16)}

    assert infer_kv_dtype(_Model()) == mx.bfloat16


def test_infer_kv_dtype_defaults_to_float16_when_no_float_param():
    mx = pytest.importorskip("mlx.core")

    class _Model:
        def parameters(self):
            return {"i": mx.ones((2,), dtype=mx.int32)}

    assert infer_kv_dtype(_Model()) == mx.float16


def test_restore_kv_state_success_and_dtype_cast():
    mx = pytest.importorskip("mlx.core")
    k, v = _make_kv(n_layers=2)
    entry = KVCacheEntry(prompt_hash="h", n_layers=2, offset=4, keys=k, values=v)
    cache = [_Layer(None, None), _Layer(None, None)]
    assert restore_kv_state(cache, entry, target_dtype=mx.float16) is True
    for layer in cache:
        assert layer.keys.dtype == mx.float16
        assert layer.offset == 4


def test_restore_kv_state_lazy_load_from_disk(tmp_path):
    mx = pytest.importorskip("mlx.core")
    store = _store(tmp_path)
    k, v = _make_kv(n_layers=2)
    store.put("p", k, v, offset=4)
    entry = store.get("p", lazy_kv=True)  # keys are placeholders
    cache = [_Layer(None, None), _Layer(None, None)]
    # restore triggers the on-demand npy load (503-512).
    assert restore_kv_state(cache, entry) is True
    assert cache[0].keys is not None


def test_restore_kv_state_lazy_load_failure_returns_false(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")
    store = _store(tmp_path)
    k, v = _make_kv(n_layers=2)
    store.put("p", k, v, offset=4)
    entry = store.get("p", lazy_kv=True)
    monkeypatch.setattr(pkv.np, "load",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    cache = [_Layer(None, None), _Layer(None, None)]
    assert restore_kv_state(cache, entry) is False  # lazy-load failure (513-515)


def test_restore_kv_state_rejects_non_list_and_none():
    pytest.importorskip("mlx.core")
    entry = KVCacheEntry(prompt_hash="h", n_layers=1, offset=0,
                         keys=[np.ones((1, 1, 1, 1), np.float16)],
                         values=[np.ones((1, 1, 1, 1), np.float16)])
    assert restore_kv_state(None, entry) is False          # 483-484
    assert restore_kv_state("not-a-list", entry) is False  # 486-487


def test_restore_kv_state_layer_mismatch_returns_false():
    pytest.importorskip("mlx.core")
    k, v = _make_kv(n_layers=2)
    entry = KVCacheEntry(prompt_hash="h", n_layers=2, offset=4, keys=k, values=v)
    assert restore_kv_state([_Layer(None, None)], entry) is False  # 488-493


def test_restore_kv_state_unsupported_layer_returns_false():
    pytest.importorskip("mlx.core")
    k, v = _make_kv(n_layers=1)
    entry = KVCacheEntry(prompt_hash="h", n_layers=1, offset=4, keys=k, values=v)
    # A layer lacking .keys/.values → False (529-530).
    assert restore_kv_state([object()], entry) is False


def test_restore_kv_state_returns_false_without_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    k, v = _make_kv(n_layers=1)
    entry = KVCacheEntry(prompt_hash="h", n_layers=1, offset=4, keys=k, values=v)
    # mlx import fails inside restore → False (498-499).
    assert restore_kv_state([_Layer(None, None)], entry) is False


def test_restore_kv_state_layer_without_offset_attr():
    pytest.importorskip("mlx.core")

    class _LayerNoOffset:
        keys = None
        values = None
        # has keys/values but no `offset` → the offset assignment is skipped (531→517)

    k, v = _make_kv(n_layers=1)
    entry = KVCacheEntry(prompt_hash="h", n_layers=1, offset=4, keys=k, values=v)
    cache = [_LayerNoOffset()]
    assert restore_kv_state(cache, entry) is True
    assert cache[0].keys is not None


def test_restore_kv_state_swallows_runtime_error():
    pytest.importorskip("mlx.core")

    class _BadSetLayer:
        keys = None
        values = None
        offset = 0

        def __setattr__(self, name, val):
            if name == "keys":
                raise RuntimeError("write boom")
            object.__setattr__(self, name, val)

    k, v = _make_kv(n_layers=1)
    entry = KVCacheEntry(prompt_hash="h", n_layers=1, offset=4, keys=k, values=v)
    # The write raising is caught by the outer handler → False (534-536).
    assert restore_kv_state([_BadSetLayer()], entry) is False
