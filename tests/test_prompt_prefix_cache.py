"""Unit + integration tests for the in-memory prompt-prefix KV reuse slot.

The pure borrow/store/prefix-match logic is covered without a model; the
byte-identity of reuse vs cold prefill is covered by a model-gated test.
"""

import os

import pytest

from squish.kv.prompt_prefix_cache import PromptPrefixCache, _common_prefix_len

_MODEL = os.environ.get("PL_TEST_MODEL", "") or os.path.expanduser(
    "~/models/Qwen2.5-1.5B-Instruct-int4"
)
_needs_model = pytest.mark.skipif(not os.path.isdir(_MODEL), reason=f"model not present: {_MODEL}")


def test_common_prefix_len():
    assert _common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert _common_prefix_len([1, 2, 3], [1, 2, 3, 4]) == 3  # one is a prefix
    assert _common_prefix_len([5, 6], [7, 8]) == 0
    assert _common_prefix_len([], [1]) == 0


def test_borrow_requires_min_prefix():
    pc = PromptPrefixCache(min_prefix=4)
    pc.store([1, 2, 3, 4, 5], object())
    # only 3 shared tokens (< min_prefix=4) → no reuse
    assert pc.borrow([1, 2, 3, 9, 9]) == (None, 0)


def test_borrow_returns_cache_and_shared_len_then_empties_slot():
    pc = PromptPrefixCache(min_prefix=2)
    sentinel = object()
    pc.store([1, 2, 3, 4], sentinel)
    cache, shared = pc.borrow([1, 2, 3, 9])
    assert cache is sentinel and shared == 3
    # slot was handed off — a second borrow finds nothing
    assert pc.borrow([1, 2, 3, 9]) == (None, 0)


def test_borrow_empty_slot_is_none():
    assert PromptPrefixCache().borrow([1, 2, 3]) == (None, 0)


def test_store_then_borrow_roundtrip():
    pc = PromptPrefixCache(min_prefix=1)
    a = object()
    pc.store([10, 11], a)
    assert pc.borrow([10, 11, 12]) == (a, 2)


def test_reuse_safe_only_for_plain_kvcache():
    """Reuse must be allowed for plain KVCache and disabled for windowed/hybrid
    caches (RotatingKVCache) whose rolling window would drop prefix tokens."""
    pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from squish.kv.prompt_prefix_cache import _caches_reuse_safe

    assert _caches_reuse_safe([KVCache(), KVCache()]) is True
    assert _caches_reuse_safe([KVCache(), RotatingKVCache(max_size=8, keep=2)]) is False
    assert _caches_reuse_safe([]) is False


def test_borrow_returns_none_when_slot_locked():
    """A concurrent holder of the lock must not block a borrow — it falls back to
    a cold prefill by returning ``(None, 0)`` immediately."""
    pc = PromptPrefixCache(min_prefix=1)
    pc.store([1, 2, 3], object())
    assert pc._lock.acquire(blocking=False) is True
    try:
        assert pc.borrow([1, 2, 3, 4]) == (None, 0)
    finally:
        pc._lock.release()


def test_store_is_dropped_when_slot_locked():
    """If another request owns the slot, ``store`` drops our cache rather than
    block; the slot keeps whatever it held before."""
    pc = PromptPrefixCache(min_prefix=1)
    keeper = object()
    pc.store([1, 2], keeper)
    assert pc._lock.acquire(blocking=False) is True
    try:
        pc.store([9, 9, 9], object())  # contended → no-op
    finally:
        pc._lock.release()
    # original entry survived the dropped store
    assert pc.borrow([1, 2, 7]) == (keeper, 2)


def test_default_prefix_cache_is_process_singleton():
    from squish.kv.prompt_prefix_cache import default_prefix_cache

    a = default_prefix_cache()
    b = default_prefix_cache()
    assert a is b
    assert isinstance(a, PromptPrefixCache)


def _install_fake_mlx_lm_cache(monkeypatch, make_prompt_cache):
    """Inject a minimal fake ``mlx_lm.models.cache`` so the lazy imports inside
    prompt_prefix_cache resolve without Apple Silicon. Returns the fake KVCache
    type so tests can build reuse-safe probes."""
    import sys
    import types

    fake_cache = types.ModuleType("mlx_lm.models.cache")

    class KVCache:  # plain cache → reuse-safe
        pass

    fake_cache.KVCache = KVCache
    fake_cache.make_prompt_cache = make_prompt_cache
    fake_models = types.ModuleType("mlx_lm.models")
    fake_models.cache = fake_cache
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.models = fake_models
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)
    return KVCache


def test_caches_reuse_safe_with_fake_mlx(monkeypatch):
    from squish.kv.prompt_prefix_cache import _caches_reuse_safe

    KVCache = _install_fake_mlx_lm_cache(monkeypatch, make_prompt_cache=lambda m: [])

    class Windowed:
        pass

    assert _caches_reuse_safe([KVCache(), KVCache()]) is True
    assert _caches_reuse_safe([KVCache(), Windowed()]) is False
    assert _caches_reuse_safe([]) is False


def test_reuse_safe_true_false_and_memoized(monkeypatch):
    from squish.kv import prompt_prefix_cache as ppc

    ppc._reuse_safe_by_model.clear()

    class Windowed:
        pass

    calls = {"n": 0}

    def make_prompt_cache(model):
        calls["n"] += 1
        return model._probe

    KVCache = _install_fake_mlx_lm_cache(monkeypatch, make_prompt_cache)

    safe_model = type("M", (), {})()
    safe_model._probe = [KVCache(), KVCache()]
    assert ppc.reuse_safe(safe_model) is True
    # second call is memoized by id(model) — make_prompt_cache not called again
    assert ppc.reuse_safe(safe_model) is True
    assert calls["n"] == 1

    unsafe_model = type("M", (), {})()
    unsafe_model._probe = [KVCache(), Windowed()]
    assert ppc.reuse_safe(unsafe_model) is False  # exercises the warning branch
    ppc._reuse_safe_by_model.clear()


@_needs_model
def test_reuse_is_byte_identical_to_cold():
    """A prompt that extends a recent one reuses its prefix and yields the exact
    same tokens as a cold (full-prefill) run — reuse changes only how much
    prefill runs, never the logits."""
    import mlx.core as mx

    mx.set_default_device(mx.gpu)  # warm/cold share forward math → identical here
    from mlx_lm import load

    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate

    model, tok = load(_MODEL)
    eos = {tok.eos_token_id}

    def gen(ids, n, reuse):
        return [
            t for t, _ in prompt_lookup_generate(model, ids, n, eos_ids=eos, reuse_prefix=reuse)
        ]

    # >128 tokens so the prefix-reuse threshold engages on turn 2.
    p1 = tok.encode("Background: " + "the river flows past the old stone bridge. " * 14)
    g1 = gen(p1, 16, reuse=True)  # turn 1 populates the slot
    p2 = p1 + g1 + tok.encode(" Then answer briefly:")  # turn 2 extends turn 1

    warm = gen(p2, 16, reuse=True)  # reuses the shared prefix
    cold = gen(p2, 16, reuse=False)  # full cold prefill (truth)
    assert warm == cold


def _install_fake_mlx_runtime(monkeypatch):
    """Inject fake ``mlx.core`` + ``mlx_lm.models.cache`` so prefill_with_reuse
    runs without Apple Silicon. Returns (calls dict) recording model/trim use."""
    import sys
    import types

    calls = {"model": [], "trim": [], "eval": 0}

    class _Arr:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, _key):
            return ("BATCHED", self.data)

    mx = types.ModuleType("mlx.core")
    mx.uint32 = "u32"
    mx.array = lambda data, dtype=None: _Arr(data)
    mx.eval = lambda *a: calls.__setitem__("eval", calls["eval"] + 1)
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    cache_mod = types.ModuleType("mlx_lm.models.cache")

    class KVCache:
        pass

    cache_mod.KVCache = KVCache
    cache_mod.make_prompt_cache = lambda model: [types.SimpleNamespace(offset=0, state=())]
    cache_mod.trim_prompt_cache = lambda cache, drop: calls["trim"].append(drop)
    models = types.ModuleType("mlx_lm.models")
    models.cache = cache_mod
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    return calls


def test_prefill_cold_when_no_prefix_cache(monkeypatch):
    from squish.kv.prompt_prefix_cache import prefill_with_reuse

    calls = _install_fake_mlx_runtime(monkeypatch)
    model = lambda x, cache=None: calls["model"].append(x) or "out"  # noqa: E731
    cache = prefill_with_reuse(model, [1, 2, 3, 4], None)
    assert isinstance(cache, list)
    assert calls["model"] and calls["eval"] == 1  # suffix prefilled + evaluated


def test_prefill_reuses_shared_prefix_and_trims(monkeypatch):
    import types

    from squish.kv.prompt_prefix_cache import PromptPrefixCache, prefill_with_reuse

    calls = _install_fake_mlx_runtime(monkeypatch)
    pc = PromptPrefixCache(min_prefix=2)
    # stored cache covers more tokens (offset=4) than the shared prefix (3) → trim.
    pc.store([1, 2, 3, 9], [types.SimpleNamespace(offset=4, state=())])
    model = lambda x, cache=None: "out"  # noqa: E731
    prefill_with_reuse(model, [1, 2, 3, 7, 8], pc)
    assert calls["trim"] == [1]  # offset(4) - keep(min(3,4,4)=3) == 1


def test_prefill_reuse_with_no_suffix_skips_model(monkeypatch):
    import types

    from squish.kv.prompt_prefix_cache import PromptPrefixCache, prefill_with_reuse

    calls = _install_fake_mlx_runtime(monkeypatch)
    pc = PromptPrefixCache(min_prefix=2)
    pc.store([1, 2, 3], [types.SimpleNamespace(offset=2, state=())])
    model = lambda x, cache=None: calls["model"].append(x)  # noqa: E731
    # prompt_ids[:-1] == [1,2]; keep=min(2,2,2)=2 → start=2 → suffix empty
    prefill_with_reuse(model, [1, 2, 3], pc)
    assert calls["model"] == [] and calls["eval"] == 0
