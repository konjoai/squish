"""Regression test for the quantized-KV mx.compile incompatibility.

Numpy-quantized KV caches (KIVI int8 / snap) eval inside their per-step update,
which is illegal under mx.compile.  The server must NOT wrap the decode forward
in mx.compile when such a cache is attached, otherwise the first decode step
raises "[eval] Attempting to eval an array during ... compile" and the request
degrades to a slow stream_generate fallback.

``_kv_cache_compile_safe`` is the gate that decides this.  These tests pin its
contract so the bug cannot silently regress.
"""
from squish.server import _kv_cache_compile_safe


def test_none_cache_is_compile_safe():
    # No quantized cache attached → plain fp16 path → compile is fine.
    assert _kv_cache_compile_safe(None) is True


def test_numpy_quantized_cache_is_not_compile_safe():
    # A KIVI-style cache that does not advertise compile_safe must be treated as
    # unsafe (the real QuantizedKVCache has no compile_safe attribute).
    class _FakeQuantCache:
        mode = "int8"

    assert _kv_cache_compile_safe(_FakeQuantCache()) is False


def test_explicitly_safe_cache_is_compile_safe():
    # A future compile-safe (e.g. MLX-native quant) cache may opt back in.
    class _SafeCache:
        compile_safe = True

    assert _kv_cache_compile_safe(_SafeCache()) is True


def test_falsey_compile_safe_attr_is_not_safe():
    class _Cache:
        compile_safe = False

    assert _kv_cache_compile_safe(_Cache()) is False
