"""Host-agnostic coverage for chunk_prefill via a mocked mlx.core.

The sibling test_chunked_prefill.py importorskips mlx, so it skips on the Linux
coverage runner. This injects a minimal fake mlx.core so the generator body
(chunking math, per-chunk forward, is_final flagging) runs anywhere.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

from squish.streaming.chunked_prefill import ChunkedPrefillConfig, chunk_prefill


class _X2D:
    def __init__(self, seq: int):
        self.shape = (1, seq)


class _X1D:
    def __init__(self, n: int):
        self.n = n

    def __getitem__(self, _key):  # x[None] → batched 2-D view
        return _X2D(self.n)


class _Logits:
    def __getitem__(self, key):  # logits[0, -1] → last-position vector
        return ("LOGIT", key)


def _fake_mlx():
    mx = types.ModuleType("mlx.core")
    mx.int32 = "i32"
    mx.array = lambda data, dtype=None: _X1D(len(data))
    mx.eval = lambda *a: None
    pkg = types.ModuleType("mlx")
    pkg.core = mx
    return pkg, mx


def _model(x, cache=None):
    assert x.shape[0] == 1  # batched
    return _Logits()


def _run(ids, config):
    pkg, mx = _fake_mlx()
    with patch.dict(sys.modules, {"mlx": pkg, "mlx.core": mx}):
        return list(chunk_prefill(_model, ids, layer_caches=object(), config=config))


def test_config_defaults_and_custom():
    assert ChunkedPrefillConfig().chunk_size == 512
    assert ChunkedPrefillConfig().interleave_decode is True
    c = ChunkedPrefillConfig(chunk_size=128, interleave_decode=False)
    assert (c.chunk_size, c.interleave_decode) == (128, False)


def test_single_chunk_is_final_and_default_config():
    # config=None → default; short prompt → exactly one final chunk
    out = _run([1, 2, 3], config=None)
    assert len(out) == 1 and out[0][1] is True


def test_multi_chunk_flags_only_last_final():
    out = _run(list(range(9)), config=ChunkedPrefillConfig(chunk_size=4))
    assert [is_final for _, is_final in out] == [False, False, True]
    assert out[0][0][0] == "LOGIT"  # logit came from the last-position index


def test_cache_forwarded_to_model():
    pkg, mx = _fake_mlx()
    seen = {}

    def model(x, cache=None):
        seen["cache"] = cache
        return _Logits()

    sentinel = object()
    with patch.dict(sys.modules, {"mlx": pkg, "mlx.core": mx}):
        list(chunk_prefill(model, [1, 2], sentinel, ChunkedPrefillConfig(chunk_size=512)))
    assert seen["cache"] is sentinel
