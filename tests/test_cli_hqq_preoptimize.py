"""Coverage for cli._preoptimize_weights_with_hqq (cli.py lines ~4534-4593).

The HQQ quantizer itself runs natively on Linux (numpy-only); only ``mlx.core``
is faked so the load/save/array round-trip can be exercised without Apple
Silicon.  Both the happy path and every guard (missing mlx, missing source,
no shards, mid-pass failure) are covered deterministically.
"""

import sys
import types

import numpy as np
import pytest

from squish import cli


# ── fake mlx.core ───────────────────────────────────────────────────────────────


class _FakeArray:
    """Minimal mx.array stand-in backed by a numpy buffer."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def dtype(self):
        return self._arr.dtype

    def astype(self, dt):
        return _FakeArray(self._arr.astype(dt))

    def __array__(self, dtype=None, copy=None):
        if dtype is not None:
            return self._arr.astype(dtype)
        return self._arr


def _install_fake_mlx(monkeypatch, *, weights, saver=None):
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = np.float32
    fake_mx.load = lambda _path: weights
    fake_mx.array = _FakeArray
    fake_mx.save_safetensors = saver or (lambda _path, _w: None)
    pkg = types.ModuleType("mlx")
    pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    return fake_mx


def _make_source(tmp_path):
    """A source dir with one shard plus non-shard file + dir (copy branches)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    sub = src / "tokenizer"
    sub.mkdir()
    (sub / "vocab.txt").write_text("hi")
    (src / "model.safetensors").write_bytes(b"")  # content ignored — mx.load is faked
    return src


# ── happy path ──────────────────────────────────────────────────────────────────


def test_preoptimize_happy_path(tmp_path, monkeypatch):
    src = _make_source(tmp_path)
    saved = {}
    weights = {
        # FFN 2-D .weight → optimised
        "model.layers.0.mlp.gate_proj.weight": _FakeArray(
            np.random.randn(64, 128).astype(np.float16)
        ),
        "model.layers.0.mlp.down_proj.weight": _FakeArray(
            np.random.randn(128, 64).astype(np.float16)
        ),
        # FFN but 1-D → skipped
        "model.layers.0.mlp.up_proj.bias": _FakeArray(np.zeros(64, np.float16)),
        # non-FFN weight → skipped
        "model.layers.0.self_attn.q_proj.weight": _FakeArray(
            np.random.randn(64, 64).astype(np.float16)
        ),
    }
    _install_fake_mlx(monkeypatch, weights=weights, saver=lambda p, w: saved.__setitem__(p, w))

    out = cli._preoptimize_weights_with_hqq(src, ffn_bits=3, group_size=64, max_iter=3)

    assert out.exists()
    # non-shard file + dir were copied across
    assert (out / "config.json").exists()
    assert (out / "tokenizer" / "vocab.txt").exists()
    # exactly the two 2-D FFN weights were optimised (replaced by fresh _FakeArray)
    assert len(saved) == 1
    written = next(iter(saved.values()))
    assert written["model.layers.0.mlp.gate_proj.weight"]._arr.dtype == np.float16
    # the skipped tensors are passed through unchanged (same object)
    assert (
        written["model.layers.0.self_attn.q_proj.weight"]
        is (weights["model.layers.0.self_attn.q_proj.weight"])
    )


# ── guard paths ─────────────────────────────────────────────────────────────────


def test_preoptimize_requires_mlx(tmp_path, monkeypatch):
    src = _make_source(tmp_path)
    # Force ImportError for mlx on every host (incl. Apple runners).
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(SystemExit):
        cli._preoptimize_weights_with_hqq(src, ffn_bits=3, group_size=64)


def test_preoptimize_missing_source(tmp_path, monkeypatch):
    _install_fake_mlx(monkeypatch, weights={})
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit):
        cli._preoptimize_weights_with_hqq(missing, ffn_bits=3, group_size=64)


def test_preoptimize_no_shards(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")  # no .safetensors
    _install_fake_mlx(monkeypatch, weights={})
    with pytest.raises(SystemExit):
        cli._preoptimize_weights_with_hqq(src, ffn_bits=3, group_size=64)


def test_preoptimize_cleans_up_on_failure(tmp_path, monkeypatch):
    src = _make_source(tmp_path)

    def _boom(_path, _w):
        raise ValueError("disk full")

    weights = {
        "model.layers.0.mlp.gate_proj.weight": _FakeArray(
            np.random.randn(32, 64).astype(np.float16)
        ),
    }
    _install_fake_mlx(monkeypatch, weights=weights, saver=_boom)
    with pytest.raises(ValueError, match="disk full"):
        cli._preoptimize_weights_with_hqq(src, ffn_bits=3, group_size=64, max_iter=3)
