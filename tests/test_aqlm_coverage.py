"""Behavioral coverage for the validation / kmeans / compress_dir paths of
``squish.quant.aqlm`` left untested by the baseline suite. Pure-Python numpy
(sklearn is faked; safetensors is real). No MLX.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from squish.quant import aqlm as aq
from squish.quant.aqlm import (
    AQLMCodebook,
    AQLMConfig,
    AQLMEncoder,
    AQLMLayer,
    _kmeans_fit,
    aqlm_dequantize,
)


def _cfg(n_codebooks=1, codebook_size=4, group_size=4):
    return AQLMConfig(n_codebooks=n_codebooks, codebook_size=codebook_size, group_size=group_size)


def _layer(out=4, in_=8, cfg=None):
    cfg = cfg or _cfg()
    n_groups = in_ // cfg.group_size
    layer = AQLMLayer(out_features=out, in_features=in_, cfg=cfg)
    layer.indices = np.zeros((out, n_groups, cfg.n_codebooks), dtype=np.int32)
    layer.codebooks = [
        AQLMCodebook(vectors=np.ones((cfg.codebook_size, cfg.group_size), np.float32))
        for _ in range(cfg.n_codebooks)
    ]
    layer.scale = 1.0
    return layer


# ── aqlm_dequantize validation ──────────────────────────────────────────────


def test_dequantize_happy_path():
    out = aqlm_dequantize(_layer(out=4, in_=8))
    assert out.shape == (4, 8) and np.all(out == 1.0)  # 1 codebook of ones, scale 1


def test_dequantize_indices_not_3d():
    layer = _layer()
    layer.indices = np.zeros((4, 2), dtype=np.int32)  # 2-D → 168
    with pytest.raises(ValueError, match="must be 3-D"):
        aqlm_dequantize(layer)


def test_dequantize_codebook_count_mismatch():
    layer = _layer()
    layer.indices = np.zeros((4, 2, 3), dtype=np.int32)  # K=3 != n_codebooks=1 → 173
    with pytest.raises(ValueError, match="does not match cfg.n_codebooks"):
        aqlm_dequantize(layer)


def test_dequantize_codebook_list_length_mismatch():
    layer = _layer()
    layer.codebooks = []  # length 0 != K=1 → 177
    with pytest.raises(ValueError, match="codebooks length"):
        aqlm_dequantize(layer)


def test_dequantize_codebook_vector_shape_mismatch():
    layer = _layer()
    layer.codebooks = [AQLMCodebook(vectors=np.ones((2, 2), np.float32))]  # wrong shape → 187
    with pytest.raises(ValueError, match="does not match"):
        aqlm_dequantize(layer)


# ── _kmeans_fit ──────────────────────────────────────────────────────────────


def test_kmeans_few_samples_pads():
    data = np.array([[1.0, 2.0]], dtype=np.float32)  # 1 sample, 4 clusters
    centres = _kmeans_fit(data, n_clusters=4, seed=0, max_iter=10)
    assert centres.shape == (4, 2)
    assert np.all(centres[1:] == 0.0)  # padded rows


def test_kmeans_sklearn_fast_path(monkeypatch):
    captured = {}

    class _MBK:
        def __init__(self, n_clusters, **kw):
            captured["n"] = n_clusters

        def fit(self, data):
            self.cluster_centers_ = np.zeros((captured["n"], data.shape[1]), np.float32)
            return self

    fake_cluster = types.ModuleType("sklearn.cluster")
    fake_cluster.MiniBatchKMeans = _MBK
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_cluster)
    data = np.random.default_rng(0).standard_normal((50, 4)).astype(np.float32)
    centres = _kmeans_fit(data, n_clusters=3, seed=0, max_iter=5)
    assert centres.shape == (3, 4) and captured["n"] == 3  # 228-236


def test_kmeans_numpy_fallback_converges(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)  # ImportError → numpy path
    # Two tight, well-separated clusters → Lloyd converges and breaks early.
    data = np.array([[0.0, 0.0]] * 10 + [[10.0, 10.0]] * 10, dtype=np.float32)
    centres = _kmeans_fit(data, n_clusters=2, seed=1, max_iter=50)
    assert centres.shape == (2, 2)
    sorted_x = sorted(centres[:, 0])
    assert sorted_x[0] < 1.0 and sorted_x[1] > 9.0


def test_kmeans_numpy_fallback_max_iter_exhausted(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    data = np.random.default_rng(2).standard_normal((40, 3)).astype(np.float32)
    # max_iter=1 → the loop runs once and exits by exhaustion (246→267).
    centres = _kmeans_fit(data, n_clusters=4, seed=2, max_iter=1)
    assert centres.shape == (4, 3)


# ── encode_layer / compress_dir ──────────────────────────────────────────────


def test_encode_layer_roundtrip():
    enc = AQLMEncoder(_cfg(codebook_size=4, group_size=4), seed=0, max_iter=3)
    weight = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
    layer = enc.encode_layer(weight)
    recon = aqlm_dequantize(layer)
    assert recon.shape == (8, 8)


def test_compress_dir(tmp_path, monkeypatch, capsys):
    import safetensors.numpy as stn
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)  # deterministic numpy kmeans
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    stn.save_file({
        "layer.0.mlp.gate_proj.weight": np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32),
        "layer.0.attn.q_proj.weight": np.random.default_rng(1).standard_normal((8, 6)).astype(np.float32),
        "norm.weight": np.ones((2,), np.float32),  # 1-D → passthrough (not encoded)
    }, str(model_dir / "model.safetensors"))

    enc = AQLMEncoder(_cfg(codebook_size=4, group_size=4), seed=0, max_iter=2, min_out_features=4)
    out_dir = tmp_path / "out"
    enc.compress_dir(model_dir, out_dir, progress=True)
    printed = capsys.readouterr().out
    assert "[aqlm]" in printed         # 522 — encoded the divisible-shape weight
    assert "[skip]" in printed         # 513 — skipped in_features-not-divisible weight
    assert (out_dir / "layer_0_mlp_gate_proj_weight__aqlm_idx.npy").exists()


def test_compress_dir_no_safetensors(tmp_path):
    enc = AQLMEncoder(_cfg(), seed=0, max_iter=2)
    with pytest.raises(FileNotFoundError, match="No .*safetensors"):
        enc.compress_dir(tmp_path, tmp_path / "out", progress=False)
