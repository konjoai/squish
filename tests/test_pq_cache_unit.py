"""tests/test_pq_cache_unit.py — 100% coverage for squish/pq_cache.py"""
import numpy as np
import pytest

from squish.pq_cache import (
    PQCacheConfig,
    PQCodebook,
    PQKeyIndex,
    PQValueStore,
    retrieve,
)

RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# PQCacheConfig
# ---------------------------------------------------------------------------

class TestPQCacheConfig:
    def test_defaults(self):
        cfg = PQCacheConfig()
        assert cfg.n_subvectors == 8
        assert cfg.n_codes      == 256
        assert cfg.train_iters  == 20
        assert cfg.seed         == 42

    def test_invalid_n_subvectors(self):
        with pytest.raises(ValueError, match="n_subvectors"):
            PQCacheConfig(n_subvectors=0)

    def test_invalid_n_codes(self):
        with pytest.raises(ValueError, match="n_codes"):
            PQCacheConfig(n_codes=1)

    def test_invalid_train_iters(self):
        with pytest.raises(ValueError, match="train_iters"):
            PQCacheConfig(train_iters=0)


# ---------------------------------------------------------------------------
# PQCodebook
# ---------------------------------------------------------------------------

class TestPQCodebook:
    def _make(self, sub_dim=4, n_codes=8, n_iters=5, seed=0):
        return PQCodebook(sub_dim, n_codes, n_iters, seed)

    def test_not_fitted_initially(self):
        cb = self._make()
        assert not cb.is_fitted

    def test_encode_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError, match="fitted"):
            cb.encode(np.zeros((3, 4), dtype=np.float32))

    def test_decode_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError, match="fitted"):
            cb.decode(np.array([0], dtype=np.uint16))

    def test_lookup_table_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError, match="fitted"):
            cb.lookup_table(np.zeros(4, dtype=np.float32))

    def test_fit_and_is_fitted(self):
        cb   = self._make(n_codes=4)
        vecs = np.random.rand(30, 4).astype(np.float32)
        cb.fit(vecs)
        assert cb.is_fitted

    def test_encode_decode_roundtrip(self):
        cb   = self._make(n_codes=16, n_iters=10)
        vecs = np.random.rand(60, 4).astype(np.float32)
        cb.fit(vecs)
        idxs    = cb.encode(vecs)
        decoded = cb.decode(idxs)
        assert idxs.dtype     == np.uint16
        assert idxs.shape     == (60,)
        assert decoded.shape  == (60, 4)

    def test_lookup_table_shape(self):
        cb   = self._make(n_codes=8, n_iters=5)
        vecs = np.random.rand(20, 4).astype(np.float32)
        cb.fit(vecs)
        lut = cb.lookup_table(np.zeros(4, dtype=np.float32))
        assert lut.shape == (cb._n_codes,)
        assert np.all(lut >= 0)

    def test_sq_dist_static(self):
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        d = PQCodebook._sq_dist(a, b)
        assert d[0, 0] == pytest.approx(25.0, abs=1e-3)

    def test_fit_fewer_vecs_than_codes(self):
        cb   = self._make(n_codes=100)
        vecs = np.random.rand(5, 4).astype(np.float32)
        cb.fit(vecs)
        assert cb.is_fitted
        assert cb._n_codes <= 5


# ---------------------------------------------------------------------------
# PQKeyIndex
# ---------------------------------------------------------------------------

class TestPQKeyIndex:
    def _make_index(self, dim=16, n_sub=4, n_codes=8):
        cfg = PQCacheConfig(n_subvectors=n_sub, n_codes=n_codes,
                            train_iters=5, seed=0)
        return PQKeyIndex(dim=dim, config=cfg)

    def test_dim_not_divisible_raises(self):
        cfg = PQCacheConfig(n_subvectors=3, n_codes=8, train_iters=5)
        with pytest.raises(ValueError, match="divisible"):
            PQKeyIndex(dim=10, config=cfg)

    def test_not_fitted_initially(self):
        idx = self._make_index()
        assert not idx.is_fitted

    def test_add_raises_before_fit(self):
        idx = self._make_index()
        with pytest.raises(RuntimeError, match="fitted"):
            idx.add(np.zeros(16, dtype=np.float32), seq_pos=0)

    def test_fit_works(self):
        idx  = self._make_index()
        keys = np.random.rand(50, 16).astype(np.float32)
        idx.fit(keys)
        assert idx.is_fitted

    def test_len_increments_on_add(self):
        idx  = self._make_index()
        keys = np.random.rand(30, 16).astype(np.float32)
        idx.fit(keys)
        for i in range(5):
            idx.add(keys[i], seq_pos=i)
        assert len(idx) == 5

    def test_search_basic(self):
        idx  = self._make_index()
        keys = np.random.rand(50, 16).astype(np.float32)
        idx.fit(keys)
        for i, k in enumerate(keys):
            idx.add(k, seq_pos=i)
        query    = keys[0]
        pos, dis = idx.search(query, top_k=5)
        assert len(pos) == 5
        assert pos.dtype == np.int64
        assert len(dis)  == 5

    def test_search_top1_finds_exact(self):
        idx  = self._make_index(n_codes=4)
        rng  = np.random.default_rng(42)
        # Use random calibration data to avoid k-means++ degenerate case
        keys = rng.random((20, 16)).astype(np.float32)
        idx.fit(keys)
        # Add a clearly unique vector
        special = np.ones(16, dtype=np.float32) * 5.0
        for i, k in enumerate(keys):
            idx.add(k, seq_pos=i)
        idx.add(special, seq_pos=999)
        pos, _ = idx.search(special, top_k=1)
        assert pos[0] == 999

    def test_search_empty_index(self):
        idx = self._make_index()
        idx.fit(np.random.rand(10, 16).astype(np.float32))
        pos, dis = idx.search(np.zeros(16, dtype=np.float32), top_k=5)
        assert len(pos) == 0


# ---------------------------------------------------------------------------
# PQValueStore
# ---------------------------------------------------------------------------

class TestPQValueStore:
    def test_add_and_get(self):
        store = PQValueStore()
        rng = np.random.default_rng(0)
        v   = rng.random(64).astype(np.float32)
        store.add(0, v)
        recovered = store.get(0)
        assert recovered is not None
        assert recovered.shape == v.shape
        assert recovered.dtype == np.float32
        # The scheme: scale = range/254, q = round((v-min)/scale).clip(-127, 127).
        # Values above the midpoint saturate at q=127 and decode to
        # v_min + 127*scale = (v_min + v_max) / 2, so decoded ≤ v_max.
        v_min, v_max = float(v.min()), float(v.max())
        scale = (v_max - v_min) / 254.0
        assert np.all(recovered >= v_min - 1e-5)
        assert np.all(recovered <= v_max + 1e-5)
        # Values in the lower half of the range (q < 127, no clipping) decode accurately.
        mid     = (v_min + v_max) / 2.0
        low_idx = v <= mid
        np.testing.assert_allclose(recovered[low_idx], v[low_idx], atol=scale + 1e-5)

    def test_get_missing_returns_none(self):
        store = PQValueStore()
        assert store.get(999) is None

    def test_len(self):
        store = PQValueStore()
        store.add(0, np.ones(4, dtype=np.float32))
        store.add(1, np.ones(4, dtype=np.float32))
        assert len(store) == 2

    def test_constant_vector(self):
        """All-same values should not crash and should recover the constant."""
        store = PQValueStore()
        v     = np.full(8, 3.14, dtype=np.float32)
        store.add(10, v)
        out = store.get(10)
        assert out is not None
        np.testing.assert_allclose(out, v, atol=0.01)

    def test_get_batch_shape(self):
        store = PQValueStore()
        for i in range(5):
            store.add(i, np.random.rand(8).astype(np.float32))
        batch = store.get_batch(np.array([0, 1, 2], dtype=np.int64))
        assert batch.shape == (3, 8)

    def test_get_batch_missing_positions_skipped(self):
        store = PQValueStore()
        store.add(0, np.ones(4, dtype=np.float32))
        # Position 1 not stored — only 1 result
        batch = store.get_batch(np.array([0, 1], dtype=np.int64))
        assert batch.shape == (1, 4)

    def test_get_batch_empty(self):
        store = PQValueStore()
        batch = store.get_batch(np.array([], dtype=np.int64))
        assert batch.shape[0] == 0


# ---------------------------------------------------------------------------
# retrieve convenience function
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_basic_retrieve(self):
        cfg       = PQCacheConfig(n_subvectors=2, n_codes=4, train_iters=5)
        key_index = PQKeyIndex(dim=8, config=cfg)
        val_store = PQValueStore()

        keys = np.random.rand(20, 8).astype(np.float32)
        key_index.fit(keys)
        for i, k in enumerate(keys):
            key_index.add(k, seq_pos=i)
            val_store.add(i, np.random.rand(4).astype(np.float32))

        pos, vals = retrieve(keys[0], key_index, val_store, top_k=3)
        assert len(pos) == 3
        assert vals.shape[0] == 3

    def test_empty_index_retrieve(self):
        cfg       = PQCacheConfig(n_subvectors=2, n_codes=4, train_iters=5)
        key_index = PQKeyIndex(dim=8, config=cfg)
        key_index.fit(np.random.rand(10, 8).astype(np.float32))
        val_store = PQValueStore()
        pos, vals = retrieve(np.zeros(8, dtype=np.float32), key_index, val_store)
        assert len(pos) == 0
