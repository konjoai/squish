"""tests/test_vector_index_mrl_unit.py — 100% coverage for MRLIndex in squish/vector_index.py"""
import numpy as np
import pytest

from squish.vector_index import MRLIndex

# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestMRLIndexInit:
    def test_defaults(self):
        idx = MRLIndex(full_dim=64, coarse_dim=16)
        assert idx.full_dim   == 64
        assert idx.coarse_dim == 16
        assert idx.count      == 0

    def test_invalid_full_dim(self):
        with pytest.raises(ValueError, match="full_dim"):
            MRLIndex(full_dim=0, coarse_dim=1)

    def test_invalid_coarse_dim_zero(self):
        with pytest.raises(ValueError, match="coarse_dim"):
            MRLIndex(full_dim=8, coarse_dim=0)

    def test_invalid_coarse_dim_greater_than_full(self):
        with pytest.raises(ValueError, match="coarse_dim"):
            MRLIndex(full_dim=8, coarse_dim=16)

    def test_coarse_dim_equals_full_dim_ok(self):
        idx = MRLIndex(full_dim=8, coarse_dim=8)
        assert idx.coarse_dim == 8

    def test_normalize_false(self):
        idx = MRLIndex(full_dim=8, coarse_dim=4, normalize=False)
        assert idx._normalize is False


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestMRLIndexAdd:
    def test_add_increments_count(self):
        idx  = MRLIndex(full_dim=8, coarse_dim=4)
        vecs = np.random.rand(5, 8).astype(np.float32)
        idx.add(vecs, np.arange(5, dtype=np.int64))
        assert idx.count == 5

    def test_add_multiple_batches(self):
        idx  = MRLIndex(full_dim=8, coarse_dim=4)
        vecs = np.random.rand(3, 8).astype(np.float32)
        idx.add(vecs, np.arange(3, dtype=np.int64))
        idx.add(vecs, np.arange(3, 6, dtype=np.int64))
        assert idx.count == 6

    def test_add_wrong_dim_raises(self):
        idx  = MRLIndex(full_dim=8, coarse_dim=4)
        vecs = np.random.rand(3, 16).astype(np.float32)
        with pytest.raises(ValueError, match="dim"):
            idx.add(vecs, np.arange(3, dtype=np.int64))

    def test_add_normalizes_vectors(self):
        idx  = MRLIndex(full_dim=4, coarse_dim=2, normalize=True)
        vecs = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        idx.add(vecs, np.array([0], dtype=np.int64))
        stored = idx._full_vecs[0]
        np.testing.assert_allclose(np.linalg.norm(stored), 1.0, atol=1e-5)

    def test_add_no_normalization(self):
        idx  = MRLIndex(full_dim=4, coarse_dim=2, normalize=False)
        vecs = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        idx.add(vecs, np.array([0], dtype=np.int64))
        np.testing.assert_allclose(idx._full_vecs[0], vecs[0])

    def test_add_1d_vector_accepted(self):
        """A single 1-D vector should be accepted and reshaped."""
        idx = MRLIndex(full_dim=8, coarse_dim=4)
        v   = np.random.rand(8).astype(np.float32)
        idx.add(v, np.array([0], dtype=np.int64))
        assert idx.count == 1


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestMRLIndexSearch:
    def _make_populated(self, n=20, full_dim=16, coarse_dim=8):
        idx  = MRLIndex(full_dim=full_dim, coarse_dim=coarse_dim)
        vecs = np.random.rand(n, full_dim).astype(np.float32)
        ids  = np.arange(n, dtype=np.int64)
        idx.add(vecs, ids)
        return idx, vecs

    def test_search_empty_index_returns_empty(self):
        idx       = MRLIndex(full_dim=8, coarse_dim=4)
        ids, dists = idx.search(np.zeros(8, dtype=np.float32), top_k=5)
        assert len(ids)   == 0
        assert len(dists) == 0

    def test_search_returns_correct_shape(self):
        idx, vecs = self._make_populated()
        ids, dists = idx.search(vecs[0], top_k=5)
        assert len(ids)   == 5
        assert len(dists) == 5

    def test_search_ids_dtype(self):
        idx, vecs = self._make_populated()
        ids, _    = idx.search(vecs[0], top_k=3)
        assert ids.dtype == np.int64

    def test_search_wrong_query_dim_raises(self):
        idx, _ = self._make_populated(full_dim=16, coarse_dim=8)
        q      = np.zeros(8, dtype=np.float32)   # wrong dim
        with pytest.raises(ValueError, match="dim"):
            idx.search(q, top_k=3)

    def test_search_top1_finds_self(self):
        """Query with the same vector that was added should return its own ID."""
        idx  = MRLIndex(full_dim=8, coarse_dim=4, normalize=True)
        vec  = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=np.float32)
        others = np.random.rand(10, 8).astype(np.float32)
        idx.add(others, np.arange(10, dtype=np.int64))
        idx.add(vec.reshape(1, -1), np.array([99], dtype=np.int64))
        ids, _ = idx.search(vec, top_k=1)
        assert ids[0] == 99

    def test_search_top_k_capped_by_index_size(self):
        idx  = MRLIndex(full_dim=8, coarse_dim=4)
        vecs = np.random.rand(3, 8).astype(np.float32)
        idx.add(vecs, np.arange(3, dtype=np.int64))
        ids, dists = idx.search(vecs[0], top_k=10)
        assert len(ids) == 3

    def test_coarse_k_manual(self):
        """Manual coarse_k should control Stage-1 candidate count."""
        idx  = MRLIndex(full_dim=8, coarse_dim=4, coarse_k=2)
        vecs = np.random.rand(20, 8).astype(np.float32)
        idx.add(vecs, np.arange(20, dtype=np.int64))
        ids, dists = idx.search(vecs[0], top_k=3)
        # Should still return top_k results (re-ranked from coarse candidates)
        assert len(ids) <= 3

    def test_distances_are_float32(self):
        idx, vecs = self._make_populated()
        _, dists  = idx.search(vecs[0], top_k=3)
        assert dists.dtype == np.float32

    def test_search_no_normalization(self):
        idx  = MRLIndex(full_dim=8, coarse_dim=4, normalize=False)
        vecs = np.random.rand(15, 8).astype(np.float32)
        idx.add(vecs, np.arange(15, dtype=np.int64))
        ids, _ = idx.search(vecs[0], top_k=3)
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestMRLIndexProperties:
    def test_count_property(self):
        idx = MRLIndex(full_dim=8, coarse_dim=4)
        assert idx.count == 0
        idx.add(np.random.rand(3, 8).astype(np.float32),
                np.arange(3, dtype=np.int64))
        assert idx.count == 3

    def test_full_dim_property(self):
        idx = MRLIndex(full_dim=32, coarse_dim=8)
        assert idx.full_dim == 32

    def test_coarse_dim_property(self):
        idx = MRLIndex(full_dim=32, coarse_dim=8)
        assert idx.coarse_dim == 8
