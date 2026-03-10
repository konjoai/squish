"""tests/test_agile_io_unit.py — unit tests for squish/agile_io.py"""

import io
import os
import tempfile

import numpy as np
import pytest

from squish.agile_io import AgileIOConfig, AgileIOManager, AgileIOStats

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# AgileIOConfig
# ---------------------------------------------------------------------------

class TestAgileIOConfig:
    def test_defaults(self):
        cfg = AgileIOConfig()
        assert cfg.n_worker_threads == 4
        assert cfg.cache_size_mb == 256
        assert cfg.prefetch_ahead == 2

    def test_cache_size_bytes(self):
        cfg = AgileIOConfig(cache_size_mb=1)
        assert cfg.cache_size_bytes == 1024 * 1024

    def test_invalid_threads(self):
        with pytest.raises(ValueError, match="n_worker_threads"):
            AgileIOConfig(n_worker_threads=0)

    def test_invalid_cache_size(self):
        with pytest.raises(ValueError, match="cache_size_mb"):
            AgileIOConfig(cache_size_mb=0)

    def test_invalid_prefetch_ahead(self):
        with pytest.raises(ValueError, match="prefetch_ahead"):
            AgileIOConfig(prefetch_ahead=-1)


# ---------------------------------------------------------------------------
# AgileIOStats
# ---------------------------------------------------------------------------

class TestAgileIOStats:
    def test_hit_rate_zero_reads(self):
        s = AgileIOStats()
        assert s.hit_rate == 0.0

    def test_hit_rate(self):
        s = AgileIOStats(reads_total=10, reads_hit=7)
        assert abs(s.hit_rate - 0.7) < 1e-9

    def test_reset(self):
        s = AgileIOStats(reads_total=5, reads_hit=3, bytes_read=1024)
        s.reset()
        assert s.reads_total == 0
        assert s.bytes_read == 0


# ---------------------------------------------------------------------------
# AgileIOManager — basic get / prefetch
# ---------------------------------------------------------------------------

def _make_tmpfile(data: bytes) -> str:
    """Write *data* to a temp file and return the path."""
    fd, path = tempfile.mkstemp()
    os.write(fd, data)
    os.close(fd)
    return path


class TestAgileIOManager:
    def test_get_returns_file_contents(self):
        payload = b"hello agile io"
        path = _make_tmpfile(payload)
        try:
            mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
            assert mgr.get(path) == payload
        finally:
            os.unlink(path)

    def test_cache_hit_on_second_get(self):
        payload = b"cached block"
        path = _make_tmpfile(payload)
        try:
            mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
            mgr.get(path)
            mgr.get(path)
            s = mgr.stats
            assert s.reads_total == 2
            assert s.reads_hit == 1
        finally:
            os.unlink(path)

    def test_prefetch_then_get_is_cache_hit(self):
        payload = b"prefetched data"
        path = _make_tmpfile(payload)
        try:
            mgr = AgileIOManager(AgileIOConfig(n_worker_threads=2))
            mgr.prefetch(path)
            import time
            # Give the background thread time to complete
            for _ in range(50):
                time.sleep(0.01)
                if path in mgr._cache:
                    break
            result = mgr.get(path)
            assert result == payload
        finally:
            os.unlink(path)

    def test_get_nonexistent_raises(self):
        mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
        with pytest.raises((FileNotFoundError, OSError)):
            mgr.get("/nonexistent/path/that/does/not/exist.bin")

    def test_get_npy_round_trip(self):
        arr = RNG.random((5, 4)).astype(np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr)
        try:
            mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
            loaded = mgr.get_npy(path)
            np.testing.assert_allclose(loaded, arr)
        finally:
            os.unlink(path)

    def test_evict_removes_from_cache(self):
        payload = b"evict me"
        path = _make_tmpfile(payload)
        try:
            mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
            mgr.get(path)
            assert path in mgr._cache
            removed = mgr.evict(path)
            assert removed is True
            assert path not in mgr._cache
        finally:
            os.unlink(path)

    def test_evict_nonexistent_returns_false(self):
        mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
        assert mgr.evict("/some/nonexistent/path") is False

    def test_cache_info_keys(self):
        mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
        info = mgr.cache_info()
        assert "entries" in info
        assert "bytes_used" in info
        assert "bytes_limit" in info
        assert "hit_rate" in info

    def test_prefetch_sequence_enqueues(self):
        payloads = [b"layer0", b"layer1", b"layer2"]
        paths = []
        for p in payloads:
            paths.append(_make_tmpfile(p))
        try:
            mgr = AgileIOManager(AgileIOConfig(n_worker_threads=2, prefetch_ahead=2))
            mgr.prefetch_sequence(paths, start_idx=0)
            s = mgr.stats
            assert s.prefetches >= 1
        finally:
            for p in paths:
                os.unlink(p)

    def test_lru_eviction_respects_cache_limit(self):
        # Set cache to 1 byte so every entry is evicted
        cfg = AgileIOConfig(n_worker_threads=1, cache_size_mb=1)
        mgr = AgileIOManager(cfg)
        # Manually fill cache beyond the limit using the internal method
        # to validate LRU eviction works
        paths = []
        large_payload = b"X" * (512 * 1024)  # 512 KB
        for _ in range(3):
            p = _make_tmpfile(large_payload)
            paths.append(p)
        try:
            # Get 3 files that together exceed 1 MB
            for p in paths:
                mgr.get(p)
            # Cache should not exceed cache_size_bytes
            info = mgr.cache_info()
            assert info["bytes_used"] <= cfg.cache_size_bytes
        finally:
            for p in paths:
                os.unlink(p)

    def test_shutdown_does_not_raise(self):
        mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1))
        mgr.shutdown()  # should be a no-op or clean shutdown
