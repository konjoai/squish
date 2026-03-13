"""
tests/test_entropy_branches.py

Branch coverage for squish/entropy.py:
  - compress_npy_dir: sentinel exists (line 75)
  - benchmark_compression: plain .npy items (lines 221-225)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

# ── compress_npy_dir: sentinel already present ───────────────────────────────

class TestCompressNpyDirSentinel:
    def test_sentinel_present_verbose_prints_skip(self, tmp_path: Path, capsys):
        pytest.importorskip("zstandard")
        from squish.entropy import compress_npy_dir

        # Write at least one .npy file so the dir check passes
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        np.save(tensors_dir / "layer.npy", arr)

        # Create the sentinel file
        sentinel = tensors_dir.parent / ".squish_zst_ready"
        sentinel.touch()

        result = compress_npy_dir(tensors_dir, verbose=True)
        captured = capsys.readouterr()
        assert result == {}
        assert "Already compressed" in captured.out or "skipping" in captured.out.lower()

    def test_sentinel_present_silent(self, tmp_path: Path):
        pytest.importorskip("zstandard")
        from squish.entropy import compress_npy_dir

        arr = np.array([1.0], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        np.save(tensors_dir / "layer.npy", arr)

        sentinel = tensors_dir.parent / ".squish_zst_ready"
        sentinel.touch()

        result = compress_npy_dir(tensors_dir, verbose=False)
        assert result == {}


# ── benchmark_compression ─────────────────────────────────────────────────────

class TestBenchmarkCompression:
    def test_runs_with_npy_files(self, tmp_path: Path, capsys):
        """benchmark_compression on a dir with .npy files produces output."""
        pytest.importorskip("zstandard")
        from squish.entropy import benchmark_compression

        arr = np.zeros((4, 4), dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        np.save(tensors_dir / "weights.npy", arr)

        benchmark_compression(tensors_dir)
        captured = capsys.readouterr()
        assert "weights.npy" in captured.out or "Tensor" in captured.out


# ── decompress_npy_dir: sentinel absent (branch [152, 155]) ──────────────────

class TestDecompressNpyDirNoSentinel:
    def test_no_sentinel_decompresses_and_continues(self, tmp_path: Path, capsys):
        """
        When sentinel file does NOT exist, the unlink step is skipped and
        execution continues to the verbose print  (line 152→155).
        """
        pytest.importorskip("zstandard")
        import zstandard

        from squish.entropy import decompress_npy_dir

        # Create a .npy.zst file
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        npy_buf = io.BytesIO()
        np.save(npy_buf, arr)

        cctx = zstandard.ZstdCompressor()
        zst_data = cctx.compress(npy_buf.getvalue())
        (tensors_dir / "layer.npy.zst").write_bytes(zst_data)

        # Ensure sentinel does NOT exist
        sentinel = tensors_dir.parent / ".squish_zst_ready"
        assert not sentinel.exists()

        decompress_npy_dir(tensors_dir, verbose=True)

        captured = capsys.readouterr()
        assert "Decompressed" in captured.out or "tensor" in captured.out.lower()


# ── Brotli paths (mocked — brotli is an optional dependency) ─────────────────

import sys
import io
import types
from unittest.mock import patch as _mock_patch


def _make_mock_brotli():
    """
    Construct a mock 'brotli' module whose compress() and decompress() methods
    use zlib under the hood so round-trip tests work correctly.
    """
    import zlib

    mod = types.ModuleType("brotli")

    def _compress(data: bytes, quality: int = 11) -> bytes:
        # Prefix so we can identify mock-compressed data in tests
        return b"brmock:" + zlib.compress(data)

    def _decompress(data: bytes) -> bytes:
        prefix = b"brmock:"
        if data.startswith(prefix):
            return zlib.decompress(data[len(prefix):])
        return zlib.decompress(data)

    mod.compress   = _compress
    mod.decompress = _decompress
    return mod


class TestRequireBrotli:
    def test_require_brotli_returns_module_when_available(self):
        """_require_brotli() returns the brotli module when it can be imported."""
        from squish.entropy import _require_brotli
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            result = _require_brotli()
        assert result is mock_brotli

    def test_require_brotli_has_compress_attr(self):
        from squish.entropy import _require_brotli
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            result = _require_brotli()
        assert callable(result.compress)


class TestCompressNpyDirBrotli:
    def _make_tensors_dir(self, tmp_path):
        td = tmp_path / "tensors"
        td.mkdir()
        for i in range(3):
            arr = np.random.default_rng(i).standard_normal((4, 8)).astype(np.float32)
            np.save(str(td / f"layer_{i}.npy"), arr)
        return td

    def test_brotli_creates_br_files(self, tmp_path):
        from squish.entropy import compress_npy_dir
        td = self._make_tensors_dir(tmp_path)
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            stats = compress_npy_dir(td, level=1, verbose=False, codec="brotli")
        br_files = list(td.glob("*.npy.br"))
        assert len(br_files) == 3
        assert stats["codec"] == "brotli"
        assert stats["files"] == 3

    def test_brotli_removes_original_npy(self, tmp_path):
        from squish.entropy import compress_npy_dir
        td = self._make_tensors_dir(tmp_path)
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            compress_npy_dir(td, level=1, verbose=False, codec="brotli")
        npy_files = list(td.glob("*.npy"))
        assert len(npy_files) == 0

    def test_brotli_creates_sentinel(self, tmp_path):
        from squish.entropy import compress_npy_dir
        td = self._make_tensors_dir(tmp_path)
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            compress_npy_dir(td, verbose=False, codec="brotli")
        sentinel = tmp_path / ".squish_br_ready"
        assert sentinel.exists()

    def test_brotli_skips_if_sentinel_present(self, tmp_path):
        from squish.entropy import compress_npy_dir
        td = self._make_tensors_dir(tmp_path)
        # Pre-create sentinel
        (tmp_path / ".squish_br_ready").write_text("squish-br-v1")
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            stats = compress_npy_dir(td, verbose=False, codec="brotli")
        assert stats == {}

    def test_brotli_verbose_output(self, tmp_path, capsys):
        from squish.entropy import compress_npy_dir
        td = self._make_tensors_dir(tmp_path)
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            compress_npy_dir(td, level=1, verbose=True, codec="brotli")
        out = capsys.readouterr().out
        assert "brotli" in out.lower() or "%" in out

    def test_brotli_stats_keys(self, tmp_path):
        from squish.entropy import compress_npy_dir
        td = self._make_tensors_dir(tmp_path)
        mock_brotli = _make_mock_brotli()
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            stats = compress_npy_dir(td, verbose=False, codec="brotli")
        for key in ("files", "codec", "orig_gb", "comp_gb", "ratio", "elapsed_s"):
            assert key in stats, f"Missing stats key: {key!r}"


class TestDecompressNpyDirBrotli:
    def _make_br_tensors(self, tmp_path):
        """Create a tensors/ dir populated with mock-brotli .npy.br files."""
        td = tmp_path / "tensors"
        td.mkdir()
        mb = _make_mock_brotli()
        for i in range(2):
            arr = np.random.default_rng(i).standard_normal((4, 8)).astype(np.float32)
            buf = io.BytesIO()
            np.save(buf, arr)
            compressed = mb.compress(buf.getvalue())
            (td / f"layer_{i}.npy.br").write_bytes(compressed)
        return td

    def test_brotli_decompress_creates_npy_files(self, tmp_path):
        from squish.entropy import decompress_npy_dir
        td = self._make_br_tensors(tmp_path)
        mock_brotli = _make_mock_brotli()
        # Also need zstandard available for the zst path (or no .zst files)
        pytest.importorskip("zstandard")
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            decompress_npy_dir(td, verbose=False)
        npy_files = list(td.glob("*.npy"))
        assert len(npy_files) == 2

    def test_brotli_decompress_removes_sentinel(self, tmp_path):
        from squish.entropy import decompress_npy_dir
        td = self._make_br_tensors(tmp_path)
        sentinel = tmp_path / ".squish_br_ready"
        sentinel.write_text("squish-br-v1")
        mock_brotli = _make_mock_brotli()
        pytest.importorskip("zstandard")
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            decompress_npy_dir(td, verbose=True)
        assert not sentinel.exists()

    def test_decompress_early_return_no_zst(self, tmp_path):
        """Line 230: early return when no zst files (and brotli already handled)."""
        from squish.entropy import decompress_npy_dir
        td = self._make_br_tensors(tmp_path)
        mock_brotli = _make_mock_brotli()
        pytest.importorskip("zstandard")
        # No .npy.zst files — should early return after brotli pass
        with _mock_patch.dict(sys.modules, {"brotli": mock_brotli}):
            decompress_npy_dir(td, verbose=False)
        # If no exception raised, early return path was hit
        assert True
