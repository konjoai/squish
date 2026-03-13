"""tests/test_entropy_brotli.py

Coverage tests for the brotli code paths in squish/entropy.py:
  - _require_brotli(): happy path (lines 50-52)
  - compress_npy_dir() with codec="brotli" (lines 88-134)
  - compress_npy_dir() brotli with sentinel already present
  - decompress_npy_dir() with .npy.br files (lines 205-219)
  - decompress_npy_dir() only brotli files (no zstd files) → early return
    at line 230

All tests use a mocked `brotli` module so they run regardless of whether
the `brotli` package is installed in the environment.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_brotli() -> MagicMock:
    """Return a minimal mock of the `brotli` module."""
    mock = MagicMock(name="brotli")

    def _compress(data, quality=11):
        # Trivial "compression": just prefix with a sentinel byte and return data
        return b"\xfb" + data  # 1 extra byte overhead

    def _decompress(data):
        # Strip the sentinel byte we added in _compress
        if data and data[0:1] == b"\xfb":
            return data[1:]
        return data

    mock.compress.side_effect = _compress
    mock.decompress.side_effect = _decompress
    return mock


def _write_npy(path: Path, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    path.write_bytes(buf.getvalue())


def _write_npy_br(path: Path, arr: np.ndarray, mock_brotli) -> None:
    """Write a fake .npy.br file using the mock compressor."""
    buf = io.BytesIO()
    np.save(buf, arr)
    compressed = mock_brotli.compress(buf.getvalue(), quality=5)
    path.write_bytes(compressed)


# ---------------------------------------------------------------------------
# _require_brotli — happy path (lines 50-52)
# ---------------------------------------------------------------------------


class TestRequireBrotli:
    def test_returns_brotli_module_when_available(self):
        mock_brotli = _make_mock_brotli()
        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            # Force re-import of the function so it sees the patched module
            import importlib
            import squish.entropy as entropy_mod
            importlib.reload(entropy_mod)
            result = entropy_mod._require_brotli()
        assert result is not None


# ---------------------------------------------------------------------------
# compress_npy_dir with codec="brotli"
# ---------------------------------------------------------------------------


class TestCompressNpyDirBrotli:
    def test_brotli_compress_basic(self, tmp_path: Path):
        """compress_npy_dir with codec='brotli' creates .npy.br files."""
        mock_brotli = _make_mock_brotli()
        arr = np.zeros((4,), dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy(tensors_dir / "w.npy", arr)

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            from squish.entropy import compress_npy_dir  # noqa: PLC0415
            stats = compress_npy_dir(tensors_dir, level=5, codec="brotli", verbose=False)

        assert isinstance(stats, dict)
        assert stats.get("codec") == "brotli"
        # .npy.br should exist; original .npy should be deleted
        assert (tensors_dir / "w.npy.br").exists()
        assert not (tensors_dir / "w.npy").exists()
        # Sentinel file should be written
        sentinel = tensors_dir.parent / ".squish_br_ready"
        assert sentinel.exists()

    def test_brotli_compress_verbose(self, tmp_path: Path, capsys):
        """With verbose=True, progress is printed."""
        mock_brotli = _make_mock_brotli()
        arr = np.zeros((8,), dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy(tensors_dir / "layer.npy", arr)

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            from squish.entropy import compress_npy_dir  # noqa: PLC0415
            stats = compress_npy_dir(tensors_dir, level=3, codec="brotli", verbose=True)

        captured = capsys.readouterr()
        assert "Compressed" in captured.out or "brotli" in captured.out.lower()

    def test_brotli_compress_sentinel_already_present(self, tmp_path: Path):
        """When .squish_br_ready exists, skip and return {}."""
        mock_brotli = _make_mock_brotli()
        arr = np.zeros((4,), dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy(tensors_dir / "w.npy", arr)

        # Create sentinel
        sentinel = tensors_dir.parent / ".squish_br_ready"
        sentinel.write_text("squish-br-v1")

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            from squish.entropy import compress_npy_dir  # noqa: PLC0415
            stats = compress_npy_dir(tensors_dir, codec="brotli", verbose=False)

        assert stats == {}
        # .npy should still exist (nothing was compressed)
        assert (tensors_dir / "w.npy").exists()

    def test_brotli_compress_sentinel_present_verbose(self, tmp_path: Path, capsys):
        """Sentinel present + verbose=True → skip message printed."""
        mock_brotli = _make_mock_brotli()
        arr = np.zeros((4,), dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy(tensors_dir / "w.npy", arr)

        sentinel = tensors_dir.parent / ".squish_br_ready"
        sentinel.write_text("squish-br-v1")

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            from squish.entropy import compress_npy_dir  # noqa: PLC0415
            compress_npy_dir(tensors_dir, codec="brotli", verbose=True)

        captured = capsys.readouterr()
        assert "Already compressed" in captured.out or "skipping" in captured.out.lower()

    def test_brotli_compress_returns_stats_dict(self, tmp_path: Path):
        """Returned stats dict has expected keys."""
        mock_brotli = _make_mock_brotli()
        arr = np.ones((16,), dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy(tensors_dir / "w.npy", arr)

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            from squish.entropy import compress_npy_dir  # noqa: PLC0415
            stats = compress_npy_dir(tensors_dir, codec="brotli", verbose=False)

        assert "files" in stats
        assert "orig_gb" in stats
        assert "comp_gb" in stats
        assert stats["files"] == 1


# ---------------------------------------------------------------------------
# decompress_npy_dir with .npy.br files (lines 205-219)
# ---------------------------------------------------------------------------


class TestDecompressNpyDirBrotli:
    def test_decompress_brotli_files(self, tmp_path: Path):
        """decompress_npy_dir handles .npy.br files correctly."""
        mock_brotli = _make_mock_brotli()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy_br(tensors_dir / "w.npy.br", arr, mock_brotli)

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            # Also need zstd to be available (decompress_npy_dir calls _require_zstd)
            pytest.importorskip("zstandard")
            from squish.entropy import decompress_npy_dir  # noqa: PLC0415
            decompress_npy_dir(tensors_dir, verbose=False)

        # .npy file should be created
        assert (tensors_dir / "w.npy").exists()

    def test_decompress_brotli_verbose(self, tmp_path: Path, capsys):
        """With verbose=True, decompression count is printed."""
        mock_brotli = _make_mock_brotli()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy_br(tensors_dir / "w.npy.br", arr, mock_brotli)

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            pytest.importorskip("zstandard")
            from squish.entropy import decompress_npy_dir  # noqa: PLC0415
            decompress_npy_dir(tensors_dir, verbose=True)

        captured = capsys.readouterr()
        assert "Decompressed" in captured.out or "brotli" in captured.out.lower()

    def test_decompress_brotli_removes_sentinel(self, tmp_path: Path):
        """When .squish_br_ready sentinel exists, it is removed after decompress."""
        mock_brotli = _make_mock_brotli()
        arr = np.array([0.5], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy_br(tensors_dir / "w.npy.br", arr, mock_brotli)

        sentinel = tensors_dir.parent / ".squish_br_ready"
        sentinel.write_text("squish-br-v1")

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            pytest.importorskip("zstandard")
            from squish.entropy import decompress_npy_dir  # noqa: PLC0415
            decompress_npy_dir(tensors_dir, verbose=False)

        assert not sentinel.exists()

    def test_decompress_only_brotli_files_no_zstd_early_return(self, tmp_path: Path):
        """When only .npy.br files exist (no .npy.zst), returns after brotli decompression.

        This covers line 230: `if not zst_files: return`
        """
        mock_brotli = _make_mock_brotli()
        arr = np.array([7.0, 8.0], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        _write_npy_br(tensors_dir / "tensor.npy.br", arr, mock_brotli)

        # There are NO .npy.zst files — only .npy.br
        assert not list(tensors_dir.glob("*.npy.zst"))

        with patch.dict(sys.modules, {"brotli": mock_brotli}):
            pytest.importorskip("zstandard")
            from squish.entropy import decompress_npy_dir  # noqa: PLC0415
            # Should run without error and return after handling brotli files
            decompress_npy_dir(tensors_dir, verbose=False)

        # The .npy file was created from the .br file
        assert (tensors_dir / "tensor.npy").exists()
