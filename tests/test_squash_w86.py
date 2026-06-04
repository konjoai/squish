"""Wave 86 tests — pre-load ModelScan gate for `squish pull hf:`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from squish.cli import _pull_from_hf
from squish.serving.local_model_scanner import LocalModelScanner, PreDownloadScanResult

pytest.importorskip("huggingface_hub")


def test_scan_before_load_flags_synthetic_malicious_pickle(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Synthetic malicious pickle payload (contains REDUCE opcode 0x52).
    (model_dir / "weights.pkl").write_bytes(b"\x80\x04\x52\x2e")

    result = LocalModelScanner.scan_before_load(model_dir)
    assert result.status == "unsafe"
    assert len(result.findings) >= 1


def test_pull_from_hf_runs_preload_scan(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()

    clean = PreDownloadScanResult(status="clean")

    with patch("squish.cli._CATALOG_AVAILABLE", False):
        with patch("huggingface_hub.snapshot_download", return_value=str(downloaded)):
            with patch(
                "squish.serving.local_model_scanner.LocalModelScanner.scan_before_load",
                return_value=clean,
            ) as scan_mock:
                _pull_from_hf("org/model", tmp_path, token=None)

    scan_mock.assert_called_once_with(Path(downloaded))


def test_pull_from_hf_unsafe_exits_code_2(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "weights.pkl").write_bytes(b"\x80\x04\x52\x2e")

    with patch("squish.cli._CATALOG_AVAILABLE", False):
        with patch("huggingface_hub.snapshot_download", return_value=str(downloaded)):
            with pytest.raises(SystemExit) as exc:
                _pull_from_hf("org/model", tmp_path, token=None)

    assert exc.value.code == 2
