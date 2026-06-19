"""Behavioral coverage for the error/edge paths of
``squish.serving.local_model_scanner`` left untested by the baseline suite:
the generic-HTTP-error branch, the >50-file overflow accounting, file-read
error handlers in the pre-download scanners, ``_dir_size`` failure, and the
ollama / lm-studio directory-walk edge cases.

All paths are pure-Python + filesystem (tmp_path) — no MLX, no network.
"""
from __future__ import annotations

import json
import struct
import urllib.error
from pathlib import Path
from unittest.mock import patch

from squish.serving.local_model_scanner import (
    LocalModelScanner,
    _classify_hf_siblings,
    _dir_size,
    _scan_gguf_file,
    _scan_pickle_file,
    _scan_safetensors_file,
    scan_before_load,
    scan_hf_repo_metadata,
)


# ── scan_hf_repo_metadata: generic HTTP error ───────────────────────────────


def test_hf_scan_generic_http_error():
    exc = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
    with patch("urllib.request.urlopen", side_effect=exc):
        r = scan_hf_repo_metadata("owner/model")
    assert r.status == "error"
    # Generic branch (not 401/404) includes the HTTP code and reason.
    assert any("HTTP 503" in f and "Service Unavailable" in f for f in r.findings)


# ── _classify_hf_siblings: files beyond _REPORT_MAX_FILES ────────────────────


def test_classify_counts_files_beyond_report_cap():
    # 50 padding safe files + 3 overflow files (one of each class) at indices 50+.
    siblings = [{"rfilename": f"shard{i}.safetensors", "size": 10} for i in range(50)]
    siblings += [
        {"rfilename": "overflow.pkl", "size": 5},          # dangerous (overflow)
        {"rfilename": "overflow.bin", "size": 5},          # potentially unsafe (overflow)
        {"rfilename": "overflow.safetensors", "size": 5},  # safe (overflow)
        {"rfilename": "overflow.json", "size": 5},         # neutral suffix → no class
    ]
    result = _classify_hf_siblings("owner/big", siblings)
    # The overflow .pkl makes the repo unsafe and is counted in total size.
    assert result.status == "unsafe"
    assert result.total_size_bytes == 50 * 10 + 5 * 4
    # Only the first 50 files get per-file summaries (the cap).
    assert len(result.file_summary) == 50


# ── scan_before_load: non-file entries skipped ──────────────────────────────


def test_scan_before_load_skips_directories(tmp_path):
    # A subdirectory whose name matches a scanned suffix must be skipped, not read.
    (tmp_path / "weights.safetensors").mkdir()  # dir, not file → continue (351)
    (tmp_path / "real.gguf").write_bytes(b"GGUF" + b"\x00" * 16)
    result = scan_before_load(tmp_path)
    assert result.status == "clean"
    assert result.scanned == 1  # only the real .gguf file was scanned


# ── pre-download file scanners: read-error handlers ─────────────────────────


def _patch_read_oserror():
    return patch.object(Path, "read_bytes",
                        side_effect=OSError("permission denied"))


def test_scan_pickle_read_error(tmp_path):
    p = tmp_path / "x.pkl"
    p.write_bytes(b"\x80\x04.")
    with _patch_read_oserror():
        out = _scan_pickle_file(p)
    assert out == [f"[READ ERROR] {p.name}: permission denied"]


def test_scan_gguf_read_error(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"GGUF")
    with _patch_read_oserror():
        out = _scan_gguf_file(p)
    assert out == [f"[READ ERROR] {p.name}: permission denied"]


def test_scan_safetensors_read_error(tmp_path):
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", 8) + b"{}")
    with _patch_read_oserror():
        out = _scan_safetensors_file(p)
    assert out == [f"[READ ERROR] {p.name}: permission denied"]


# ── _dir_size: failure → 0 with a warning ───────────────────────────────────


def test_dir_size_oserror_returns_zero(tmp_path, caplog):
    (tmp_path / "f.bin").write_bytes(b"x" * 100)
    with patch.object(Path, "rglob", side_effect=OSError("walk failed")):
        assert _dir_size(tmp_path) == 0


# ── scan_ollama: non-dir entry + non-file tag skips ─────────────────────────


def test_scan_ollama_skips_non_dir_and_non_file_tag(tmp_path):
    ollama = tmp_path / "ollama"
    ollama.mkdir()
    (ollama / "loose.txt").write_text("x")          # non-dir at top → skip (557)
    model_dir = ollama / "qwen3"
    model_dir.mkdir()
    (model_dir / "subdir").mkdir()                   # non-file tag → skip (561)
    tag = model_dir / "8b"
    tag.write_text(json.dumps({"layers": [{"size": 123}]}))
    scanner = LocalModelScanner(
        squish_models_dir=tmp_path / "none",
        ollama_manifests_dir=ollama,
        lm_studio_dir=tmp_path / "none",
    )
    models = scanner.scan_ollama()
    assert len(models) == 1
    assert models[0].name == "qwen3:8b"
    assert models[0].size_bytes == 123


def test_scan_ollama_bad_manifest_reports_zero_size(tmp_path):
    ollama = tmp_path / "ollama"
    (ollama / "qwen3").mkdir(parents=True)
    (ollama / "qwen3" / "8b").write_text("{ not json")
    scanner = LocalModelScanner(
        squish_models_dir=tmp_path / "none",
        ollama_manifests_dir=ollama,
        lm_studio_dir=tmp_path / "none",
    )
    models = scanner.scan_ollama()
    assert len(models) == 1 and models[0].size_bytes == 0


# ── scan_lm_studio: name derivation + dedup + non-file marker ───────────────


def _lm_scanner(tmp_path, root):
    return LocalModelScanner(
        squish_models_dir=tmp_path / "none",
        ollama_manifests_dir=tmp_path / "none",
        lm_studio_dir=root,
    )


def test_lm_studio_gguf_two_level_name(tmp_path):
    root = tmp_path / "lm"
    pub = root / "publisher"
    pub.mkdir(parents=True)
    (pub / "model.gguf").write_bytes(b"GGUF" + b"\x00" * 8)  # 2 rel parts → name = publisher
    models = _lm_scanner(tmp_path, root).scan_lm_studio()
    assert len(models) == 1 and models[0].name == "publisher"


def test_lm_studio_safetensors_dedup_and_name_levels(tmp_path):
    root = tmp_path / "lm"
    repo = root / "pub" / "repo"
    repo.mkdir(parents=True)
    # Two markers in the same repo_dir → second hits the seen-dir dedup (644).
    (repo / "model.safetensors").write_bytes(b"\x00" * 16)
    (repo / "model.safetensors.index.json").write_text("{}")
    # A non-file matching the marker glob → skipped (641).
    (root / "pub" / "model.safetensors.d").mkdir()
    models = _lm_scanner(tmp_path, root).scan_lm_studio()
    repo_models = [m for m in models if m.source == "lm_studio"]
    assert len(repo_models) == 1
    assert repo_models[0].name == "pub/repo"  # >= 2 rel parts


def test_lm_studio_safetensors_one_level_name(tmp_path):
    root = tmp_path / "lm"
    repo = root / "solo"
    repo.mkdir(parents=True)
    (repo / "model.safetensors").write_bytes(b"\x00" * 16)  # 1 rel part → name = solo
    models = _lm_scanner(tmp_path, root).scan_lm_studio()
    assert any(m.name == "solo" for m in models)


def test_lm_studio_safetensors_at_root_uses_dir_name(tmp_path):
    root = tmp_path / "lm"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"\x00" * 16)  # 0 rel parts → repo_dir.name = lm
    models = _lm_scanner(tmp_path, root).scan_lm_studio()
    assert any(m.name == "lm" for m in models)
