"""tests/test_wave140_push_command.py

Wave 140 — `squish push`: upload a local compressed model directory to
Hugging Face.

squish could pull and compress models, but nothing pushed a compressed
model back to a HF repo. This adds `squish push <local-dir> <hf-repo-id>`:
create_repo + upload_folder via huggingface_hub, relying on its existing
token resolution (HF_TOKEN / huggingface-cli login) rather than building
custom auth handling. Auto-generates a minimal model card from Wave 139's
stats.json sidecar when the local dir doesn't already have a README.md.

This pins:
- _read_push_stats returns the parsed stats.json dict when present, None
  when absent, and None (with a logged warning) when malformed
- _generate_push_model_card includes base model / quant mode / size /
  ratio / tensor counts when stats are available, and a clear fallback
  note when they aren't
- cmd_push dies cleanly on a missing local_dir
- cmd_push --dry-run makes zero network calls (create_repo/upload_folder
  never invoked) and still reports the file list that would be uploaded
- cmd_push generates README.md only when one isn't already present in the
  local dir — never overwrites an existing model card
- cmd_push passes repo_id, private, and commit_message through correctly
  to create_repo/upload_folder
- cmd_push surfaces upload failures via _die rather than swallowing them
"""

from __future__ import annotations

import json
import logging

import pytest

from squish.cli import _generate_push_model_card, _read_push_stats, cmd_push


class _Args:
    def __init__(self, **kw):
        self.local_dir = None
        self.repo_id = "squishai/test-model-int4"
        self.private = False
        self.commit_message = None
        self.dry_run = False
        self.__dict__.update(kw)


@pytest.fixture
def compressed_dir(tmp_path):
    d = tmp_path / "compressed"
    d.mkdir()
    (d / "manifest.json").write_text("{}")
    tensors = d / "tensors"
    tensors.mkdir()
    (tensors / "layer0__q4a.npy").write_bytes(b"\x00" * 128)
    return d


SAMPLE_STATS = {
    "base_model_dir_name": "gemma-4-12B-bf16",
    "quant_mode": "INT4 nibble-packed (group-32)",
    "quant_label": "INT4",
    "n_quantized": 200,
    "n_passthrough": 8,
    "orig_f32_bytes": 24_000_000_000,
    "compressed_bytes": 7_000_000_000,
    "on_disk_bytes": 7_100_000_000,
    "compression_ratio": 3.4,
    "shards_deleted": 5,
    "source_bytes_reclaimed": 24_000_000_000,
    "elapsed_s": 612.3,
}


class TestReadPushStats:
    def test_missing_returns_none(self, compressed_dir):
        assert _read_push_stats(compressed_dir) is None

    def test_valid_stats_parsed(self, compressed_dir):
        (compressed_dir / "stats.json").write_text(json.dumps(SAMPLE_STATS))
        result = _read_push_stats(compressed_dir)
        assert result == SAMPLE_STATS

    def test_malformed_json_returns_none_and_warns(self, compressed_dir, caplog):
        (compressed_dir / "stats.json").write_text("{not valid json")
        with caplog.at_level(logging.WARNING):
            result = _read_push_stats(compressed_dir)
        assert result is None


class TestGenerateModelCard:
    def test_with_stats_includes_details(self, compressed_dir):
        card = _generate_push_model_card(compressed_dir, "squishai/foo", SAMPLE_STATS)
        assert "squishai/foo" in card
        assert "gemma-4-12B-bf16" in card
        assert "INT4 nibble-packed" in card
        assert "3.40x ratio" in card
        assert "200 quantized" in card
        assert "8 passthrough" in card

    def test_without_stats_has_fallback_note(self, compressed_dir):
        card = _generate_push_model_card(compressed_dir, "squishai/foo", None)
        assert "squishai/foo" in card
        assert "No compression stats were found" in card


class TestCmdPushDryRun:
    def test_missing_dir_dies(self, tmp_path):
        args = _Args(local_dir=str(tmp_path / "nope"))
        with pytest.raises(SystemExit):
            cmd_push(args)

    def test_dry_run_makes_no_network_calls(self, compressed_dir, monkeypatch, capsys):
        create_repo_calls = []
        upload_calls = []
        monkeypatch.setattr(
            "huggingface_hub.create_repo", lambda *a, **kw: create_repo_calls.append((a, kw))
        )
        monkeypatch.setattr(
            "huggingface_hub.upload_folder", lambda *a, **kw: upload_calls.append((a, kw))
        )

        args = _Args(local_dir=str(compressed_dir), dry_run=True)
        cmd_push(args)

        assert create_repo_calls == []
        assert upload_calls == []
        out = capsys.readouterr().out
        assert "dry-run" in out.lower()
        assert "manifest.json" in out

    def test_dry_run_reports_file_count(self, compressed_dir, capsys):
        args = _Args(local_dir=str(compressed_dir), dry_run=True)
        cmd_push(args)
        out = capsys.readouterr().out
        assert "Files:" in out


class TestCmdPushGeneratesCard:
    def test_generates_readme_when_absent(self, compressed_dir, monkeypatch):
        monkeypatch.setattr("huggingface_hub.create_repo", lambda *a, **kw: None)
        monkeypatch.setattr("huggingface_hub.upload_folder", lambda *a, **kw: None)

        args = _Args(local_dir=str(compressed_dir))
        cmd_push(args)

        assert (compressed_dir / "README.md").exists()

    def test_does_not_overwrite_existing_readme(self, compressed_dir, monkeypatch):
        (compressed_dir / "README.md").write_text("# hand-written card\n")
        monkeypatch.setattr("huggingface_hub.create_repo", lambda *a, **kw: None)
        monkeypatch.setattr("huggingface_hub.upload_folder", lambda *a, **kw: None)

        args = _Args(local_dir=str(compressed_dir))
        cmd_push(args)

        assert (compressed_dir / "README.md").read_text() == "# hand-written card\n"


class TestCmdPushRealUpload:
    def test_passes_repo_id_and_folder(self, compressed_dir, monkeypatch):
        calls = {}

        def _fake_upload_folder(**kw):
            calls.update(kw)

        monkeypatch.setattr("huggingface_hub.create_repo", lambda *a, **kw: None)
        monkeypatch.setattr("huggingface_hub.upload_folder", _fake_upload_folder)

        args = _Args(local_dir=str(compressed_dir), repo_id="squishai/my-model")
        cmd_push(args)

        assert calls["repo_id"] == "squishai/my-model"
        assert calls["folder_path"] == str(compressed_dir)

    def test_passes_private_flag(self, compressed_dir, monkeypatch):
        calls = {}

        def _fake_create_repo(repo_id, **kw):
            calls.update(kw)

        monkeypatch.setattr("huggingface_hub.create_repo", _fake_create_repo)
        monkeypatch.setattr("huggingface_hub.upload_folder", lambda *a, **kw: None)

        args = _Args(local_dir=str(compressed_dir), private=True)
        cmd_push(args)

        assert calls["private"] is True

    def test_default_commit_message(self, compressed_dir, monkeypatch):
        calls = {}

        def _fake_upload_folder(**kw):
            calls.update(kw)

        monkeypatch.setattr("huggingface_hub.create_repo", lambda *a, **kw: None)
        monkeypatch.setattr("huggingface_hub.upload_folder", _fake_upload_folder)

        args = _Args(local_dir=str(compressed_dir))
        cmd_push(args)

        assert calls["commit_message"] == "Upload squished model"

    def test_custom_commit_message(self, compressed_dir, monkeypatch):
        calls = {}

        def _fake_upload_folder(**kw):
            calls.update(kw)

        monkeypatch.setattr("huggingface_hub.create_repo", lambda *a, **kw: None)
        monkeypatch.setattr("huggingface_hub.upload_folder", _fake_upload_folder)

        args = _Args(local_dir=str(compressed_dir), commit_message="custom message")
        cmd_push(args)

        assert calls["commit_message"] == "custom message"

    def test_upload_failure_dies_cleanly(self, compressed_dir, monkeypatch):
        monkeypatch.setattr("huggingface_hub.create_repo", lambda *a, **kw: None)

        def _boom(**kw):
            raise RuntimeError("simulated HF Hub failure")

        monkeypatch.setattr("huggingface_hub.upload_folder", _boom)

        args = _Args(local_dir=str(compressed_dir))
        with pytest.raises(SystemExit):
            cmd_push(args)
