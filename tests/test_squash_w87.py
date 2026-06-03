"""Wave 87 tests — QTIP/YAQA pre-quantized load-only pathway."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from squish.squash.cli import _cmd_load_qtip
from squish.squash.qtip_loader import load_qtip_checkpoint, load_yaqa_checkpoint


class _FakeModelClass:
    last_call: dict[str, object] | None = None

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        cls.last_call = {"path": path, **kwargs}
        return SimpleNamespace(kind="model", path=path)


class _FakeTokenizerClass:
    last_call: dict[str, object] | None = None

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        cls.last_call = {"path": path, **kwargs}
        return SimpleNamespace(kind="tokenizer", path=path)


def test_load_qtip_checkpoint_returns_handle(tmp_path):
    target = tmp_path / "hf-snapshot"
    target.mkdir()

    def _snapshot_download(**kwargs):
        assert kwargs["repo_id"] == "org/qtip-model"
        return str(target)

    with patch("squish.squash.qtip_loader._import_snapshot_download", return_value=_snapshot_download):
        with patch(
            "squish.squash.qtip_loader._import_transformers",
            return_value=(_FakeModelClass, _FakeTokenizerClass),
        ):
            handle = load_qtip_checkpoint("org/qtip-model")

    assert handle.family == "qtip"
    assert handle.repo_id == "org/qtip-model"
    assert handle.local_path == target
    assert handle.model.kind == "model"
    assert handle.tokenizer.kind == "tokenizer"


def test_load_qtip_checkpoint_forwards_optional_args(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    local_dir = tmp_path / "cache"
    calls: dict[str, dict[str, object]] = {}

    def _snapshot_download(**kwargs):
        calls["snapshot"] = kwargs
        return str(snapshot_dir)

    with patch("squish.squash.qtip_loader._import_snapshot_download", return_value=_snapshot_download):
        with patch(
            "squish.squash.qtip_loader._import_transformers",
            return_value=(_FakeModelClass, _FakeTokenizerClass),
        ):
            load_qtip_checkpoint(
                "org/model",
                local_dir=local_dir,
                hf_token="hf_x",
                revision="main",
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype="float16",
            )

    assert calls["snapshot"]["local_dir"] == str(local_dir)
    assert calls["snapshot"]["token"] == "hf_x"
    assert calls["snapshot"]["revision"] == "main"
    assert _FakeModelClass.last_call is not None
    assert _FakeModelClass.last_call["trust_remote_code"] is True
    assert _FakeModelClass.last_call["device_map"] == "cpu"
    assert _FakeModelClass.last_call["torch_dtype"] == "float16"


def test_load_yaqa_checkpoint_marks_family(tmp_path):
    target = tmp_path / "yaqa-snapshot"
    target.mkdir()

    with patch(
        "squish.squash.qtip_loader._import_snapshot_download",
        return_value=lambda **_: str(target),
    ):
        with patch(
            "squish.squash.qtip_loader._import_transformers",
            return_value=(_FakeModelClass, _FakeTokenizerClass),
        ):
            handle = load_yaqa_checkpoint("org/yaqa-model")

    assert handle.family == "yaqa"


def test_load_qtip_checkpoint_empty_repo_raises_value_error():
    with pytest.raises(ValueError):
        load_qtip_checkpoint("   ")


def test_load_qtip_checkpoint_logs_license_warning(tmp_path, caplog):
    target = tmp_path / "snapshot"
    target.mkdir()

    with patch(
        "squish.squash.qtip_loader._import_snapshot_download",
        return_value=lambda **_: str(target),
    ):
        with patch(
            "squish.squash.qtip_loader._import_transformers",
            return_value=(_FakeModelClass, _FakeTokenizerClass),
        ):
            caplog.set_level("WARNING")
            load_qtip_checkpoint("org/model")

    assert "squish does not distribute GPL code" in caplog.text


def test_cmd_load_qtip_success_outputs_json(tmp_path, capsys):
    args = argparse.Namespace(
        hf_repo="org/model",
        variant="qtip",
        local_dir=str(tmp_path / "cache"),
        hf_token="hf_test",
        revision="main",
        trust_remote_code=False,
        device_map="auto",
        torch_dtype="auto",
    )

    fake_handle = SimpleNamespace(
        family="qtip",
        repo_id="org/model",
        local_path=tmp_path / "snapshot",
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
    )

    with patch("squish.squash.qtip_loader.load_qtip_checkpoint", return_value=fake_handle) as loader:
        rc = _cmd_load_qtip(args, quiet=True)

    assert rc == 0
    loader.assert_called_once()
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "loaded"
    assert payload["variant"] == "qtip"
    assert payload["repo_id"] == "org/model"


def test_cmd_load_qtip_uses_yaqa_variant(tmp_path):
    args = argparse.Namespace(
        hf_repo="org/yaqa",
        variant="yaqa",
        local_dir=None,
        hf_token=None,
        revision=None,
        trust_remote_code=False,
        device_map="auto",
        torch_dtype="auto",
    )

    fake_handle = SimpleNamespace(
        family="yaqa",
        repo_id="org/yaqa",
        local_path=tmp_path / "snapshot",
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
    )

    with patch("squish.squash.qtip_loader.load_yaqa_checkpoint", return_value=fake_handle) as yaqa_loader:
        with patch("squish.squash.qtip_loader.load_qtip_checkpoint") as qtip_loader:
            rc = _cmd_load_qtip(args, quiet=True)

    assert rc == 0
    yaqa_loader.assert_called_once()
    qtip_loader.assert_not_called()


def test_cmd_load_qtip_uses_hf_token_env(tmp_path):
    args = argparse.Namespace(
        hf_repo="org/model",
        variant="qtip",
        local_dir=None,
        hf_token=None,
        revision=None,
        trust_remote_code=False,
        device_map="auto",
        torch_dtype="auto",
    )

    fake_handle = SimpleNamespace(
        family="qtip",
        repo_id="org/model",
        local_path=tmp_path / "snapshot",
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
    )

    with patch.dict(os.environ, {"HF_TOKEN": "from-env"}, clear=False):
        with patch("squish.squash.qtip_loader.load_qtip_checkpoint", return_value=fake_handle) as loader:
            rc = _cmd_load_qtip(args, quiet=True)

    assert rc == 0
    assert loader.call_args.kwargs["hf_token"] == "from-env"


def test_cmd_load_qtip_returns_2_on_loader_error(capsys):
    args = argparse.Namespace(
        hf_repo="org/model",
        variant="qtip",
        local_dir=None,
        hf_token=None,
        revision=None,
        trust_remote_code=False,
        device_map="auto",
        torch_dtype="auto",
    )

    with patch("squish.squash.qtip_loader.load_qtip_checkpoint", side_effect=RuntimeError("boom")):
        rc = _cmd_load_qtip(args, quiet=True)

    assert rc == 2
    assert "load-qtip failed" in capsys.readouterr().err


def test_cli_help_includes_load_qtip_command():
    proc = subprocess.run(
        [sys.executable, "-m", "squish.squash.cli", "load-qtip", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "usage: squash load-qtip" in proc.stdout
    assert "--variant" in proc.stdout
    assert "--trust-remote-code" in proc.stdout
