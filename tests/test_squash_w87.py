"""Wave 87 tests — QTIP/YAQA pre-quantized load-only pathway."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
