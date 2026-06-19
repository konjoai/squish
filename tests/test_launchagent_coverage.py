"""Behavioral coverage for ``squish.daemon.launchagent`` — macOS LaunchAgent
plist generation + install/uninstall/status lifecycle.

macOS-gating, the plist paths, and ``launchctl`` subprocess calls are all
controlled via monkeypatch, so the tests run host-agnostically (they never
touch the real ~/Library/LaunchAgents or invoke launchctl).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from squish.daemon import launchagent as la


@pytest.fixture
def macos(monkeypatch, tmp_path):
    """Force macOS and redirect the plist path into tmp_path."""
    monkeypatch.setattr(la, "_is_macos", lambda: True)
    plist = tmp_path / "LaunchAgents" / "ai.konjo.squishd.plist"
    monkeypatch.setattr(la, "PLIST_DIR", plist.parent)
    monkeypatch.setattr(la, "PLIST_PATH", plist)
    return plist


def _ok_run(monkeypatch):
    monkeypatch.setattr(la.subprocess, "run", lambda *a, **k: None)


# ── _is_macos ───────────────────────────────────────────────────────────────


def test_is_macos_true_and_false(monkeypatch):
    monkeypatch.setattr(la.platform, "system", lambda: "Darwin")
    assert la._is_macos() is True
    monkeypatch.setattr(la.platform, "system", lambda: "Linux")
    assert la._is_macos() is False


# ── plist_content / _xml_args ───────────────────────────────────────────────


def test_plist_content_includes_args_and_paths():
    xml = la.plist_content(squishd_bin="/bin/squishd", model_dir="/m",
                           compressed_dir="/c", sock_path="/tmp/s.sock",
                           max_models=3, log_path="/var/log/x")
    assert "<string>/bin/squishd</string>" in xml
    assert "--compressed-dir" in xml and "<string>/c</string>" in xml  # 118
    assert "<string>/var/log/x</string>" in xml
    assert "ai.konjo.squishd" in xml


def test_plist_content_without_model_dir_skips_model_args():
    xml = la.plist_content(squishd_bin="/bin/squishd")
    assert "--compressed-dir" not in xml
    assert "<string>--sock</string>" in xml  # sock/max-models still present


def test_plist_content_default_log_path():
    xml = la.plist_content(squishd_bin="/bin/squishd")  # log_path="" → squish_home default
    assert "daemon.log" in xml


# ── install ─────────────────────────────────────────────────────────────────


def test_install_non_macos_raises(monkeypatch):
    monkeypatch.setattr(la, "_is_macos", lambda: False)
    with pytest.raises(RuntimeError, match="macOS-only"):
        la.install()


def test_install_writes_plist_and_loads(macos, monkeypatch, tmp_path):
    bin_path = tmp_path / "squishd"
    bin_path.write_text("#!/bin/sh\n")
    calls = []
    monkeypatch.setattr(la.subprocess, "run", lambda *a, **k: calls.append(a))
    out = la.install(model_dir="/m", squishd_bin=str(bin_path))
    assert out == str(macos)
    assert macos.exists() and "ai.konjo.squishd" in macos.read_text()
    assert calls  # launchctl load was invoked


def test_install_resolves_bin_via_which(macos, monkeypatch, tmp_path):
    bin_path = tmp_path / "found_squishd"
    bin_path.write_text("x")
    monkeypatch.setattr(la.shutil if hasattr(la, "shutil") else __import__("shutil"),
                        "which", lambda name: str(bin_path))
    _ok_run(monkeypatch)
    out = la.install()  # no explicit bin → resolved via shutil.which
    assert out == str(macos)


def test_install_missing_bin_raises(macos, monkeypatch, tmp_path):
    import sys
    monkeypatch.setattr(__import__("shutil"), "which", lambda name: None)
    # which → None → derive from sys.executable; point it at a dir with no
    # squishd so the derived path doesn't exist → FileNotFoundError (151-155).
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    with pytest.raises(FileNotFoundError, match="squishd binary not found"):
        la.install()


def test_install_launchctl_failure_warns(macos, monkeypatch, tmp_path):
    bin_path = tmp_path / "squishd"
    bin_path.write_text("x")

    def _boom(*a, **k):
        raise subprocess.CalledProcessError(1, "launchctl", stderr=b"nope")

    monkeypatch.setattr(la.subprocess, "run", _boom)
    with pytest.warns(UserWarning, match="launchctl load returned non-zero"):
        out = la.install(squishd_bin=str(bin_path))
    assert out == str(macos)  # plist still installed despite load failure


# ── uninstall ───────────────────────────────────────────────────────────────


def test_uninstall_non_macos_raises(monkeypatch):
    monkeypatch.setattr(la, "_is_macos", lambda: False)
    with pytest.raises(RuntimeError, match="macOS-only"):
        la.uninstall()


def test_uninstall_removes_plist(macos, monkeypatch):
    macos.parent.mkdir(parents=True, exist_ok=True)
    macos.write_text("<plist/>")
    _ok_run(monkeypatch)
    la.uninstall()
    assert not macos.exists()


def test_uninstall_no_plist_is_noop(macos, monkeypatch):
    _ok_run(monkeypatch)
    la.uninstall()  # PLIST_PATH absent → nothing to do
    assert not macos.exists()


def test_uninstall_swallows_launchctl_error(macos, monkeypatch):
    macos.parent.mkdir(parents=True, exist_ok=True)
    macos.write_text("<plist/>")
    monkeypatch.setattr(la.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("launchctl gone")))
    la.uninstall()  # OSError swallowed (204-205); plist still removed
    assert not macos.exists()


# ── is_installed / status ───────────────────────────────────────────────────


def test_is_installed(macos):
    assert la.is_installed() is False
    macos.parent.mkdir(parents=True, exist_ok=True)
    macos.write_text("<plist/>")
    assert la.is_installed() is True


def test_status_not_installed(macos):
    st = la.status()
    assert st["installed"] is False and st["loaded"] is False
    assert st["plist_path"] is None


def test_status_installed_and_loaded(macos, monkeypatch):
    macos.parent.mkdir(parents=True, exist_ok=True)
    macos.write_text("<plist/>")
    monkeypatch.setattr(la.subprocess, "check_output", lambda *a, **k: b'{"PID" = 123;}')
    st = la.status()
    assert st["installed"] is True and st["loaded"] is True
    assert st["plist_path"] == str(macos)


def test_status_installed_not_loaded(macos, monkeypatch):
    macos.parent.mkdir(parents=True, exist_ok=True)
    macos.write_text("<plist/>")

    def _boom(*a, **k):
        raise subprocess.CalledProcessError(1, "launchctl")

    monkeypatch.setattr(la.subprocess, "check_output", _boom)
    st = la.status()
    assert st["installed"] is True and st["loaded"] is False
