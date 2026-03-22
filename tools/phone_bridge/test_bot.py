"""
tests for phone_bridge/bot.py — pure helper functions only.

All handlers depend on python-telegram-bot internals (Update, ContextTypes)
and are tested via the helper layer.  The helpers are fully deterministic
and have no I/O dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── isolate module from real env vars before import ───────────────────────────

_FAKE_ENV = {
    "TELEGRAM_BOT_TOKEN": "0:fake_token",
    "ALLOWED_USER_IDS": "111,222",
    "WORKING_DIR": "/tmp",
}


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, val in _FAKE_ENV.items():
        monkeypatch.setenv(key, val)


# Import AFTER patching env so module-level constants use fake values.
with patch.dict(os.environ, _FAKE_ENV):
    sys.path.insert(0, str(Path(__file__).parent))
    import bot  # noqa: E402  (must come after env patch)


# ── is_dangerous ──────────────────────────────────────────────────────────────


class TestIsDangerous:
    # ── should block ─────────────────────────────────────────────────────────

    def test_rm_recursive_long(self) -> None:
        assert bot.is_dangerous("rm -rf /tmp/foo")

    def test_rm_recursive_combined_flags(self) -> None:
        assert bot.is_dangerous("rm -Rf /home/user")

    def test_rm_recursive_flag_order(self) -> None:
        assert bot.is_dangerous("rm -fr .")

    def test_dd_basic(self) -> None:
        assert bot.is_dangerous("dd if=/dev/zero of=/dev/sda bs=1M")

    def test_dd_standalone(self) -> None:
        assert bot.is_dangerous("dd if=/dev/urandom of=file")

    def test_mkfs(self) -> None:
        assert bot.is_dangerous("mkfs.ext4 /dev/sdb1")

    def test_redirect_to_block_device_sda(self) -> None:
        assert bot.is_dangerous("cat /dev/urandom > /dev/sda")

    def test_redirect_to_disk_macos(self) -> None:
        assert bot.is_dangerous("cat file > /dev/disk2")

    def test_truncate_root(self) -> None:
        assert bot.is_dangerous(":> /etc/passwd")

    def test_sudo(self) -> None:
        assert bot.is_dangerous("sudo reboot")

    def test_sudo_embedded(self) -> None:
        assert bot.is_dangerous("echo foo | sudo tee /etc/hosts")

    def test_shred(self) -> None:
        assert bot.is_dangerous("shred -uz secret.key")

    def test_wipe(self) -> None:
        assert bot.is_dangerous("wipe -rf /home/old")

    def test_chmod_world_writable_777(self) -> None:
        assert bot.is_dangerous("chmod 777 /etc/shadow")

    def test_chmod_world_writable_776(self) -> None:
        assert bot.is_dangerous("chmod 776 myfile")

    # ── should allow ─────────────────────────────────────────────────────────

    def test_safe_rm_single_file(self) -> None:
        assert not bot.is_dangerous("rm file.txt")

    def test_safe_ls(self) -> None:
        assert not bot.is_dangerous("ls -la /tmp")

    def test_safe_git_status(self) -> None:
        assert not bot.is_dangerous("git status --short")

    def test_safe_echo(self) -> None:
        assert not bot.is_dangerous("echo hello world")

    def test_safe_python(self) -> None:
        assert not bot.is_dangerous("python -m pytest")

    def test_safe_chmod_644(self) -> None:
        assert not bot.is_dangerous("chmod 644 file.txt")

    def test_safe_chmod_755(self) -> None:
        assert not bot.is_dangerous("chmod 755 script.sh")

    def test_safe_redirect_to_file(self) -> None:
        assert not bot.is_dangerous("echo hi > /tmp/out.txt")

    def test_empty_string(self) -> None:
        assert not bot.is_dangerous("")

    def test_safe_dd_word_in_variable_name(self) -> None:
        # "dd" as a substring of a normal word should not match because
        # the regex uses \b word boundaries.
        assert not bot.is_dangerous("address")

    def test_safe_sudo_as_substring(self) -> None:
        assert not bot.is_dangerous("pseudocode analysis")


# ── truncate ──────────────────────────────────────────────────────────────────


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        text = "hello world"
        assert bot.truncate(text, limit=100) == text

    def test_exactly_at_limit(self) -> None:
        text = "a" * 100
        assert bot.truncate(text, limit=100) == text

    def test_one_over_limit(self) -> None:
        # Use a large enough overage so the truncated result (head + notice + tail)
        # is shorter than the original text.
        text = "a" * 200
        result = bot.truncate(text, limit=100)
        assert "omitted" in result
        assert len(result) < len(text)

    def test_preserves_head_and_tail(self) -> None:
        text = "HEAD" + "x" * 10000 + "TAIL"
        result = bot.truncate(text, limit=20)
        assert result.startswith("HEAD")
        assert result.endswith("TAIL")

    def test_omitted_count_is_accurate(self) -> None:
        text = "a" * 200
        limit = 100
        result = bot.truncate(text, limit=limit)
        omitted = len(text) - limit  # 100
        assert str(omitted) in result

    def test_empty_string(self) -> None:
        assert bot.truncate("", limit=50) == ""

    def test_default_limit_applied(self) -> None:
        # Default is MAX_MESSAGE_CHARS; anything at or below passes unchanged.
        text = "x" * bot.MAX_MESSAGE_CHARS
        assert bot.truncate(text) == text

    def test_over_default_limit_truncated(self) -> None:
        text = "x" * (bot.MAX_MESSAGE_CHARS * 2)
        assert len(bot.truncate(text)) < len(text)


# ── get_cwd ───────────────────────────────────────────────────────────────────


class TestGetCwd:
    def setup_method(self) -> None:
        # Clear session state between tests.
        bot._session_cwd.clear()

    def test_returns_default_for_unknown_user(self) -> None:
        assert bot.get_cwd(99999) == bot._DEFAULT_CWD

    def test_returns_session_cwd_when_set(self, tmp_path: Path) -> None:
        bot._session_cwd[42] = tmp_path
        assert bot.get_cwd(42) == tmp_path

    def test_different_users_have_independent_cwds(self, tmp_path: Path) -> None:
        bot._session_cwd[1] = tmp_path / "user1"
        assert bot.get_cwd(2) == bot._DEFAULT_CWD

    def test_session_cleared_returns_default(self, tmp_path: Path) -> None:
        bot._session_cwd[7] = tmp_path
        bot._session_cwd.clear()
        assert bot.get_cwd(7) == bot._DEFAULT_CWD


# ── format_pre ────────────────────────────────────────────────────────────────


class TestFormatPre:
    def test_wraps_in_pre_tags(self) -> None:
        result = bot.format_pre("hello")
        assert result.startswith("<pre>")
        assert result.endswith("</pre>")

    def test_escapes_html_angle_brackets(self) -> None:
        result = bot.format_pre("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escapes_ampersand(self) -> None:
        result = bot.format_pre("a & b")
        assert "&amp;" in result

    def test_escapes_double_quote(self) -> None:
        result = bot.format_pre('say "hi"')
        assert "&quot;" in result

    def test_truncates_before_escaping(self) -> None:
        # A very long string should be truncated, not cause an enormous message.
        long_text = "z" * (bot.MAX_MESSAGE_CHARS * 2)
        result = bot.format_pre(long_text)
        assert len(result) < len(long_text)


# ── module constants ──────────────────────────────────────────────────────────


class TestModuleConstants:
    def test_allowed_user_ids_parsed(self) -> None:
        assert 111 in bot._ALLOWED_USER_IDS
        assert 222 in bot._ALLOWED_USER_IDS

    def test_allowed_user_ids_is_frozenset(self) -> None:
        assert isinstance(bot._ALLOWED_USER_IDS, frozenset)

    def test_default_cwd_is_path(self) -> None:
        assert isinstance(bot._DEFAULT_CWD, Path)

    def test_default_cwd_exists(self) -> None:
        # /tmp is guaranteed to exist on macOS/Linux.
        assert bot._DEFAULT_CWD.exists()

    def test_max_message_chars_reasonable(self) -> None:
        # Must be positive and below Telegram's 4096 hard limit.
        assert 0 < bot.MAX_MESSAGE_CHARS < 4096

    def test_command_timeout_positive(self) -> None:
        assert bot.COMMAND_TIMEOUT_S > 0

    def test_copilot_timeout_gte_command_timeout(self) -> None:
        assert bot.COPILOT_TIMEOUT_S >= bot.COMMAND_TIMEOUT_S


# ── is_authorized ─────────────────────────────────────────────────────────────


class TestIsAuthorized:
    def _make_update(self, user_id: int | None) -> MagicMock:
        update = MagicMock()
        if user_id is None:
            update.effective_user = None
        else:
            update.effective_user = MagicMock()
            update.effective_user.id = user_id
        return update

    def test_allowed_user_is_authorized(self) -> None:
        assert bot.is_authorized(self._make_update(111))

    def test_second_allowed_user(self) -> None:
        assert bot.is_authorized(self._make_update(222))

    def test_unknown_user_not_authorized(self) -> None:
        assert not bot.is_authorized(self._make_update(99999))

    def test_none_user_not_authorized(self) -> None:
        assert not bot.is_authorized(self._make_update(None))
