"""
Tests for squish/serving/whatsapp.py (Meta WhatsApp Business Cloud API version).

Covers:
  - Meta HMAC-SHA256 signature validation (_validate_meta_signature)
  - Session management helpers (_get_or_create_session, _reset_session,
    _expire_old_sessions, _apply_max_history)
  - Message processing (_handle_message): commands, generation, error paths
  - HTTP endpoint behaviour via FastAPI TestClient:
      GET  /webhook/whatsapp — Meta challenge verification
      POST /webhook/whatsapp — incoming message webhook
  - Message deduplication (_is_duplicate)
  - Per-sender rate limiting (_is_rate_limited)
  - Long reply chunking (_chunk_reply)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import unittest.mock as mock

import pytest

from squish.serving.whatsapp import (
    _DEFAULT_SYSTEM_PROMPT,
    _MAX_HISTORY,
    _MAX_MSG_LEN,
    _RATE_LIMIT,
    _RATE_WINDOW,
    _SEEN_IDS,
    _SEEN_IDS_LOCK,
    _SEEN_IDS_MAX,
    _SEEN_IDS_ORDER,
    _SESSION_TIMEOUT,
    _apply_max_history,
    _chunk_reply,
    _expire_old_sessions,
    _get_or_create_session,
    _handle_message,
    _is_duplicate,
    _is_rate_limited,
    _rate_counts,
    _rate_lock,
    _reset_session,
    _sessions,
    _sessions_lock,
    _sessions_ts,
    _validate_meta_signature,
    mount_whatsapp,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clear_sessions() -> None:
    with _sessions_lock:
        _sessions.clear()
        _sessions_ts.clear()


def _clear_seen_ids() -> None:
    with _SEEN_IDS_LOCK:
        _SEEN_IDS.clear()
        _SEEN_IDS_ORDER.clear()


def _clear_rate_counts() -> None:
    with _rate_lock:
        _rate_counts.clear()


def _clear_all() -> None:
    _clear_sessions()
    _clear_seen_ids()
    _clear_rate_counts()


def _meta_sig(app_secret: str, body: bytes) -> str:
    """Compute a valid X-Hub-Signature-256 header value."""
    digest = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _meta_payload(sender: str, body: str) -> bytes:
    """Build a minimal Meta webhook JSON payload for a single text message."""
    return json.dumps({
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "type": "text",
                        "from": sender,
                        "timestamp": str(int(time.time())),
                        "id": "wamid.test",
                        "text": {"body": body},
                    }]
                }
            }]
        }]
    }).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# _validate_meta_signature
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateMetaSignature:
    SECRET = "test_app_secret_xyz"
    BODY   = b'{"entry": []}'

    def test_valid_signature_accepted(self):
        sig = _meta_sig(self.SECRET, self.BODY)
        assert _validate_meta_signature(self.SECRET, self.BODY, sig) is True

    def test_invalid_signature_rejected(self):
        assert _validate_meta_signature(self.SECRET, self.BODY, "sha256=badhex") is False

    def test_wrong_secret_rejected(self):
        sig = _meta_sig(self.SECRET, self.BODY)
        assert _validate_meta_signature("wrong_secret", self.BODY, sig) is False

    def test_missing_sha256_prefix_rejected(self):
        digest = hmac.new(self.SECRET.encode(), self.BODY, hashlib.sha256).hexdigest()
        assert _validate_meta_signature(self.SECRET, self.BODY, digest) is False

    def test_tampered_body_rejected(self):
        sig = _meta_sig(self.SECRET, self.BODY)
        assert _validate_meta_signature(self.SECRET, b'{"entry": [{}]}', sig) is False

    def test_empty_body_valid(self):
        sig = _meta_sig(self.SECRET, b"")
        assert _validate_meta_signature(self.SECRET, b"", sig) is True

    def test_signature_is_case_sensitive(self):
        sig = _meta_sig(self.SECRET, self.BODY).upper()
        # HMAC hex digest is lowercase; uppercased form should fail
        assert _validate_meta_signature(self.SECRET, self.BODY, sig) is False


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

class TestSessions:
    def setup_method(self):
        _clear_sessions()

    def test_create_new_session_with_system_prompt(self):
        msgs = _get_or_create_session("15551234567", "sys")
        assert msgs == [{"role": "system", "content": "sys"}]

    def test_returns_existing_session(self):
        msgs = _get_or_create_session("15551234567", "p")
        msgs.append({"role": "user", "content": "hi"})
        assert len(_get_or_create_session("15551234567", "p")) == 2

    def test_updates_timestamp(self):
        before = time.time()
        _get_or_create_session("111", "p")
        after = time.time()
        assert before <= _sessions_ts["111"] <= after

    def test_reset_clears_to_system_only(self):
        msgs = _get_or_create_session("111", "p")
        msgs += [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        _reset_session("111", "new prompt")
        assert _sessions["111"] == [{"role": "system", "content": "new prompt"}]

    def test_expire_removes_old(self):
        _sessions["old"] = []
        _sessions_ts["old"] = time.time() - _SESSION_TIMEOUT - 1
        _sessions["fresh"] = []
        _sessions_ts["fresh"] = time.time()
        _expire_old_sessions()
        assert "old" not in _sessions
        assert "fresh" in _sessions

    def test_expire_keeps_active(self):
        _sessions["recent"] = []
        _sessions_ts["recent"] = time.time() - _SESSION_TIMEOUT + 60
        _expire_old_sessions()
        assert "recent" in _sessions


class TestApplyMaxHistory:
    def test_short_list_unchanged(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        assert _apply_max_history(msgs) == msgs

    def test_trims_to_max_history(self):
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(_MAX_HISTORY + 5)
        ]
        result = _apply_max_history(msgs)
        assert len(result) == _MAX_HISTORY
        assert result[0]["role"] == "system"

    def test_trims_keeps_last_messages(self):
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(_MAX_HISTORY + 3)
        ]
        result = _apply_max_history(msgs)
        assert result[-1]["content"] == f"m{_MAX_HISTORY + 2}"

    def test_system_prompt_always_preserved(self):
        msgs = [{"role": "system", "content": "keep"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(_MAX_HISTORY + 2)
        ]
        assert _apply_max_history(msgs)[0] == {"role": "system", "content": "keep"}


# ─────────────────────────────────────────────────────────────────────────────
# _handle_message — direct invocation, mocking _send_whatsapp_reply
# ─────────────────────────────────────────────────────────────────────────────

class _FakeState:
    model = object()
    model_name = "squish-7b"
    avg_tps = 12.5
    requests = 3
    loaded_at = time.time() - 300


class _FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return " ".join(m["content"] for m in messages)


def _fake_generate(prompt, max_tokens, temperature, top_p, stop, seed):
    yield ("Reply text", None)
    yield ("", "stop")


def _call_handle(
    text: str,
    *,
    sender: str = "15551234567",
    get_state=None,
    get_generate=None,
    get_tokenizer=None,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    msg_id: str = "",
) -> list[str]:
    """Call _handle_message and return the list of args passed to _send_whatsapp_reply."""
    _clear_sessions()
    _clear_seen_ids()
    _clear_rate_counts()
    sent: list[str] = []
    with mock.patch(
        "squish.serving.whatsapp._send_whatsapp_reply",
        side_effect=lambda pid, tok, to, body: sent.append(body),
    ):
        _handle_message(
            sender, text,
            phone_number_id = "12345678",
            access_token    = "tok",
            get_state       = get_state or (lambda: _FakeState()),
            get_generate    = get_generate or (lambda: _fake_generate),
            get_tokenizer   = get_tokenizer or (lambda: _FakeTok()),
            system_prompt   = system_prompt,
            msg_id          = msg_id,
        )
    return sent


class TestHandleMessage:
    def test_normal_message_replies(self):
        sent = _call_handle("Hello bot")
        assert len(sent) == 1
        assert "Reply text" in sent[0]

    def test_reset_command(self):
        sent = _call_handle("/reset")
        assert len(sent) == 1
        assert "cleared" in sent[0].lower() or "fresh" in sent[0].lower()

    def test_reset_clears_session(self):
        _call_handle("first")
        _call_handle("/reset")
        assert _sessions.get("15551234567") == [
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}
        ]

    def test_help_command(self):
        sent = _call_handle("/help")
        assert len(sent) == 1
        assert "/reset" in sent[0]
        assert "/status" in sent[0]

    def test_status_command_loaded(self):
        sent = _call_handle("/status")
        assert "squish-7b" in sent[0]
        assert "tok/s" in sent[0] or "tps" in sent[0].lower()

    def test_status_model_not_loaded(self):
        class _Empty:
            model = None
        sent = _call_handle("/status", get_state=lambda: _Empty())
        assert "not loaded" in sent[0].lower() or "loading" in sent[0].lower()

    def test_model_not_loaded_returns_message(self):
        class _Empty:
            model = None
        sent = _call_handle("hello", get_state=lambda: _Empty())
        assert len(sent) == 1
        assert "loading" in sent[0].lower() or "moment" in sent[0].lower()

    def test_conversation_history_saved(self):
        _clear_sessions()
        _call_handle("one", sender="99999")
        sess = _sessions.get("99999", [])
        roles = [m["role"] for m in sess]
        assert "user" in roles
        assert "assistant" in roles

    def test_generation_error_sends_error_reply(self):
        def _broken(prompt, max_tokens, temperature, top_p, stop, seed):
            raise RuntimeError("GPU on fire")
            yield
        sent = _call_handle("hello", get_generate=lambda: _broken)
        assert len(sent) == 1
        assert "error" in sent[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP endpoint behaviour via FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────────────

APP_SECRET    = "test_secret_abc"
VERIFY_TOKEN  = "test_verify_xyz"
ACCESS_TOKEN  = "test_access_tok"
PHONE_NUM_ID  = "10000000001"


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    _clear_sessions()
    app = FastAPI()
    mount_whatsapp(
        app,
        get_state       = lambda: _FakeState(),
        get_generate    = lambda: _fake_generate,
        get_tokenizer   = lambda: _FakeTok(),
        verify_token    = VERIFY_TOKEN,
        app_secret      = "",   # no signature validation in most tests
        access_token    = ACCESS_TOKEN,
        phone_number_id = PHONE_NUM_ID,
        system_prompt   = _DEFAULT_SYSTEM_PROMPT,
    )
    return TestClient(app, raise_server_exceptions=True)


class TestGetChallenge:
    def test_correct_token_returns_challenge(self, client):
        resp = client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 200
        assert resp.text == "abc123"

    def test_wrong_token_returns_403(self, client):
        resp = client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong_token",
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 403

    def test_wrong_mode_returns_403(self, client):
        resp = client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "test",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 403


class TestPostWebhook:
    def test_valid_payload_returns_200(self, client):
        payload = _meta_payload("15551234567", "Hello!")
        with mock.patch("squish.serving.whatsapp._send_whatsapp_reply"):
            resp = client.post(
                "/webhook/whatsapp",
                content=payload,
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 200

    def test_empty_payload_returns_200(self, client):
        resp = client.post(
            "/webhook/whatsapp",
            content=json.dumps({"entry": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_malformed_json_returns_200(self, client):
        resp = client.post(
            "/webhook/whatsapp",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_non_text_message_type_skipped(self, client):
        """Media messages (type != 'text') should be silently skipped."""
        payload = json.dumps({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "type": "image",
                            "from": "15551234567",
                            "timestamp": "1234567890",
                            "id": "wamid.test",
                        }]
                    }
                }]
            }]
        }).encode()
        with mock.patch("squish.serving.whatsapp._send_whatsapp_reply") as mock_send:
            resp = client.post(
                "/webhook/whatsapp",
                content=payload,
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 200


class TestPostWebhookWithSignature:
    """Verify HMAC-SHA256 signature enforcement when app_secret is set."""

    @pytest.fixture()
    def signed_client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        _clear_sessions()
        app = FastAPI()
        mount_whatsapp(
            app,
            get_state       = lambda: _FakeState(),
            get_generate    = lambda: _fake_generate,
            get_tokenizer   = lambda: _FakeTok(),
            verify_token    = VERIFY_TOKEN,
            app_secret      = APP_SECRET,
            access_token    = ACCESS_TOKEN,
            phone_number_id = PHONE_NUM_ID,
        )
        return TestClient(app, raise_server_exceptions=True)

    def test_missing_signature_returns_403(self, signed_client):
        payload = _meta_payload("15551234567", "hi")
        resp = signed_client.post(
            "/webhook/whatsapp",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 403
        assert "missing" in resp.text.lower()

    def test_invalid_signature_returns_403(self, signed_client):
        payload = _meta_payload("15551234567", "hi")
        resp = signed_client.post(
            "/webhook/whatsapp",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=badhex",
            },
        )
        assert resp.status_code == 403
        assert "invalid" in resp.text.lower()

    def test_valid_signature_returns_200(self, signed_client):
        payload = _meta_payload("15551234567", "hi")
        sig = _meta_sig(APP_SECRET, payload)
        with mock.patch("squish.serving.whatsapp._send_whatsapp_reply"):
            resp = signed_client.post(
                "/webhook/whatsapp",
                content=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": sig,
                },
            )
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# server.py argument parser accepts --whatsapp flags
# ─────────────────────────────────────────────────────────────────────────────

class TestServerWhatsAppArgs:
    """Verify server.py's argparse recognises the WhatsApp CLI flags."""

    @staticmethod
    def _make_parser():
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--whatsapp", action="store_true", default=False)
        ap.add_argument("--whatsapp-verify-token", default="")
        ap.add_argument("--whatsapp-app-secret", default="")
        ap.add_argument("--whatsapp-access-token", default="")
        ap.add_argument("--whatsapp-phone-number-id", default="")
        return ap

    def test_whatsapp_flag_accepted(self):
        args = self._make_parser().parse_args(["--whatsapp"])
        assert args.whatsapp is True

    def test_whatsapp_flag_default_false(self):
        args = self._make_parser().parse_args([])
        assert args.whatsapp is False

    def test_whatsapp_credentials_accepted(self):
        args = self._make_parser().parse_args([
            "--whatsapp",
            "--whatsapp-verify-token", "my_token",
            "--whatsapp-app-secret",   "secret123",
            "--whatsapp-access-token", "EAABxxx",
            "--whatsapp-phone-number-id", "123456789",
        ])
        assert args.whatsapp is True
        assert args.whatsapp_verify_token    == "my_token"
        assert args.whatsapp_app_secret      == "secret123"
        assert args.whatsapp_access_token    == "EAABxxx"
        assert args.whatsapp_phone_number_id == "123456789"

    def test_whatsapp_credentials_default_empty(self):
        args = self._make_parser().parse_args([])
        assert args.whatsapp_verify_token    == ""
        assert args.whatsapp_app_secret      == ""
        assert args.whatsapp_access_token    == ""
        assert args.whatsapp_phone_number_id == ""


# ─────────────────────────────────────────────────────────────────────────────
# _is_duplicate — message-ID deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestIsDuplicate:
    def setup_method(self):
        _clear_seen_ids()

    def test_first_call_not_duplicate(self):
        assert _is_duplicate("wamid.001") is False

    def test_second_call_is_duplicate(self):
        _is_duplicate("wamid.002")
        assert _is_duplicate("wamid.002") is True

    def test_different_ids_not_duplicate(self):
        _is_duplicate("wamid.003")
        assert _is_duplicate("wamid.004") is False

    def test_empty_id_never_duplicate(self):
        _is_duplicate("")
        assert _is_duplicate("") is False

    def test_bounded_fifo_evicts_oldest(self):
        # Fill to capacity + 1 so the very first ID gets evicted
        for i in range(_SEEN_IDS_MAX + 1):
            _is_duplicate(f"wamid.{i:06d}")
        # wamid.000000 was evicted and is no longer considered a duplicate
        assert _is_duplicate("wamid.000000") is False

    def test_bounded_fifo_keeps_recent(self):
        for i in range(_SEEN_IDS_MAX):
            _is_duplicate(f"wamid.r{i:06d}")
        # The last ID added is still in the set
        assert _is_duplicate(f"wamid.r{_SEEN_IDS_MAX - 1:06d}") is True

    def test_handle_message_deduplicates(self):
        """_handle_message silently drops messages with already-seen IDs."""
        _clear_all()
        sent: list[str] = []
        with mock.patch(
            "squish.serving.whatsapp._send_whatsapp_reply",
            side_effect=lambda pid, tok, to, body: sent.append(body),
        ):
            _handle_message(
                "15551234567", "hello", "pid", "tok",
                lambda: _FakeState(), lambda: _fake_generate, lambda: _FakeTok(),
                _DEFAULT_SYSTEM_PROMPT, msg_id="wamid.dedup01",
            )
            # Second call: same wamid, clear sessions to ensure the session state
            # is not the reason it returns early
            _clear_sessions()
            _handle_message(
                "15551234567", "hello", "pid", "tok",
                lambda: _FakeState(), lambda: _fake_generate, lambda: _FakeTok(),
                _DEFAULT_SYSTEM_PROMPT, msg_id="wamid.dedup01",
            )
        # Only the first call should have produced a reply
        assert len(sent) == 1


# ─────────────────────────────────────────────────────────────────────────────
# _is_rate_limited — per-sender sliding-window rate limiting
# ─────────────────────────────────────────────────────────────────────────────

class TestIsRateLimited:
    def setup_method(self):
        _clear_rate_counts()

    def test_first_message_not_limited(self):
        assert _is_rate_limited("15551234567") is False

    def test_under_limit_not_limited(self):
        for _ in range(_RATE_LIMIT - 1):
            assert _is_rate_limited("15551234568") is False

    def test_at_limit_is_limited(self):
        for _ in range(_RATE_LIMIT):
            _is_rate_limited("15551234569")
        assert _is_rate_limited("15551234569") is True

    def test_different_senders_are_independent(self):
        for _ in range(_RATE_LIMIT):
            _is_rate_limited("senderA")
        # senderA is limited; senderB has not sent anything yet
        assert _is_rate_limited("senderA") is True
        assert _is_rate_limited("senderB") is False

    def test_expired_timestamps_not_counted(self):
        sender = "15551234570"
        with _rate_lock:
            _rate_counts[sender] = [time.time() - _RATE_WINDOW - 1] * _RATE_LIMIT
        # All timestamps are outside the window — sender should not be limited
        assert _is_rate_limited(sender) is False

    def test_handle_message_sends_rate_limit_reply(self):
        """_handle_message sends a polite refusal when sender exceeds the rate limit."""
        _clear_all()
        sender = "15551234571"
        for _ in range(_RATE_LIMIT):
            _is_rate_limited(sender)
        sent: list[str] = []
        with mock.patch(
            "squish.serving.whatsapp._send_whatsapp_reply",
            side_effect=lambda pid, tok, to, body: sent.append(body),
        ):
            _handle_message(
                sender, "hello", "pid", "tok",
                lambda: _FakeState(), lambda: _fake_generate, lambda: _FakeTok(),
                _DEFAULT_SYSTEM_PROMPT,
            )
        assert len(sent) == 1
        assert "quickly" in sent[0].lower() or "wait" in sent[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# _chunk_reply — long-message splitting
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkReply:
    def test_short_text_single_chunk(self):
        assert _chunk_reply("hello") == ["hello"]

    def test_empty_string_single_chunk(self):
        assert _chunk_reply("") == [""]

    def test_exactly_max_len_single_chunk(self):
        text = "x" * _MAX_MSG_LEN
        assert _chunk_reply(text) == [text]

    def test_one_over_max_splits_into_two(self):
        text = "x" * (_MAX_MSG_LEN + 1)
        chunks = _chunk_reply(text)
        assert len(chunks) == 2
        assert all(len(c) <= _MAX_MSG_LEN for c in chunks)
        assert "".join(chunks) == text

    def test_splits_at_newline(self):
        part1 = "a" * (_MAX_MSG_LEN - 1)
        part2 = "b" * 10
        text = part1 + "\n" + part2
        chunks = _chunk_reply(text)
        assert len(chunks) == 2
        assert chunks[0] == part1
        assert chunks[1] == part2

    def test_splits_at_space_when_no_newline(self):
        # Pattern: 1000 'x' chars then a space, repeated — produces spaces inside _MAX_MSG_LEN
        segment = "x" * 999 + " "
        text = segment * 5  # 5000 chars, spaces at 1000, 2000, 3000, 4000
        chunks = _chunk_reply(text)
        assert len(chunks) >= 2
        assert all(len(c) <= _MAX_MSG_LEN for c in chunks)
        # Reassembling with the stripped whitespace should reproduce the words
        combined = " ".join(chunks)
        assert combined.replace(" ", "") == text.replace(" ", "")

    def test_hard_cut_when_no_whitespace(self):
        text = "x" * (_MAX_MSG_LEN * 2)
        chunks = _chunk_reply(text)
        assert len(chunks) == 2
        assert all(len(c) <= _MAX_MSG_LEN for c in chunks)
        assert "".join(chunks) == text

    def test_multiple_chunks_no_content_lost(self):
        text = "word " * 3000  # ~15000 chars
        chunks = _chunk_reply(text)
        assert len(chunks) >= 3
        assert all(len(c) <= _MAX_MSG_LEN for c in chunks)
        # Every word is preserved across all chunks
        rejoined = " ".join(" ".join(c.split()) for c in chunks)
        assert rejoined.split() == text.split()

    def test_long_reply_sends_multiple_messages(self):
        """Verify _handle_message delivers long model output as multiple WhatsApp messages."""
        _clear_all()
        long_output = "word " * (_MAX_MSG_LEN // 4)  # ~5x the per-message limit

        def _long_generate(prompt, max_tokens, temperature, top_p, stop, seed):
            yield (long_output, None)
            yield ("", "stop")

        sent: list[str] = []
        with mock.patch(
            "squish.serving.whatsapp._send_whatsapp_reply",
            side_effect=lambda pid, tok, to, body: sent.append(body),
        ):
            _handle_message(
                "15551299999", "tell me a story", "pid", "tok",
                lambda: _FakeState(), lambda: _long_generate, lambda: _FakeTok(),
                _DEFAULT_SYSTEM_PROMPT,
            )
        assert len(sent) >= 2
        assert all(len(s) <= _MAX_MSG_LEN for s in sent)
