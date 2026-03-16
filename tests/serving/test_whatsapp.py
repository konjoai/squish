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
    _SESSION_TIMEOUT,
    _apply_max_history,
    _expire_old_sessions,
    _get_or_create_session,
    _handle_message,
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
) -> list[str]:
    """Call _handle_message and return the list of args passed to _send_whatsapp_reply."""
    _clear_sessions()
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
