"""
Tests for squish/serving/whatsapp.py.

Covers:
  - TwiML response builder (_twiml_reply)
  - Session management helpers (_get_or_create_session, _reset_session,
    _expire_old_sessions, _apply_max_history)
  - Twilio HMAC-SHA1 signature validation (_validate_twilio_signature)
  - HTTP endpoint behaviour via FastAPI TestClient (GET challenge, POST handler)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
import unittest.mock as mock

import pytest

# ── import the module under test ──────────────────────────────────────────────
from squish.serving.whatsapp import (
    _DEFAULT_SYSTEM_PROMPT,
    _MAX_HISTORY,
    _SESSION_TIMEOUT,
    _apply_max_history,
    _expire_old_sessions,
    _get_or_create_session,
    _reset_session,
    _sessions,
    _sessions_lock,
    _sessions_ts,
    _twiml_reply,
    _validate_twilio_signature,
    mount_whatsapp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clear_sessions() -> None:
    """Reset module-level session stores before each test."""
    with _sessions_lock:
        _sessions.clear()
        _sessions_ts.clear()


def _make_signature(auth_token: str, url: str, form_data: dict[str, str]) -> str:
    """Compute a valid X-Twilio-Signature for the given parameters."""
    s = url
    for k in sorted(form_data.keys()):
        s += k + form_data[k]
    mac = hmac.new(auth_token.encode("utf-8"), s.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(mac).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# _twiml_reply
# ─────────────────────────────────────────────────────────────────────────────

class TestTwimlReply:
    def test_basic_wrapping(self):
        result = _twiml_reply("Hello!")
        assert result.startswith('<?xml version="1.0" encoding="UTF-8"?>')
        assert "<Response><Message>Hello!</Message></Response>" in result

    def test_escapes_ampersand(self):
        result = _twiml_reply("bread & butter")
        assert "&amp;" in result
        # Unescaped bare & should no longer appear as " & " in the xml body
        assert "bread & butter" not in result

    def test_escapes_less_than(self):
        assert "&lt;" in _twiml_reply("<b>bold</b>")

    def test_escapes_greater_than(self):
        assert "&gt;" in _twiml_reply("a > b")

    def test_escapes_double_quote(self):
        assert "&quot;" in _twiml_reply('say "yes"')

    def test_empty_body(self):
        result = _twiml_reply("")
        assert "<Message></Message>" in result

    def test_multiline_body(self):
        body = "Line 1\nLine 2\nLine 3"
        result = _twiml_reply(body)
        assert "Line 1\nLine 2\nLine 3" in result


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

class TestSessions:
    def setup_method(self):
        _clear_sessions()

    def test_creates_new_session_with_system_prompt(self):
        msgs = _get_or_create_session("whatsapp:+15551234567", "Custom prompt")
        assert len(msgs) == 1
        assert msgs[0] == {"role": "system", "content": "Custom prompt"}

    def test_returns_existing_session(self):
        msgs1 = _get_or_create_session("whatsapp:+15551234567", "Custom prompt")
        msgs1.append({"role": "user", "content": "hi"})
        msgs2 = _get_or_create_session("whatsapp:+15551234567", "Custom prompt")
        assert len(msgs2) == 2  # system + user

    def test_updates_last_activity_timestamp(self):
        before = time.time()
        _get_or_create_session("whatsapp:+1111", "p")
        after = time.time()
        assert before <= _sessions_ts["whatsapp:+1111"] <= after

    def test_reset_clears_history(self):
        msgs = _get_or_create_session("whatsapp:+1111", "p")
        msgs.append({"role": "user", "content": "hello"})
        msgs.append({"role": "assistant", "content": "hi"})
        _reset_session("whatsapp:+1111", "new prompt")
        assert _sessions["whatsapp:+1111"] == [{"role": "system", "content": "new prompt"}]

    def test_expire_old_sessions(self):
        _sessions["whatsapp:+old"] = [{"role": "system", "content": "p"}]
        _sessions_ts["whatsapp:+old"] = time.time() - _SESSION_TIMEOUT - 1
        _sessions["whatsapp:+fresh"] = [{"role": "system", "content": "p"}]
        _sessions_ts["whatsapp:+fresh"] = time.time()
        _expire_old_sessions()
        assert "whatsapp:+old" not in _sessions
        assert "whatsapp:+fresh" in _sessions

    def test_expire_does_not_remove_active_sessions(self):
        _sessions["whatsapp:+recent"] = [{"role": "system", "content": "p"}]
        _sessions_ts["whatsapp:+recent"] = time.time() - _SESSION_TIMEOUT + 60
        _expire_old_sessions()
        assert "whatsapp:+recent" in _sessions


class TestApplyMaxHistory:
    def test_passes_through_short_history(self):
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg{i}"} for i in range(5)
        ]
        result = _apply_max_history(msgs)
        assert result == msgs

    def test_trims_to_max_history(self):
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg{i}"} for i in range(_MAX_HISTORY + 5)
        ]
        result = _apply_max_history(msgs)
        # system prompt + (_MAX_HISTORY - 1) non-system
        assert len(result) == _MAX_HISTORY
        assert result[0]["role"] == "system"
        # should keep the *last* messages
        assert result[-1]["content"] == f"msg{_MAX_HISTORY + 4}"

    def test_preserves_system_prompt(self):
        msgs = [{"role": "system", "content": "keep me"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(_MAX_HISTORY + 2)
        ]
        result = _apply_max_history(msgs)
        assert result[0] == {"role": "system", "content": "keep me"}

    def test_multiple_system_prompts_all_kept(self):
        msgs = [
            {"role": "system", "content": "s1"},
            {"role": "system", "content": "s2"},
            {"role": "user", "content": "hello"},
        ]
        result = _apply_max_history(msgs)
        assert result[:2] == [
            {"role": "system", "content": "s1"},
            {"role": "system", "content": "s2"},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Twilio signature validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateTwilioSignature:
    URL = "https://example.ngrok.io/webhook/whatsapp"
    AUTH = "test_auth_token_abc123"
    FORM = {"From": "whatsapp:+15551234567", "Body": "Hello there", "NumMedia": "0"}

    def test_valid_signature_accepted(self):
        sig = _make_signature(self.AUTH, self.URL, self.FORM)
        assert _validate_twilio_signature(self.AUTH, self.URL, self.FORM, sig) is True

    def test_invalid_signature_rejected(self):
        assert _validate_twilio_signature(self.AUTH, self.URL, self.FORM, "badsig") is False

    def test_wrong_auth_token_rejected(self):
        sig = _make_signature(self.AUTH, self.URL, self.FORM)
        assert _validate_twilio_signature("wrong_token", self.URL, self.FORM, sig) is False

    def test_different_url_rejected(self):
        sig = _make_signature(self.AUTH, self.URL, self.FORM)
        assert _validate_twilio_signature(
            self.AUTH, "https://evil.example.com/webhook/whatsapp", self.FORM, sig
        ) is False

    def test_empty_form_valid(self):
        empty_form: dict[str, str] = {}
        sig = _make_signature(self.AUTH, self.URL, empty_form)
        assert _validate_twilio_signature(self.AUTH, self.URL, empty_form, sig) is True

    def test_sorted_parameters_deterministic(self):
        # Inserting form data in different orders should produce the same sig
        form_a = {"Body": "hi", "From": "+1234"}
        form_b = {"From": "+1234", "Body": "hi"}
        sig_a = _make_signature(self.AUTH, self.URL, form_a)
        sig_b = _make_signature(self.AUTH, self.URL, form_b)
        assert sig_a == sig_b


# ─────────────────────────────────────────────────────────────────────────────
# mount_whatsapp endpoint tests via FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Return a TestClient with WhatsApp endpoints mounted."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    _clear_sessions()

    app = FastAPI()

    # Mock state: model loaded
    class _FakeState:
        model = object()
        model_name = "squish-7b"
        avg_tps = 12.3
        requests = 5
        loaded_at = time.time() - 120

    fake_state = _FakeState()

    def _fake_generate(prompt, max_tokens, temperature, top_p, stop, seed):
        yield ("Hello", None)
        yield (" world", None)
        yield ("", "stop")

    # Tokenizer with apply_chat_template
    class _FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            parts = " ".join(m["content"] for m in messages)
            return parts

    mount_whatsapp(
        app,
        get_state=lambda: fake_state,
        get_generate=lambda: _fake_generate,
        get_tokenizer=lambda: _FakeTok(),
        account_sid="",
        auth_token="",  # no signature validation
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
    )

    return TestClient(app, raise_server_exceptions=True)


class TestGetChallenge:
    def test_returns_200(self, client):
        resp = client.get("/webhook/whatsapp")
        assert resp.status_code == 200

    def test_returns_xml_content_type(self, client):
        resp = client.get("/webhook/whatsapp")
        assert "xml" in resp.headers["content-type"]

    def test_empty_twiml_envelope(self, client):
        resp = client.get("/webhook/whatsapp")
        assert "<Response>" in resp.text
        assert resp.text.startswith("<?xml")


class TestPostIncoming:
    def _post(self, client, *, from_num="whatsapp:+15551234567", body="Hello!"):
        return client.post(
            "/webhook/whatsapp",
            data={"From": from_num, "Body": body},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    def test_basic_reply_returns_200(self, client):
        resp = self._post(client)
        assert resp.status_code == 200

    def test_reply_is_xml(self, client):
        resp = self._post(client)
        assert "xml" in resp.headers["content-type"]
        assert "<Response><Message>" in resp.text

    def test_reply_contains_generated_text(self, client):
        resp = self._post(client)
        assert "Hello world" in resp.text

    def test_missing_from_returns_200_with_error(self, client):
        resp = client.post(
            "/webhook/whatsapp",
            data={"Body": "hi"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        assert "No message received" in resp.text

    def test_missing_body_returns_200_with_error(self, client):
        resp = client.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+1234"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        assert "No message received" in resp.text

    def test_reset_command(self, client):
        resp = self._post(client, body="/reset")
        assert resp.status_code == 200
        assert "Session cleared" in resp.text

    def test_help_command(self, client):
        resp = self._post(client, body="/help")
        assert resp.status_code == 200
        assert "/reset" in resp.text
        assert "/status" in resp.text

    def test_status_command(self, client):
        resp = self._post(client, body="/status")
        assert resp.status_code == 200
        assert "squish-7b" in resp.text

    def test_session_history_persists(self, client):
        _clear_sessions()
        self._post(client, from_num="whatsapp:+99999", body="first message")
        assert "whatsapp:+99999" in _sessions
        # Session should contain system, user, assistant
        session = _sessions["whatsapp:+99999"]
        roles = [m["role"] for m in session]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles


class TestPostIncomingNoModel:
    """Verify graceful handling when model is not yet loaded."""

    def test_model_not_loaded_returns_friendly_message(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        _clear_sessions()
        app = FastAPI()

        class _NotLoaded:
            model = None

        mount_whatsapp(
            app,
            get_state=lambda: _NotLoaded(),
            get_generate=lambda: None,
            get_tokenizer=lambda: None,
            auth_token="",
        )
        c = TestClient(app, raise_server_exceptions=True)
        resp = c.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+1", "Body": "hi"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        assert "still loading" in resp.text


class TestPostIncomingWithSignature:
    """Verify signature validation enforced when auth_token is set."""

    AUTH = "supersecret"
    URL = "http://testserver/webhook/whatsapp"

    def _mount(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        _clear_sessions()
        app = FastAPI()

        class _FakeState:
            model = object()
            model_name = "q"
            avg_tps = 1.0
            requests = 0
            loaded_at = time.time()

        def _fake_gen(prompt, max_tokens, temperature, top_p, stop, seed):
            yield ("ok", "stop")

        class _FakeTok:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

        mount_whatsapp(
            app,
            get_state=lambda: _FakeState(),
            get_generate=lambda: _fake_gen,
            get_tokenizer=lambda: _FakeTok(),
            auth_token=self.AUTH,
        )
        return TestClient(app, raise_server_exceptions=True)

    def test_missing_signature_returns_403(self):
        c = self._mount()
        resp = c.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+1", "Body": "hi"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 403
        assert "missing signature" in resp.text

    def test_invalid_signature_returns_403(self):
        c = self._mount()
        resp = c.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+1", "Body": "hi"},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Twilio-Signature": "invalidsig==",
            },
        )
        assert resp.status_code == 403
        assert "invalid signature" in resp.text

    def test_valid_signature_returns_200(self):
        c = self._mount()
        form = {"From": "whatsapp:+1", "Body": "hi"}
        sig = _make_signature(self.AUTH, self.URL, form)
        resp = c.post(
            "/webhook/whatsapp",
            data=form,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Twilio-Signature": sig,
            },
        )
        assert resp.status_code == 200


class TestGenerationError:
    """Verify generation exceptions produce a friendly TwiML error response."""

    def test_generate_exception_returns_200_with_error(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        _clear_sessions()
        app = FastAPI()

        class _FakeState:
            model = object()
            model_name = "q"
            avg_tps = 0.0
            requests = 0
            loaded_at = time.time()

        def _broken_gen(prompt, max_tokens, temperature, top_p, stop, seed):
            raise RuntimeError("GPU on fire")
            yield  # make it a generator

        class _FakeTok:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

        mount_whatsapp(
            app,
            get_state=lambda: _FakeState(),
            get_generate=lambda: _broken_gen,
            get_tokenizer=lambda: _FakeTok(),
            auth_token="",
        )
        c = TestClient(app, raise_server_exceptions=True)
        resp = c.post(
            "/webhook/whatsapp",
            data={"From": "whatsapp:+1", "Body": "hello"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        assert "error" in resp.text.lower()
