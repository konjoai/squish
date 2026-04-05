"""Wave 17 — PolicyWebhook tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ENV_VAR = "SQUASH_WEBHOOK_URL"


def _make_policy_result(passed: bool = True, error_count: int = 0):
    from squish.squash.policy import PolicyResult, PolicyFinding
    findings = [
        PolicyFinding(
            rule_id=f"E{i}", severity="error", passed=False,
            field="", rationale="", remediation="",
        )
        for i in range(error_count)
    ]
    return PolicyResult(policy_name="no-cvss", passed=passed, findings=findings)


class TestPolicyWebhookNoUrl(unittest.TestCase):
    def setUp(self):
        # Ensure env var is absent
        os.environ.pop(_ENV_VAR, None)

    def test_notify_no_url_returns_false(self):
        """notify() returns False when no URL is provided and env var is unset."""
        from squish.squash.policy import PolicyWebhook
        wh = PolicyWebhook()
        result = wh.notify(_make_policy_result(), model_path=Path("/tmp/model"))
        self.assertFalse(result)

    def test_notify_raw_empty_url_returns_false(self):
        from squish.squash.policy import PolicyWebhook
        result = PolicyWebhook.notify_raw({"event": "test"}, webhook_url="")
        self.assertFalse(result)


class TestPolicyWebhookEnvVar(unittest.TestCase):
    def setUp(self):
        os.environ[_ENV_VAR] = "https://example.com/webhook"

    def tearDown(self):
        os.environ.pop(_ENV_VAR, None)

    def test_notify_reads_env_var(self):
        """notify() picks up SQUASH_WEBHOOK_URL from environment."""
        from squish.squash.policy import PolicyWebhook

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            wh = PolicyWebhook()
            result = wh.notify(_make_policy_result(), model_path=Path("/tmp/model"))
        self.assertTrue(result)


class TestPolicyWebhookMockUrllib(unittest.TestCase):
    def test_notify_sends_correct_payload(self):
        """notify() sends the expected event name and model_path."""
        from squish.squash.policy import PolicyWebhook, PolicyResult

        captured_request = {}

        def fake_urlopen(req, timeout=None):
            captured_request["url"] = req.full_url
            captured_request["data"] = json.loads(req.data.decode())
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", fake_urlopen):
            wh = PolicyWebhook()
            wh.notify(
                _make_policy_result(passed=False, error_count=3),
                model_path=Path("/models/my-model"),
                webhook_url="https://hooks.example.com/squash",
            )

        self.assertEqual(captured_request["url"], "https://hooks.example.com/squash")
        payload = captured_request["data"]
        self.assertEqual(payload["event"], "squash.policy.result")
        self.assertIn("model_path", payload)
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["error_count"], 3)

    def test_notify_returns_true_on_200(self):
        from squish.squash.policy import PolicyWebhook

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            wh = PolicyWebhook()
            result = wh.notify(
                _make_policy_result(),
                model_path=Path("/models/m"),
                webhook_url="https://example.com/wh",
            )
        self.assertTrue(result)

    def test_notify_returns_false_on_http_error(self):
        from squish.squash.policy import PolicyWebhook
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("conn refused")):
            wh = PolicyWebhook()
            result = wh.notify(
                _make_policy_result(),
                model_path=Path("/models/m"),
                webhook_url="https://example.com/wh",
            )
        self.assertFalse(result)

    def test_notify_raw_sends_payload(self):
        from squish.squash.policy import PolicyWebhook

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = PolicyWebhook.notify_raw(
                {"event": "custom", "data": 42},
                webhook_url="https://example.com/wh",
            )
        self.assertTrue(result)

    def test_notify_never_raises(self):
        """notify() must not propagate exceptions — always returns bool."""
        from squish.squash.policy import PolicyWebhook

        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            wh = PolicyWebhook()
            result = wh.notify(
                _make_policy_result(),
                model_path=Path("/models/m"),
                webhook_url="https://example.com/wh",
            )
        self.assertIsInstance(result, bool)
        self.assertFalse(result)


class TestPolicyWebhookDtypeContracts(unittest.TestCase):
    def test_notify_return_type(self):
        from squish.squash.policy import PolicyWebhook
        os.environ.pop(_ENV_VAR, None)
        wh = PolicyWebhook()
        result = wh.notify(_make_policy_result(), model_path=Path("/m"))
        self.assertIsInstance(result, bool)

    def test_notify_raw_return_type(self):
        from squish.squash.policy import PolicyWebhook
        result = PolicyWebhook.notify_raw({}, webhook_url="")
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
