"""W75 — VEX feed plan gating.

Tests for:
- GET /cloud/vex-feed returns only alerts within the plan window.
- Community plan: alerts ≤7 days old are included; older ones are excluded.
- Community plan: 402 is returned when all existing alerts are beyond the window.
- Community plan: 200 with empty list is returned when no alerts exist at all.
- Professional plan: alerts within 90 days included; older ones excluded.
- Enterprise plan: full history returned, no age filter.
- ``X-Squash-Plan`` header is present on all responses (200 and 402).
- ``_get_plan_limits()`` returns correct dicts per plan.

Simulated today: 2026-04-15
"""

from __future__ import annotations

import datetime
import importlib
from typing import Any
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import squish.squash.api as _api_module
from squish.squash.api import _PLAN_LIMITS, _get_plan_limits, app

# ── helpers ───────────────────────────────────────────────────────────────────

_TODAY = datetime.date(2026, 4, 15)


def _iso(d: datetime.date) -> str:
    """Return an ISO 8601 datetime string for a given date (midnight UTC)."""
    return f"{d.isoformat()}T00:00:00"


def _alert(cve_id: str, days_ago: int, tenant: str = "t1") -> dict[str, Any]:
    """Build a minimal VEX alert dict with a ``created_at`` that is *days_ago* days old."""
    created = _TODAY - datetime.timedelta(days=days_ago)
    return {
        "alert_id": f"alert-{cve_id}-{days_ago}",
        "cve_id": cve_id,
        "severity": "HIGH",
        "model_id": "model-1",
        "status": "affected",
        "detail": "",
        "tenant_id": tenant,
        "created_at": _iso(created),
    }


def _vex_feed(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the dict _db_read_vex_feed() would return."""
    return {
        "total_alerts": len(alerts),
        "tenant_count": len({a["tenant_id"] for a in alerts}),
        "alerts": alerts,
    }


def _client(plan: str, today: datetime.date = _TODAY) -> TestClient:
    """Return a TestClient with SQUASH_PLAN patched to *plan*."""
    with (
        mock.patch.object(_api_module, "SQUASH_PLAN", plan),
        mock.patch(
            "squish.squash.api.datetime",
            **{"date.today.return_value": today,
               "date.fromisoformat.side_effect": datetime.date.fromisoformat,
               "datetime.fromisoformat.side_effect": datetime.datetime.fromisoformat,
               "timedelta.side_effect": datetime.timedelta},
        ),
    ):
        # Re-import to capture patched SQUASH_PLAN in _get_plan_limits
        importlib.reload(_api_module)
    # The TestClient is intentionally created outside the with-block so the
    # reload is captured but patching is transparent to the test itself.
    return TestClient(app)


# ── _get_plan_limits() unit tests ─────────────────────────────────────────────


class TestGetPlanLimits:
    def test_community_limits(self) -> None:
        with mock.patch.object(_api_module, "SQUASH_PLAN", "community"):
            limits = _get_plan_limits()
        assert limits["vex_max_age_days"] == 7
        assert limits["vex_rate_limit"] == 10

    def test_professional_limits(self) -> None:
        with mock.patch.object(_api_module, "SQUASH_PLAN", "professional"):
            limits = _get_plan_limits()
        assert limits["vex_max_age_days"] == 90
        assert limits["vex_rate_limit"] == 120

    def test_enterprise_limits(self) -> None:
        with mock.patch.object(_api_module, "SQUASH_PLAN", "enterprise"):
            limits = _get_plan_limits()
        assert limits["vex_max_age_days"] is None
        assert limits["vex_rate_limit"] is None

    def test_unknown_plan_falls_back_to_community(self) -> None:
        with mock.patch.object(_api_module, "SQUASH_PLAN", "ghost_tier"):
            limits = _get_plan_limits()
        assert limits == _PLAN_LIMITS["community"]


# ── Community plan tests ───────────────────────────────────────────────────────


class TestVexFeedCommunityPlan:
    """Community plan: only alerts ≤7 days old; 402 when history is stale."""

    def _get(self, alerts: list[dict[str, Any]]) -> "Response":  # type: ignore[name-defined]
        client = TestClient(app)
        feed = _vex_feed(alerts)
        with (
            mock.patch.object(_api_module, "SQUASH_PLAN", "community"),
            mock.patch(
                "squish.squash.api._db_read_vex_feed", return_value=feed
            ),
            mock.patch(
                "squish.squash.api.datetime",
                **{
                    "date.today.return_value": _TODAY,
                    "date.fromisoformat.side_effect": datetime.date.fromisoformat,
                    "datetime.fromisoformat.side_effect": datetime.datetime.fromisoformat,
                    "timedelta.side_effect": datetime.timedelta,
                },
            ),
        ):
            return client.get("/cloud/vex-feed")

    def test_empty_feed_returns_200_empty(self) -> None:
        r = self._get([])
        assert r.status_code == 200
        data = r.json()
        assert data["total_alerts"] == 0
        assert data["alerts"] == []

    def test_fresh_alert_within_window_returned(self) -> None:
        fresh = _alert("CVE-2026-0001", days_ago=3)
        r = self._get([fresh])
        assert r.status_code == 200
        data = r.json()
        assert data["total_alerts"] == 1
        assert data["alerts"][0]["cve_id"] == "CVE-2026-0001"

    def test_stale_alert_outside_window_excluded(self) -> None:
        old = _alert("CVE-2026-0002", days_ago=30)
        fresh = _alert("CVE-2026-0003", days_ago=1)
        r = self._get([old, fresh])
        assert r.status_code == 200
        data = r.json()
        assert data["total_alerts"] == 1
        assert data["alerts"][0]["cve_id"] == "CVE-2026-0003"

    def test_boundary_7_days_included(self) -> None:
        boundary = _alert("CVE-2026-0004", days_ago=7)
        r = self._get([boundary])
        assert r.status_code == 200
        assert r.json()["total_alerts"] == 1

    def test_boundary_8_days_excluded(self) -> None:
        old = _alert("CVE-2026-0005", days_ago=8)
        r = self._get([old])
        assert r.status_code == 402

    def test_all_stale_returns_402(self) -> None:
        stale = [
            _alert("CVE-2026-0006", days_ago=60),
            _alert("CVE-2026-0007", days_ago=120),
        ]
        r = self._get(stale)
        assert r.status_code == 402
        body = r.json()
        assert body["error"] == "plan_upgrade_required"
        assert body["plan"] == "community"
        assert body["upgrade_to"] == "professional"

    def test_402_body_contains_reason(self) -> None:
        r = self._get([_alert("CVE-2026-0008", days_ago=365)])
        assert r.status_code == 402
        assert "reason" in r.json()
        assert "7 days" in r.json()["reason"]


# ── Header tests ──────────────────────────────────────────────────────────────


class TestVexFeedPlanHeader:
    """X-Squash-Plan header must be present on all responses."""

    def _get(self, plan: str, alerts: list[dict[str, Any]]) -> "Response":  # type: ignore[name-defined]
        client = TestClient(app)
        feed = _vex_feed(alerts)
        with (
            mock.patch.object(_api_module, "SQUASH_PLAN", plan),
            mock.patch("squish.squash.api._db_read_vex_feed", return_value=feed),
            mock.patch(
                "squish.squash.api.datetime",
                **{
                    "date.today.return_value": _TODAY,
                    "date.fromisoformat.side_effect": datetime.date.fromisoformat,
                    "datetime.fromisoformat.side_effect": datetime.datetime.fromisoformat,
                    "timedelta.side_effect": datetime.timedelta,
                },
            ),
        ):
            return client.get("/cloud/vex-feed")

    def test_header_present_on_200_community(self) -> None:
        r = self._get("community", [_alert("CVE-2026-1001", days_ago=1)])
        assert r.headers.get("x-squash-plan") == "community"

    def test_header_present_on_200_professional(self) -> None:
        r = self._get("professional", [_alert("CVE-2026-1002", days_ago=30)])
        assert r.headers.get("x-squash-plan") == "professional"

    def test_header_present_on_200_enterprise(self) -> None:
        r = self._get("enterprise", [_alert("CVE-2026-1003", days_ago=365)])
        assert r.headers.get("x-squash-plan") == "enterprise"

    def test_header_present_on_402(self) -> None:
        r = self._get("community", [_alert("CVE-2026-1004", days_ago=30)])
        assert r.status_code == 402
        assert r.headers.get("x-squash-plan") == "community"

    def test_header_value_matches_squash_plan_module_var(self) -> None:
        for plan in ("community", "professional", "enterprise"):
            r = self._get(plan, [])
            assert r.headers.get("x-squash-plan") == plan


# ── Professional plan tests ───────────────────────────────────────────────────


class TestVexFeedProfessionalPlan:
    """Professional plan: 90-day window, no 402."""

    def _get(self, alerts: list[dict[str, Any]]) -> "Response":  # type: ignore[name-defined]
        client = TestClient(app)
        feed = _vex_feed(alerts)
        with (
            mock.patch.object(_api_module, "SQUASH_PLAN", "professional"),
            mock.patch("squish.squash.api._db_read_vex_feed", return_value=feed),
            mock.patch(
                "squish.squash.api.datetime",
                **{
                    "date.today.return_value": _TODAY,
                    "date.fromisoformat.side_effect": datetime.date.fromisoformat,
                    "datetime.fromisoformat.side_effect": datetime.datetime.fromisoformat,
                    "timedelta.side_effect": datetime.timedelta,
                },
            ),
        ):
            return client.get("/cloud/vex-feed")

    def test_alert_within_90_days_included(self) -> None:
        r = self._get([_alert("CVE-2026-2001", days_ago=60)])
        assert r.status_code == 200
        assert r.json()["total_alerts"] == 1

    def test_alert_beyond_90_days_excluded(self) -> None:
        r = self._get([_alert("CVE-2026-2002", days_ago=91)])
        assert r.status_code == 200
        assert r.json()["total_alerts"] == 0

    def test_no_402_when_all_alerts_beyond_window(self) -> None:
        r = self._get([_alert("CVE-2026-2003", days_ago=365)])
        # Professional never gets 402 — just returns filtered (possibly empty) list.
        assert r.status_code == 200

    def test_boundary_90_days_included(self) -> None:
        r = self._get([_alert("CVE-2026-2004", days_ago=90)])
        assert r.status_code == 200
        assert r.json()["total_alerts"] == 1


# ── Enterprise plan tests ─────────────────────────────────────────────────────


class TestVexFeedEnterprisePlan:
    """Enterprise plan: no age filter, full history returned."""

    def _get(self, alerts: list[dict[str, Any]]) -> "Response":  # type: ignore[name-defined]
        client = TestClient(app)
        feed = _vex_feed(alerts)
        with (
            mock.patch.object(_api_module, "SQUASH_PLAN", "enterprise"),
            mock.patch("squish.squash.api._db_read_vex_feed", return_value=feed),
        ):
            return client.get("/cloud/vex-feed")

    def test_old_alert_returned(self) -> None:
        r = self._get([_alert("CVE-2026-3001", days_ago=730)])
        assert r.status_code == 200
        assert r.json()["total_alerts"] == 1

    def test_mixed_age_alerts_all_returned(self) -> None:
        alerts = [
            _alert("CVE-2026-3002", days_ago=1),
            _alert("CVE-2026-3003", days_ago=90),
            _alert("CVE-2026-3004", days_ago=365),
        ]
        r = self._get(alerts)
        assert r.status_code == 200
        assert r.json()["total_alerts"] == 3

    def test_no_filtering_applied(self) -> None:
        alerts = [_alert(f"CVE-2026-30{i:02d}", days_ago=i * 100) for i in range(1, 6)]
        r = self._get(alerts)
        assert r.status_code == 200
        assert r.json()["total_alerts"] == 5
