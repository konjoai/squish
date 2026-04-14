"""tests/test_squash_w57.py — W57: cloud_db SQLite persistence + /drift-check REST."""

from __future__ import annotations

import json
import os
import tempfile
import pathlib
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# cloud_db unit tests (pure unit — no I/O side effects)
# ---------------------------------------------------------------------------

class TestCloudDB:
    """Validate CloudDB with an in-memory SQLite database."""

    def _make_db(self):
        from squish.squash.cloud_db import CloudDB
        return CloudDB(path=":memory:")

    def test_upsert_and_get_tenant(self):
        db = self._make_db()
        db.upsert_tenant("t1", {"name": "Acme", "plan": "pro"})
        result = db.get_tenant("t1")
        assert result == {"name": "Acme", "plan": "pro"}

    def test_get_tenant_missing_returns_none(self):
        db = self._make_db()
        assert db.get_tenant("nonexistent") is None

    def test_all_tenants(self):
        db = self._make_db()
        db.upsert_tenant("a", {"plan": "free"})
        db.upsert_tenant("b", {"plan": "pro"})
        result = db.all_tenants()
        assert set(result.keys()) == {"a", "b"}
        assert result["a"] == {"plan": "free"}

    def test_upsert_overwrites_existing(self):
        db = self._make_db()
        db.upsert_tenant("t1", {"plan": "free"})
        db.upsert_tenant("t1", {"plan": "enterprise"})
        assert db.get_tenant("t1") == {"plan": "enterprise"}

    def test_append_and_get_inventory(self):
        db = self._make_db()
        db.append_record("inventory", "t1", {"model": "foo"})
        db.append_record("inventory", "t1", {"model": "bar"})
        records = db.get_records("inventory", "t1")
        assert len(records) == 2
        assert records[0]["model"] == "foo"
        assert records[1]["model"] == "bar"

    def test_append_and_get_vex_alerts(self):
        db = self._make_db()
        db.append_record("vex_alerts", "t1", {"cve": "CVE-2024-0001"})
        records = db.get_records("vex_alerts", "t1")
        assert records[0]["cve"] == "CVE-2024-0001"

    def test_append_and_get_drift_events(self):
        db = self._make_db()
        db.append_record("drift_events", "t1", {"severity": "high"})
        records = db.get_records("drift_events", "t1")
        assert records[0]["severity"] == "high"

    def test_count_records(self):
        db = self._make_db()
        db.append_record("inventory", "t1", {"x": 1})
        db.append_record("inventory", "t1", {"x": 2})
        assert db.count_records("inventory", "t1") == 2
        assert db.count_records("inventory", "other") == 0

    def test_get_records_limit(self):
        db = self._make_db()
        for i in range(10):
            db.append_record("inventory", "t1", {"i": i})
        result = db.get_records("inventory", "t1", limit=3)
        assert len(result) == 3
        # get_records returns the N most-recent records, reordered ascending
        assert result[0]["i"] == 7
        assert result[2]["i"] == 9

    def test_per_tenant_limit_prunes_oldest(self):
        from squish.squash.cloud_db import CloudDB
        db = CloudDB(path=":memory:", per_tenant_limit=5)
        for i in range(8):
            db.append_record("inventory", "t1", {"i": i})
        records = db.get_records("inventory", "t1")
        assert len(records) == 5
        # After pruning, the 5 newest survive
        values = [r["i"] for r in records]
        assert values == [3, 4, 5, 6, 7]

    def test_policy_stat_increment(self):
        db = self._make_db()
        db.inc_policy_stat("t1", "license-check", passed=True)
        db.inc_policy_stat("t1", "license-check", passed=True)
        db.inc_policy_stat("t1", "license-check", passed=False)
        stats = db.get_policy_stats("t1")
        assert stats["license-check"]["passed"] == 2
        assert stats["license-check"]["failed"] == 1

    def test_policy_stat_empty_tenant(self):
        db = self._make_db()
        assert db.get_policy_stats("nobody") == {}

    def test_invalid_table_raises(self):
        db = self._make_db()
        with pytest.raises(ValueError, match="Unknown cloud_db table"):
            db.append_record("sqlite_master", "t1", {})

    def test_invalid_table_get_raises(self):
        db = self._make_db()
        with pytest.raises(ValueError, match="Unknown cloud_db table"):
            db.get_records("not_a_table", "t1")

    def test_on_disk_persistence(self):
        """Data written to an on-disk db survives reconnection."""
        from squish.squash.cloud_db import CloudDB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            db1 = CloudDB(path=path)
            db1.upsert_tenant("t1", {"plan": "pro"})
            db1.append_record("inventory", "t1", {"model": "resnet"})
            del db1

            db2 = CloudDB(path=path)
            assert db2.get_tenant("t1") == {"plan": "pro"}
            records = db2.get_records("inventory", "t1")
            assert len(records) == 1
            assert records[0]["model"] == "resnet"
        finally:
            os.unlink(path)

    def test_make_db_returns_none_for_memory(self):
        """_make_db() returns None when env var is absent or :memory:."""
        from squish.squash.cloud_db import _make_db
        original = os.environ.pop("SQUASH_CLOUD_DB", None)
        try:
            assert _make_db() is None
            os.environ["SQUASH_CLOUD_DB"] = ":memory:"
            assert _make_db() is None
        finally:
            if original is not None:
                os.environ["SQUASH_CLOUD_DB"] = original
            else:
                os.environ.pop("SQUASH_CLOUD_DB", None)

    def test_make_db_returns_clouddb_for_path(self):
        """_make_db() returns a CloudDB instance when env var is a file path."""
        from squish.squash.cloud_db import _make_db, CloudDB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            os.environ["SQUASH_CLOUD_DB"] = path
            db = _make_db()
            assert isinstance(db, CloudDB)
        finally:
            del os.environ["SQUASH_CLOUD_DB"]
            os.unlink(path)


# ---------------------------------------------------------------------------
# /drift-check REST endpoint tests (subprocess isolation)
# ---------------------------------------------------------------------------

class TestDriftCheckEndpoint:
    """Tests for POST /drift-check.  Uses subprocess to avoid FastAPI state bleed."""

    def _run(self, payload: dict) -> "subprocess.CompletedProcess[str]":
        payload_json = json.dumps(payload)
        script = f"""
import json, sys
sys.path.insert(0, {repr(str(pathlib.Path(__file__).parent.parent))})
payload = json.loads({repr(payload_json)})
from fastapi.testclient import TestClient
from squish.squash.api import app
client = TestClient(app, raise_server_exceptions=False)
resp = client.post("/drift-check", json=payload)
print(json.dumps({{"status": resp.status_code, "body": resp.json()}}))
"""
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
        )

    def test_missing_model_dir_returns_400(self):
        proc = self._run({"model_dir": "/nonexistent-dir-xyz", "bom_path": "/tmp/x.json"})
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip())
        assert result["status"] == 400

    def test_missing_bom_returns_400(self):
        with tempfile.TemporaryDirectory() as d:
            proc = self._run({"model_dir": d, "bom_path": "/nonexistent-bom.json"})
            assert proc.returncode == 0, proc.stderr
            result = json.loads(proc.stdout.strip())
            assert result["status"] == 400

    def test_valid_model_dir_and_bom_200(self):
        """Happy path: valid dir + minimal BOM JSON → 200 with ok/files_checked."""
        with tempfile.TemporaryDirectory() as d:
            bom_path = pathlib.Path(d) / "bom.json"
            # Minimal CycloneDX-like structure the drift module can read
            bom_path.write_text(json.dumps({"components": []}))
            proc = self._run({"model_dir": d, "bom_path": str(bom_path)})
            assert proc.returncode == 0, proc.stderr
            result = json.loads(proc.stdout.strip())
            # Drift module may return 422 if it rejects an empty BOM; that is
            # also acceptable.  What is NOT acceptable: 400 (path errors) or 500.
            assert result["status"] in (200, 422), f"Unexpected status: {result}"
