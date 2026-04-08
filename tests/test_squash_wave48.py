"""tests/test_squash_wave48.py — Wave 48: model transformation lineage chain.

Tests for squish/squash/lineage.py (TransformationEvent, LineageVerifyResult,
LineageChain, _load_chain_json, _utc_now), the ``squash lineage`` CLI
subcommand, and the POST /lineage/record, GET /lineage/show,
POST /lineage/verify REST routes.

All tests are pure-unit or integration (temp dirs, cleaned up in tearDown).
No network, no model weights, no subprocess state mutation.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_SQUISH_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_SQUISH_ROOT))


# ===========================================================================
# TransformationEvent
# ===========================================================================


class TestTransformationEvent(unittest.TestCase):
    def _make(self) -> "TransformationEvent":
        from squish.squash.lineage import TransformationEvent

        return TransformationEvent(
            event_id="e1",
            model_id="my-model",
            operation="compress",
            operator="alice@box",
            timestamp="2026-04-08T12:00:00Z",
            input_dir="/src",
            output_dir="/dst",
            params={"format": "INT4"},
            prev_hash="",
            event_hash="abc",
        )

    def test_fields_preserved(self) -> None:
        e = self._make()
        self.assertEqual(e.event_id, "e1")
        self.assertEqual(e.model_id, "my-model")
        self.assertEqual(e.operation, "compress")
        self.assertEqual(e.operator, "alice@box")
        self.assertEqual(e.timestamp, "2026-04-08T12:00:00Z")
        self.assertEqual(e.input_dir, "/src")
        self.assertEqual(e.output_dir, "/dst")
        self.assertEqual(e.params, {"format": "INT4"})
        self.assertEqual(e.prev_hash, "")
        self.assertEqual(e.event_hash, "abc")

    def test_to_dict_has_all_keys(self) -> None:
        e = self._make()
        d = e.to_dict()
        self.assertIsInstance(d, dict)
        for key in (
            "event_id", "model_id", "operation", "operator", "timestamp",
            "input_dir", "output_dir", "params", "prev_hash", "event_hash",
        ):
            self.assertIn(key, d)

    def test_to_dict_values_correct(self) -> None:
        e = self._make()
        d = e.to_dict()
        self.assertEqual(d["operation"], "compress")
        self.assertEqual(d["params"], {"format": "INT4"})

    def test_to_dict_equals_asdict(self) -> None:
        e = self._make()
        self.assertEqual(e.to_dict(), asdict(e))


# ===========================================================================
# LineageVerifyResult
# ===========================================================================


class TestLineageVerifyResult(unittest.TestCase):
    def test_fields_intact(self) -> None:
        from squish.squash.lineage import LineageVerifyResult

        r = LineageVerifyResult(
            ok=True,
            model_dir="/mdir",
            verified_at="2026-04-08T12:00:00Z",
            event_count=3,
            broken_at=None,
            message="chain intact (3 event(s))",
        )
        self.assertTrue(r.ok)
        self.assertEqual(r.model_dir, "/mdir")
        self.assertEqual(r.event_count, 3)
        self.assertIsNone(r.broken_at)

    def test_fields_broken(self) -> None:
        from squish.squash.lineage import LineageVerifyResult

        r = LineageVerifyResult(
            ok=False,
            model_dir="/mdir",
            verified_at="2026-04-08T12:00:00Z",
            event_count=2,
            broken_at=1,
            message="event 1 hash mismatch",
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.broken_at, 1)

    def test_to_dict_has_all_keys(self) -> None:
        from squish.squash.lineage import LineageVerifyResult

        r = LineageVerifyResult(
            ok=True, model_dir="/m", verified_at="...", event_count=0,
            broken_at=None, message="ok",
        )
        d = r.to_dict()
        for key in ("ok", "model_dir", "verified_at", "event_count", "broken_at", "message"):
            self.assertIn(key, d)

    def test_to_dict_broken_at_none_serialised(self) -> None:
        from squish.squash.lineage import LineageVerifyResult

        r = LineageVerifyResult(
            ok=True, model_dir="/m", verified_at="...", event_count=1,
            broken_at=None, message="ok",
        )
        d = r.to_dict()
        self.assertIsNone(d["broken_at"])


# ===========================================================================
# _utc_now helper
# ===========================================================================


class TestUtcNow(unittest.TestCase):
    def test_returns_utc_iso_string(self) -> None:
        from squish.squash.lineage import _utc_now

        ts = _utc_now()
        self.assertIsInstance(ts, str)
        self.assertTrue(ts.endswith("Z"), f"Expected Z suffix, got {ts!r}")

    def test_two_calls_are_monotonic(self) -> None:
        from squish.squash.lineage import _utc_now
        import time

        t1 = _utc_now()
        time.sleep(0.01)
        t2 = _utc_now()
        self.assertGreaterEqual(t2, t1)


# ===========================================================================
# _load_chain_json helper
# ===========================================================================


class TestLoadChainJson(unittest.TestCase):
    def test_missing_file_returns_empty_list(self) -> None:
        from squish.squash.lineage import _load_chain_json

        result = _load_chain_json(Path("/no/such/file/lineage.json"))
        self.assertEqual(result, [])

    def test_valid_json_array_returned(self) -> None:
        from squish.squash.lineage import _load_chain_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"event_id": "x"}], f)
            fname = f.name
        try:
            result = _load_chain_json(Path(fname))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["event_id"], "x")
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_malformed_json_returns_empty_list(self) -> None:
        from squish.squash.lineage import _load_chain_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            fname = f.name
        try:
            result = _load_chain_json(Path(fname))
            self.assertEqual(result, [])
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_non_array_json_returns_empty_list(self) -> None:
        from squish.squash.lineage import _load_chain_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            fname = f.name
        try:
            result = _load_chain_json(Path(fname))
            self.assertEqual(result, [])
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_empty_array_returns_empty_list(self) -> None:
        from squish.squash.lineage import _load_chain_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            fname = f.name
        try:
            result = _load_chain_json(Path(fname))
            self.assertEqual(result, [])
        finally:
            Path(fname).unlink(missing_ok=True)


# ===========================================================================
# LineageChain._hash_event
# ===========================================================================


class TestLineageChainHashEvent(unittest.TestCase):
    def _base_event(self) -> "TransformationEvent":
        from squish.squash.lineage import TransformationEvent

        return TransformationEvent(
            event_id="e1", model_id="m", operation="compress",
            operator="u@h", timestamp="2026-04-08T00:00:00Z",
            input_dir="/in", output_dir="/out",
            params={"format": "INT4"}, prev_hash="", event_hash="",
        )

    def test_deterministic(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = self._base_event()
        h1 = LineageChain._hash_event(evt)
        h2 = LineageChain._hash_event(evt)
        self.assertEqual(h1, h2)

    def test_returns_64_char_hex(self) -> None:
        from squish.squash.lineage import LineageChain

        h = LineageChain._hash_event(self._base_event())
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_event_hash_field_excluded_from_input(self) -> None:
        """Two events identical except event_hash → identical computed hash."""
        from squish.squash.lineage import LineageChain, TransformationEvent

        base = self._base_event()
        base.event_hash = ""
        h1 = LineageChain._hash_event(base)
        base.event_hash = "DIFFERENT_VALUE"
        h2 = LineageChain._hash_event(base)
        self.assertEqual(h1, h2)

    def test_sensitive_to_operation_change(self) -> None:
        from squish.squash.lineage import LineageChain

        e1 = self._base_event()
        e2 = self._base_event()
        e2.operation = "sign"
        self.assertNotEqual(
            LineageChain._hash_event(e1),
            LineageChain._hash_event(e2),
        )

    def test_sensitive_to_params_change(self) -> None:
        from squish.squash.lineage import LineageChain

        e1 = self._base_event()
        e2 = self._base_event()
        e2.params = {"format": "INT3"}
        self.assertNotEqual(
            LineageChain._hash_event(e1),
            LineageChain._hash_event(e2),
        )

    def test_sensitive_to_prev_hash_change(self) -> None:
        from squish.squash.lineage import LineageChain

        e1 = self._base_event()
        e2 = self._base_event()
        e2.prev_hash = "some_previous_hash_value"
        self.assertNotEqual(
            LineageChain._hash_event(e1),
            LineageChain._hash_event(e2),
        )


# ===========================================================================
# LineageChain.create_event
# ===========================================================================


class TestLineageChainCreateEvent(unittest.TestCase):
    def test_returns_transformation_event(self) -> None:
        from squish.squash.lineage import LineageChain, TransformationEvent

        evt = LineageChain.create_event(
            operation="compress",
            model_id="test-model",
            input_dir="/src",
            output_dir="/dst",
        )
        self.assertIsInstance(evt, TransformationEvent)

    def test_prev_hash_and_event_hash_empty(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = LineageChain.create_event("compress", "m", "/s", "/d")
        self.assertEqual(evt.prev_hash, "")
        self.assertEqual(evt.event_hash, "")

    def test_operator_contains_at_sign(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = LineageChain.create_event("compress", "m", "/s", "/d")
        self.assertIn("@", evt.operator)

    def test_timestamp_is_utc_iso(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = LineageChain.create_event("compress", "m", "/s", "/d")
        self.assertTrue(evt.timestamp.endswith("Z"))

    def test_event_id_is_nonempty_string(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = LineageChain.create_event("sign", "m", "/s", "/d")
        self.assertIsInstance(evt.event_id, str)
        self.assertTrue(len(evt.event_id) > 0)

    def test_params_defaults_to_empty_dict(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = LineageChain.create_event("compress", "m", "/s", "/d")
        self.assertEqual(evt.params, {})

    def test_params_copied(self) -> None:
        from squish.squash.lineage import LineageChain

        params = {"format": "INT4", "group_size": 32}
        evt = LineageChain.create_event("compress", "m", "/s", "/d", params=params)
        self.assertEqual(evt.params, {"format": "INT4", "group_size": 32})

    def test_operation_and_model_id_stored(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = LineageChain.create_event("export", "qwen3:8b", "/in", "/out")
        self.assertEqual(evt.operation, "export")
        self.assertEqual(evt.model_id, "qwen3:8b")


# ===========================================================================
# LineageChain.record
# ===========================================================================


class TestLineageChainRecord(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.model_dir = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _compress_event(self, model_id: str = "m") -> "TransformationEvent":
        from squish.squash.lineage import LineageChain

        return LineageChain.create_event(
            operation="compress", model_id=model_id,
            input_dir=str(self.model_dir), output_dir=str(self.model_dir),
        )

    def test_returns_nonempty_event_hash(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = self._compress_event()
        event_hash = LineageChain.record(self.model_dir, evt)
        self.assertIsInstance(event_hash, str)
        self.assertEqual(len(event_hash), 64)

    def test_chain_file_created(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = self._compress_event()
        LineageChain.record(self.model_dir, evt)
        chain_file = self.model_dir / LineageChain.CHAIN_FILENAME
        self.assertTrue(chain_file.exists())

    def test_chain_file_is_json_array(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = self._compress_event()
        LineageChain.record(self.model_dir, evt)
        chain_file = self.model_dir / LineageChain.CHAIN_FILENAME
        data = json.loads(chain_file.read_text())
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)

    def test_genesis_event_prev_hash_empty(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = self._compress_event()
        LineageChain.record(self.model_dir, evt)
        self.assertEqual(evt.prev_hash, "")

    def test_event_hash_assigned_after_record(self) -> None:
        from squish.squash.lineage import LineageChain

        evt = self._compress_event()
        returned_hash = LineageChain.record(self.model_dir, evt)
        self.assertEqual(evt.event_hash, returned_hash)
        self.assertNotEqual(evt.event_hash, "")

    def test_second_event_prev_hash_links_to_first(self) -> None:
        from squish.squash.lineage import LineageChain

        evt1 = self._compress_event()
        hash1 = LineageChain.record(self.model_dir, evt1)

        evt2 = self._compress_event()
        LineageChain.record(self.model_dir, evt2)
        self.assertEqual(evt2.prev_hash, hash1)

    def test_three_events_are_chained(self) -> None:
        from squish.squash.lineage import LineageChain

        hashes = []
        for op in ("compress", "sign", "export"):
            e = LineageChain.create_event(op, "m", "/s", "/d")
            hashes.append(LineageChain.record(self.model_dir, e))

        chain_file = self.model_dir / LineageChain.CHAIN_FILENAME
        data = json.loads(chain_file.read_text())
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["prev_hash"], "")
        self.assertEqual(data[1]["prev_hash"], hashes[0])
        self.assertEqual(data[2]["prev_hash"], hashes[1])

    def test_record_creates_dir_if_needed(self) -> None:
        from squish.squash.lineage import LineageChain

        new_dir = self.model_dir / "nested" / "deep"
        evt = LineageChain.create_event("compress", "m", str(new_dir), str(new_dir))
        LineageChain.record(new_dir, evt)
        self.assertTrue((new_dir / LineageChain.CHAIN_FILENAME).exists())


# ===========================================================================
# LineageChain.load
# ===========================================================================


class TestLineageChainLoad(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.model_dir = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_round_trip(self) -> None:
        from squish.squash.lineage import LineageChain, TransformationEvent

        evt = LineageChain.create_event(
            "compress", "my-model",
            str(self.model_dir), str(self.model_dir),
            params={"format": "INT4"},
        )
        LineageChain.record(self.model_dir, evt)

        loaded = LineageChain.load(self.model_dir)
        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded[0], TransformationEvent)
        self.assertEqual(loaded[0].operation, "compress")
        self.assertEqual(loaded[0].model_id, "my-model")
        self.assertEqual(loaded[0].params, {"format": "INT4"})

    def test_missing_chain_file_returns_empty_list(self) -> None:
        from squish.squash.lineage import LineageChain

        result = LineageChain.load(self.model_dir)
        self.assertEqual(result, [])

    def test_load_preserves_event_hash_and_prev_hash(self) -> None:
        from squish.squash.lineage import LineageChain

        e1 = LineageChain.create_event("compress", "m", "/s", "/d")
        h1 = LineageChain.record(self.model_dir, e1)
        e2 = LineageChain.create_event("sign", "m", "/s", "/d")
        LineageChain.record(self.model_dir, e2)

        loaded = LineageChain.load(self.model_dir)
        self.assertEqual(loaded[0].event_hash, h1)
        self.assertEqual(loaded[1].prev_hash, h1)

    def test_multiple_round_trip(self) -> None:
        from squish.squash.lineage import LineageChain

        ops = ["compress", "sign", "export", "verify"]
        for op in ops:
            e = LineageChain.create_event(op, "m", "/s", "/d")
            LineageChain.record(self.model_dir, e)

        loaded = LineageChain.load(self.model_dir)
        self.assertEqual(len(loaded), 4)
        self.assertEqual([e.operation for e in loaded], ops)


# ===========================================================================
# LineageChain.verify
# ===========================================================================


class TestLineageChainVerify(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.model_dir = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _build_intact_chain(self, n: int = 3) -> None:
        from squish.squash.lineage import LineageChain

        ops = ["compress", "sign", "export", "verify", "transfer"]
        for i in range(n):
            e = LineageChain.create_event(ops[i % len(ops)], "m", "/s", "/d")
            LineageChain.record(self.model_dir, e)

    def test_intact_chain_returns_ok_true(self) -> None:
        from squish.squash.lineage import LineageChain

        self._build_intact_chain(3)
        result = LineageChain.verify(self.model_dir)
        self.assertTrue(result.ok)
        self.assertEqual(result.event_count, 3)
        self.assertIsNone(result.broken_at)

    def test_missing_chain_returns_ok_false(self) -> None:
        from squish.squash.lineage import LineageChain

        result = LineageChain.verify(self.model_dir)
        self.assertFalse(result.ok)
        self.assertIn("not found", result.message)

    def test_empty_chain_returns_ok_true(self) -> None:
        """Empty array in chain file is considered valid (no events to verify)."""
        from squish.squash.lineage import LineageChain

        chain_path = self.model_dir / LineageChain.CHAIN_FILENAME
        chain_path.write_text("[]", encoding="utf-8")
        result = LineageChain.verify(self.model_dir)
        self.assertTrue(result.ok)
        self.assertEqual(result.event_count, 0)

    def test_tampered_event_hash_detected(self) -> None:
        from squish.squash.lineage import LineageChain

        self._build_intact_chain(2)
        chain_path = self.model_dir / LineageChain.CHAIN_FILENAME
        data = json.loads(chain_path.read_text())
        data[0]["event_hash"] = "0" * 64  # tamper stored hash
        chain_path.write_text(json.dumps(data), encoding="utf-8")

        result = LineageChain.verify(self.model_dir)
        self.assertFalse(result.ok)
        self.assertEqual(result.broken_at, 0)

    def test_tampered_prev_hash_detected(self) -> None:
        from squish.squash.lineage import LineageChain

        self._build_intact_chain(2)
        chain_path = self.model_dir / LineageChain.CHAIN_FILENAME
        data = json.loads(chain_path.read_text())
        data[1]["prev_hash"] = "f" * 64  # tamper the link
        chain_path.write_text(json.dumps(data), encoding="utf-8")

        result = LineageChain.verify(self.model_dir)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.broken_at)

    def test_tampered_operation_field_detected(self) -> None:
        """Changing a payload field invalidates the stored event_hash."""
        from squish.squash.lineage import LineageChain

        self._build_intact_chain(1)
        chain_path = self.model_dir / LineageChain.CHAIN_FILENAME
        data = json.loads(chain_path.read_text())
        data[0]["operation"] = "EVIL_OP"  # change payload but NOT event_hash
        chain_path.write_text(json.dumps(data), encoding="utf-8")

        result = LineageChain.verify(self.model_dir)
        self.assertFalse(result.ok)

    def test_verify_returns_verified_at_field(self) -> None:
        from squish.squash.lineage import LineageChain

        self._build_intact_chain(1)
        result = LineageChain.verify(self.model_dir)
        self.assertTrue(result.verified_at.endswith("Z"))

    def test_verify_never_raises(self) -> None:
        """verify() must not raise even on a completely corrupt file."""
        from squish.squash.lineage import LineageChain

        chain_path = self.model_dir / LineageChain.CHAIN_FILENAME
        chain_path.write_text("this is not json at all", encoding="utf-8")
        try:
            result = LineageChain.verify(self.model_dir)
            # may be ok/not-ok; point is: no exception
        except Exception as exc:  # noqa: BLE001
            self.fail(f"verify() raised unexpectedly: {exc}")

    def test_model_dir_in_result(self) -> None:
        from squish.squash.lineage import LineageChain

        self._build_intact_chain(1)
        result = LineageChain.verify(self.model_dir)
        self.assertEqual(result.model_dir, str(self.model_dir))


# ===========================================================================
# squash lineage CLI
# ===========================================================================


class TestLineageCli(unittest.TestCase):
    """Integration tests for the ``squash lineage`` CLI subcommand."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.model_dir = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, argv: list[str]) -> tuple[int, str]:
        """Run squash CLI with *argv* and capture stdout; return (exit_code, output)."""
        import io
        from unittest.mock import patch

        buf = io.StringIO()
        try:
            from squish.squash.cli import main as cli_main

            with patch("sys.stdout", buf), patch("sys.argv", ["squash"] + argv):
                try:
                    cli_main()
                    code = 0
                except SystemExit as e:
                    code = e.code if isinstance(e.code, int) else 0
        except ImportError:
            return -1, ""
        return code, buf.getvalue()

    def test_record_exits_zero(self) -> None:
        code, out = self._run([
            "lineage", "record", str(self.model_dir),
            "--operation", "compress",
            "--model-id", "qwen3:8b",
        ])
        self.assertEqual(code, 0, f"Unexpected exit code: {code}\n{out}")

    def test_record_prints_event_hash(self) -> None:
        code, out = self._run([
            "lineage", "record", str(self.model_dir),
            "--operation", "sign",
        ])
        self.assertEqual(code, 0)
        self.assertIn("event_hash", out.lower())

    def test_record_creates_chain_file(self) -> None:
        self._run([
            "lineage", "record", str(self.model_dir),
            "--operation", "compress",
        ])
        from squish.squash.lineage import LineageChain
        chain_file = self.model_dir / LineageChain.CHAIN_FILENAME
        self.assertTrue(chain_file.exists(), "Chain file should be created after record")

    def test_show_exits_zero_after_record(self) -> None:
        self._run(["lineage", "record", str(self.model_dir), "--operation", "compress"])
        code, out = self._run(["lineage", "show", str(self.model_dir)])
        self.assertEqual(code, 0)

    def test_show_json_flag_produces_valid_json(self) -> None:
        self._run(["lineage", "record", str(self.model_dir), "--operation", "compress"])
        code, out = self._run(["lineage", "show", str(self.model_dir), "--json"])
        self.assertEqual(code, 0)
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            self.fail(f"--json flag produced non-JSON output: {out!r}")
        # CLI outputs a raw JSON array of event dicts
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertIn("event_hash", parsed[0])

    def test_verify_intact_chain_exits_zero(self) -> None:
        self._run(["lineage", "record", str(self.model_dir), "--operation", "compress"])
        code, _ = self._run(["lineage", "verify", str(self.model_dir)])
        self.assertEqual(code, 0)

    def test_verify_tampered_chain_exits_two(self) -> None:
        from squish.squash.lineage import LineageChain

        self._run(["lineage", "record", str(self.model_dir), "--operation", "compress"])
        chain_path = self.model_dir / LineageChain.CHAIN_FILENAME
        data = json.loads(chain_path.read_text())
        data[0]["event_hash"] = "0" * 64
        chain_path.write_text(json.dumps(data), encoding="utf-8")
        code, _ = self._run(["lineage", "verify", str(self.model_dir)])
        self.assertEqual(code, 2)

    def test_verify_missing_chain_file_exits_two(self) -> None:
        code, _ = self._run(["lineage", "verify", str(self.model_dir)])
        self.assertEqual(code, 2)

    def test_record_with_params(self) -> None:
        code, out = self._run([
            "lineage", "record", str(self.model_dir),
            "--operation", "compress",
            "--params", "format=INT4", "group_size=32",
        ])
        self.assertEqual(code, 0)
        from squish.squash.lineage import LineageChain
        events = LineageChain.load(self.model_dir)
        self.assertEqual(len(events), 1)
        self.assertIn("format", events[0].params)


# ===========================================================================
# squash lineage REST API
# ===========================================================================


class TestLineageApi(unittest.TestCase):
    """Verify the three lineage REST routes are registered and functional.

    Uses the Starlette TestClient — no real server started.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from starlette.testclient import TestClient
            from squish.squash.api import app
            cls.client = TestClient(app, raise_server_exceptions=True)
        except Exception:
            cls.client = None

    def _skip_if_no_client(self) -> None:
        if self.client is None:
            self.skipTest("starlette TestClient or api.py not available")

    def test_lineage_record_route_registered(self) -> None:
        self._skip_if_no_client()
        resp = self.client.get("/lineage/record")
        self.assertNotEqual(resp.status_code, 404, "/lineage/record route not registered")

    def test_lineage_show_route_registered(self) -> None:
        self._skip_if_no_client()
        resp = self.client.get("/lineage/show", params={"model_dir": "/tmp"})
        # 404 from model_dir not found is expected; 422 or similar if dir exists — NOT 404 on route
        # A 200, 404 (missing dir), or 422 (validation) all confirm route exists
        self.assertNotEqual(resp.status_code, 404, "/lineage/show route not registered")

    def test_lineage_verify_route_registered(self) -> None:
        self._skip_if_no_client()
        resp = self.client.get("/lineage/verify")
        self.assertNotEqual(resp.status_code, 404, "/lineage/verify route not registered")

    def test_lineage_record_ok(self) -> None:
        self._skip_if_no_client()
        with tempfile.TemporaryDirectory() as tmp:
            resp = self.client.post("/lineage/record", json={
                "model_dir": tmp,
                "operation": "compress",
                "model_id": "test-model",
                "params": {"format": "INT4"},
            })
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            for key in ("event_hash", "event_id", "model_dir"):
                self.assertIn(key, data, f"Missing key {key} in response")
            self.assertEqual(len(data["event_hash"]), 64)

    def test_lineage_show_ok_after_record(self) -> None:
        self._skip_if_no_client()
        with tempfile.TemporaryDirectory() as tmp:
            self.client.post("/lineage/record", json={
                "model_dir": tmp, "operation": "compress",
            })
            resp = self.client.get("/lineage/show", params={"model_dir": tmp})
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertIn("events", data)
            self.assertIn("event_count", data)
            self.assertEqual(data["event_count"], 1)

    def test_lineage_show_404_on_missing_dir(self) -> None:
        self._skip_if_no_client()
        resp = self.client.get("/lineage/show", params={"model_dir": "/no/such/dir/xyz123"})
        self.assertEqual(resp.status_code, 404)

    def test_lineage_verify_ok_on_intact_chain(self) -> None:
        self._skip_if_no_client()
        with tempfile.TemporaryDirectory() as tmp:
            self.client.post("/lineage/record", json={
                "model_dir": tmp, "operation": "compress",
            })
            resp = self.client.post("/lineage/verify", json={"model_dir": tmp})
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertTrue(data["ok"])

    def test_lineage_verify_false_on_tampered_chain(self) -> None:
        self._skip_if_no_client()
        from squish.squash.lineage import LineageChain

        with tempfile.TemporaryDirectory() as tmp:
            self.client.post("/lineage/record", json={
                "model_dir": tmp, "operation": "compress",
            })
            chain_path = Path(tmp) / LineageChain.CHAIN_FILENAME
            data = json.loads(chain_path.read_text())
            data[0]["event_hash"] = "0" * 64
            chain_path.write_text(json.dumps(data))

            resp = self.client.post("/lineage/verify", json={"model_dir": tmp})
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resp.json()["ok"])

    def test_lineage_verify_404_on_missing_dir(self) -> None:
        self._skip_if_no_client()
        resp = self.client.post("/lineage/verify", json={"model_dir": "/no/such/dir/xyz123"})
        self.assertEqual(resp.status_code, 404)

    def test_lineage_multi_record_and_verify(self) -> None:
        """Record two events and verify the resulting chain is intact."""
        self._skip_if_no_client()
        with tempfile.TemporaryDirectory() as tmp:
            self.client.post("/lineage/record", json={
                "model_dir": tmp, "operation": "compress",
            })
            self.client.post("/lineage/record", json={
                "model_dir": tmp, "operation": "sign",
            })
            resp = self.client.post("/lineage/verify", json={"model_dir": tmp})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["ok"])
            self.assertEqual(data["event_count"], 2)


if __name__ == "__main__":
    unittest.main()
