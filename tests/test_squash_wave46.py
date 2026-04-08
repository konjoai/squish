"""tests/test_squash_wave46.py — Wave 46: agent audit trail.

Test taxonomy:
    Pure unit — AuditEntry._compute_hash(), _hash_text(), AgentAuditLogger
                 construction, CLI parser presence, API route registration.
    Integration — AgentAuditLogger append/read_tail/verify_chain using real
                  temp files; SquashAuditCallback with injected logger;
                  chain-tamper detection; env-var log path; persistence across
                  logger instances.

Anti-mocking rule: AgentAuditLogger is never mocked when testing its own
correctness (append, chain, verify).  SquashCallback._maybe_attest() IS
patched to avoid needing a real model artifact.

Coverage:
    TestAuditEntry              — _compute_hash determinism, field coverage
    TestHashText                — _hash_text contract
    TestAgentAuditLoggerBasic   — append creates file, JSONL parseable,
                                  read_tail, read_tail on missing file
    TestAgentAuditLoggerChain   — seq increment, prev_hash chain, entry_hash
    TestAgentAuditLoggerVerify  — pristine, empty, tampered entry_hash,
                                  tampered prev_hash, bad JSON
    TestAgentAuditLoggerEnv     — SQUASH_AUDIT_LOG env var respected
    TestAgentAuditLoggerDefPath — default path is ~/.squash/audit.jsonl
    TestAgentAuditLoggerPersist — reopened logger continues chain
    TestGetAuditLoggerSingleton — get_audit_logger() returns singleton
    TestSquashAuditCallbackInit — init params, session_id, audit_logger
    TestSquashAuditCallbackLlm  — on_llm_start writes entry, correct hashes
    TestSquashAuditCallbackEnd  — on_llm_end latency_ms >= 0
    TestSquashAuditCallbackSafe — audit errors never propagate to caller
    TestAuditCli                — squash audit show / verify subcommands exist
    TestAuditApi                — GET /audit/trail route registered
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent


# ─── AuditEntry ──────────────────────────────────────────────────────────────

class TestAuditEntry(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import AuditEntry
        self.AuditEntry = AuditEntry

    def _make(self, **override):
        defaults = dict(
            seq=0, ts="2026-01-01T00:00:00Z", session_id="s1",
            event_type="llm_start", model_id="test-model",
            input_hash="aa" * 32, output_hash="", latency_ms=-1.0,
            metadata={}, prev_hash="", entry_hash="xx" * 32,
        )
        defaults.update(override)
        return self.AuditEntry(**defaults)

    def test_fields_accessible(self):
        e = self._make()
        self.assertEqual(e.seq, 0)
        self.assertEqual(e.event_type, "llm_start")

    def test_compute_hash_deterministic(self):
        h1 = self.AuditEntry._compute_hash("", 0, "llm_start", "2026-01-01T00:00:00Z", "aa", "")
        h2 = self.AuditEntry._compute_hash("", 0, "llm_start", "2026-01-01T00:00:00Z", "aa", "")
        self.assertEqual(h1, h2)

    def test_compute_hash_is_hex_64(self):
        h = self.AuditEntry._compute_hash("", 0, "x", "t", "i", "o")
        self.assertEqual(len(h), 64)
        int(h, 16)  # must be valid hex

    def test_compute_hash_differs_on_seq(self):
        h0 = self.AuditEntry._compute_hash("", 0, "x", "t", "i", "o")
        h1 = self.AuditEntry._compute_hash("", 1, "x", "t", "i", "o")
        self.assertNotEqual(h0, h1)

    def test_compute_hash_differs_on_event_type(self):
        h1 = self.AuditEntry._compute_hash("", 0, "llm_start", "t", "i", "o")
        h2 = self.AuditEntry._compute_hash("", 0, "llm_end", "t", "i", "o")
        self.assertNotEqual(h1, h2)

    def test_compute_hash_differs_on_prev(self):
        h1 = self.AuditEntry._compute_hash("prev1", 0, "x", "t", "i", "o")
        h2 = self.AuditEntry._compute_hash("prev2", 0, "x", "t", "i", "o")
        self.assertNotEqual(h1, h2)

    def test_metadata_defaults_to_dict(self):
        e = self._make(metadata={"k": "v"})
        self.assertEqual(e.metadata, {"k": "v"})


# ─── _hash_text ──────────────────────────────────────────────────────────────

class TestHashText(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import _hash_text
        self._hash_text = _hash_text

    def test_returns_64_hex(self):
        h = self._hash_text("hello")
        self.assertEqual(len(h), 64)
        int(h, 16)

    def test_deterministic(self):
        self.assertEqual(self._hash_text("abc"), self._hash_text("abc"))

    def test_empty_string(self):
        h = self._hash_text("")
        self.assertEqual(len(h), 64)

    def test_different_inputs_differ(self):
        self.assertNotEqual(self._hash_text("a"), self._hash_text("b"))

    def test_utf8_non_ascii(self):
        h = self._hash_text("ቆንጆ")
        self.assertEqual(len(h), 64)


# ─── AgentAuditLogger basic ──────────────────────────────────────────────────

class TestAgentAuditLoggerBasic(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import AgentAuditLogger
        self._td = tempfile.TemporaryDirectory()
        self.log_path = Path(self._td.name) / "audit.jsonl"
        self.logger = AgentAuditLogger(log_path=self.log_path)

    def tearDown(self):
        self._td.cleanup()

    def test_append_creates_file(self):
        self.logger.append(event_type="llm_start", model_id="m")
        self.assertTrue(self.log_path.exists())

    def test_append_returns_audit_entry(self):
        from squish.squash.governor import AuditEntry
        e = self.logger.append(event_type="llm_start", model_id="m")
        self.assertIsInstance(e, AuditEntry)

    def test_append_writes_valid_jsonl(self):
        self.logger.append(event_type="llm_start", model_id="m")
        lines = self.log_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        obj = json.loads(lines[0])
        self.assertEqual(obj["event_type"], "llm_start")
        self.assertIn("entry_hash", obj)

    def test_read_tail_returns_list(self):
        self.logger.append(event_type="llm_start", model_id="m")
        self.logger.append(event_type="llm_end", model_id="m", latency_ms=42.0)
        entries = self.logger.read_tail(10)
        self.assertEqual(len(entries), 2)

    def test_read_tail_n_limits(self):
        for _ in range(5):
            self.logger.append(event_type="llm_start", model_id="m")
        entries = self.logger.read_tail(3)
        self.assertEqual(len(entries), 3)

    def test_read_tail_on_missing_file_returns_empty(self):
        from squish.squash.governor import AgentAuditLogger
        non_existent = Path(self._td.name) / "nope.jsonl"
        l2 = AgentAuditLogger(log_path=non_existent)
        self.assertEqual(l2.read_tail(10), [])

    def test_append_metadata_stored(self):
        self.logger.append(event_type="llm_start", model_id="m", metadata={"k": "v"})
        obj = json.loads(self.log_path.read_text().strip())
        self.assertEqual(obj["metadata"]["k"], "v")

    def test_path_property(self):
        self.assertEqual(self.logger.path, self.log_path)


# ─── AgentAuditLogger hash chain ─────────────────────────────────────────────

class TestAgentAuditLoggerChain(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import AgentAuditLogger
        self._td = tempfile.TemporaryDirectory()
        self.log_path = Path(self._td.name) / "audit.jsonl"
        self.logger = AgentAuditLogger(log_path=self.log_path)

    def tearDown(self):
        self._td.cleanup()

    def test_seq_increments(self):
        e0 = self.logger.append(event_type="llm_start", model_id="m")
        e1 = self.logger.append(event_type="llm_end", model_id="m")
        self.assertEqual(e0.seq, 0)
        self.assertEqual(e1.seq, 1)

    def test_first_entry_prev_hash_empty(self):
        e = self.logger.append(event_type="llm_start", model_id="m")
        self.assertEqual(e.prev_hash, "")

    def test_second_entry_prev_hash_is_first_entry_hash(self):
        e0 = self.logger.append(event_type="llm_start", model_id="m")
        e1 = self.logger.append(event_type="llm_end", model_id="m")
        self.assertEqual(e1.prev_hash, e0.entry_hash)

    def test_entry_hash_matches_compute_hash(self):
        from squish.squash.governor import AuditEntry
        e = self.logger.append(event_type="llm_start", model_id="m", input_hash="aa")
        expected = AuditEntry._compute_hash(
            e.prev_hash, e.seq, e.event_type, e.ts, e.input_hash, e.output_hash
        )
        self.assertEqual(e.entry_hash, expected)

    def test_five_entry_chain(self):
        entries = [self.logger.append(event_type="llm_start", model_id="m") for _ in range(5)]
        for i, e in enumerate(entries):
            self.assertEqual(e.seq, i)
        for i in range(1, 5):
            self.assertEqual(entries[i].prev_hash, entries[i - 1].entry_hash)

    def test_thread_safe_seq(self):
        results = []
        lock = threading.Lock()

        def worker():
            e = self.logger.append(event_type="llm_start", model_id="m")
            with lock:
                results.append(e.seq)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results), list(range(10)))


# ─── AgentAuditLogger verify_chain ───────────────────────────────────────────

class TestAgentAuditLoggerVerify(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import AgentAuditLogger
        self._td = tempfile.TemporaryDirectory()
        self.log_path = Path(self._td.name) / "audit.jsonl"
        self.logger = AgentAuditLogger(log_path=self.log_path)

    def tearDown(self):
        self._td.cleanup()

    def test_pristine_log_verifies(self):
        for _ in range(3):
            self.logger.append(event_type="llm_start", model_id="m")
        ok, msg = self.logger.verify_chain()
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_empty_log_verifies(self):
        from squish.squash.governor import AgentAuditLogger
        empty = Path(self._td.name) / "empty.jsonl"
        l2 = AgentAuditLogger(log_path=empty)
        ok, msg = l2.verify_chain()
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_missing_file_verifies(self):
        ok, msg = self.logger.verify_chain()
        self.assertTrue(ok)

    def test_tampered_entry_hash_detected(self):
        self.logger.append(event_type="llm_start", model_id="m")
        self.logger.append(event_type="llm_end", model_id="m")
        lines = self.log_path.read_text().splitlines()
        obj = json.loads(lines[0])
        obj["entry_hash"] = "00" * 32  # corrupt
        lines[0] = json.dumps(obj)
        self.log_path.write_text("\n".join(lines))
        ok, msg = self.logger.verify_chain()
        self.assertFalse(ok)
        self.assertIn("entry_hash mismatch", msg)

    def test_tampered_prev_hash_detected(self):
        self.logger.append(event_type="llm_start", model_id="m")
        self.logger.append(event_type="llm_end", model_id="m")
        lines = self.log_path.read_text().splitlines()
        obj = json.loads(lines[1])
        obj["prev_hash"] = "00" * 32
        lines[1] = json.dumps(obj)
        self.log_path.write_text("\n".join(lines))
        ok, msg = self.logger.verify_chain()
        self.assertFalse(ok)

    def test_invalid_json_line_detected(self):
        self.logger.append(event_type="llm_start", model_id="m")
        with self.log_path.open("a") as f:
            f.write("\nNOT JSON\n")
        ok, msg = self.logger.verify_chain()
        self.assertFalse(ok)
        self.assertIn("invalid JSON", msg)


# ─── AgentAuditLogger env var + default path ─────────────────────────────────

class TestAgentAuditLoggerEnv(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._td.cleanup()

    def test_env_var_sets_log_path(self):
        from squish.squash.governor import AgentAuditLogger
        custom = str(Path(self._td.name) / "custom.jsonl")
        with patch.dict(os.environ, {"SQUASH_AUDIT_LOG": custom}):
            logger = AgentAuditLogger()
        self.assertEqual(str(logger.path), custom)

    def test_default_path_is_squash_audit(self):
        from squish.squash.governor import AgentAuditLogger
        with patch.dict(os.environ, {}, clear=False):
            env_backup = os.environ.pop("SQUASH_AUDIT_LOG", None)
            try:
                logger = AgentAuditLogger()
                self.assertTrue(str(logger.path).endswith("audit.jsonl"))
                self.assertIn(".squash", str(logger.path))
            finally:
                if env_backup is not None:
                    os.environ["SQUASH_AUDIT_LOG"] = env_backup


# ─── AgentAuditLogger persistence ────────────────────────────────────────────

class TestAgentAuditLoggerPersist(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import AgentAuditLogger
        self._td = tempfile.TemporaryDirectory()
        self.log_path = Path(self._td.name) / "audit.jsonl"
        l1 = AgentAuditLogger(log_path=self.log_path)
        l1.append(event_type="llm_start", model_id="m")
        l1.append(event_type="llm_end", model_id="m")
        self.AgentAuditLogger = AgentAuditLogger

    def tearDown(self):
        self._td.cleanup()

    def test_reopened_logger_continues_seq(self):
        l2 = self.AgentAuditLogger(log_path=self.log_path)
        e = l2.append(event_type="llm_start", model_id="m")
        self.assertEqual(e.seq, 2)

    def test_reopened_logger_continues_hash_chain(self):
        entries_before = self.AgentAuditLogger(log_path=self.log_path).read_tail(10)
        last_hash = entries_before[-1]["entry_hash"]
        l2 = self.AgentAuditLogger(log_path=self.log_path)
        e = l2.append(event_type="llm_start", model_id="m")
        self.assertEqual(e.prev_hash, last_hash)

    def test_full_chain_verifies_after_persistence(self):
        l2 = self.AgentAuditLogger(log_path=self.log_path)
        l2.append(event_type="llm_start", model_id="m")
        ok, msg = l2.verify_chain()
        self.assertTrue(ok, msg)


# ─── get_audit_logger singleton ──────────────────────────────────────────────

class TestGetAuditLoggerSingleton(unittest.TestCase):
    def test_returns_agent_audit_logger(self):
        import squish.squash.governor as gov
        from squish.squash.governor import AgentAuditLogger, get_audit_logger
        old = gov._AUDIT_LOGGER
        gov._AUDIT_LOGGER = None
        try:
            inst = get_audit_logger()
            self.assertIsInstance(inst, AgentAuditLogger)
        finally:
            gov._AUDIT_LOGGER = old

    def test_singleton_same_instance(self):
        import squish.squash.governor as gov
        from squish.squash.governor import get_audit_logger
        old = gov._AUDIT_LOGGER
        gov._AUDIT_LOGGER = None
        try:
            a = get_audit_logger()
            b = get_audit_logger()
            self.assertIs(a, b)
        finally:
            gov._AUDIT_LOGGER = old


# ─── SquashAuditCallback init ────────────────────────────────────────────────

class TestSquashAuditCallbackInit(unittest.TestCase):
    def _cb(self, **kwargs):
        from squish.squash.governor import AgentAuditLogger
        from squish.squash.integrations.langchain import SquashAuditCallback
        td = tempfile.mkdtemp()
        logger = AgentAuditLogger(log_path=Path(td) / "t.jsonl")
        return SquashAuditCallback(
            Path("/tmp/model"),
            session_id="test-session",
            audit_logger=logger,
            **kwargs,
        ), logger, td

    def test_session_id_stored(self):
        cb, _, td = self._cb()
        self.assertEqual(cb._session_id, "test-session")

    def test_custom_logger_stored(self):
        from squish.squash.governor import AgentAuditLogger
        cb, logger, td = self._cb()
        self.assertIs(cb._audit_logger, logger)

    def test_inherits_squash_callback(self):
        from squish.squash.integrations.langchain import SquashCallback, SquashAuditCallback
        cb, _, td = self._cb()
        self.assertIsInstance(cb, SquashCallback)


# ─── SquashAuditCallback on_llm_start ────────────────────────────────────────

class TestSquashAuditCallbackLlm(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import AgentAuditLogger, _hash_text
        from squish.squash.integrations.langchain import SquashAuditCallback
        self._td = tempfile.TemporaryDirectory()
        self.log_path = Path(self._td.name) / "audit.jsonl"
        self.logger = AgentAuditLogger(log_path=self.log_path)
        self._hash_text = _hash_text
        self.cb = SquashAuditCallback(
            Path("/tmp/model"),
            session_id="s1",
            audit_logger=self.logger,
        )

    def tearDown(self):
        self._td.cleanup()

    def _patch_attest(self):
        return patch.object(self.cb, "_maybe_attest")

    def test_on_llm_start_writes_entry(self):
        with self._patch_attest():
            self.cb.on_llm_start({}, ["hello world"])
        entries = self.logger.read_tail(10)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event_type"], "llm_start")

    def test_on_llm_start_input_hash_correct(self):
        with self._patch_attest():
            self.cb.on_llm_start({}, ["hello", "world"])
        entries = self.logger.read_tail(10)
        expected = self._hash_text("hello\nworld")
        self.assertEqual(entries[0]["input_hash"], expected)

    def test_on_llm_start_model_id_correct(self):
        with self._patch_attest():
            self.cb.on_llm_start({}, ["x"])
        self.assertEqual(self.logger.read_tail(1)[0]["model_id"], "/tmp/model")

    def test_on_chat_model_start_writes_entry(self):
        with self._patch_attest():
            self.cb.on_chat_model_start({}, [["msg1", "msg2"]])
        entries = self.logger.read_tail(10)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event_type"], "llm_start")


# ─── SquashAuditCallback on_llm_end ──────────────────────────────────────────

class TestSquashAuditCallbackEnd(unittest.TestCase):
    def setUp(self):
        from squish.squash.governor import AgentAuditLogger
        from squish.squash.integrations.langchain import SquashAuditCallback
        self._td = tempfile.TemporaryDirectory()
        self.log_path = Path(self._td.name) / "audit.jsonl"
        self.logger = AgentAuditLogger(log_path=self.log_path)
        self.cb = SquashAuditCallback(
            Path("/tmp/model"),
            session_id="s1",
            audit_logger=self.logger,
        )

    def tearDown(self):
        self._td.cleanup()

    def test_on_llm_end_writes_llm_end_entry(self):
        with patch.object(self.cb, "_maybe_attest"):
            self.cb.on_llm_start({}, ["hi"])
        fake_response = MagicMock()
        fake_response.generations = [["token"]]
        self.cb.on_llm_end(fake_response)
        entries = self.logger.read_tail(10)
        types = [e["event_type"] for e in entries]
        self.assertIn("llm_end", types)

    def test_on_llm_end_latency_non_negative(self):
        with patch.object(self.cb, "_maybe_attest"):
            self.cb.on_llm_start({}, ["hi"])
        time.sleep(0.01)
        fake_response = MagicMock()
        fake_response.generations = [[]]
        self.cb.on_llm_end(fake_response)
        entry = next(e for e in self.logger.read_tail(10) if e["event_type"] == "llm_end")
        self.assertGreaterEqual(entry["latency_ms"], 0.0)

    def test_chain_valid_after_start_end(self):
        with patch.object(self.cb, "_maybe_attest"):
            self.cb.on_llm_start({}, ["hi"])
        fake_response = MagicMock()
        fake_response.generations = [[]]
        self.cb.on_llm_end(fake_response)
        ok, msg = self.logger.verify_chain()
        self.assertTrue(ok, msg)


# ─── SquashAuditCallback never raises ────────────────────────────────────────

class TestSquashAuditCallbackSafe(unittest.TestCase):
    def setUp(self):
        from squish.squash.integrations.langchain import SquashAuditCallback
        self._td = tempfile.TemporaryDirectory()
        self.cb = SquashAuditCallback(Path("/tmp/model"), session_id="s")

    def tearDown(self):
        self._td.cleanup()

    def _broken_logger(self):
        m = MagicMock()
        m.append.side_effect = RuntimeError("disk full")
        return m

    def test_on_llm_start_never_raises_on_logger_error(self):
        self.cb._audit_logger = self._broken_logger()
        with patch.object(self.cb, "_maybe_attest"):
            try:
                self.cb.on_llm_start({}, ["hi"])
            except Exception as exc:
                self.fail(f"on_llm_start raised unexpectedly: {exc}")

    def test_on_llm_end_never_raises_on_logger_error(self):
        self.cb._audit_logger = self._broken_logger()
        fake = MagicMock()
        fake.generations = [[]]
        try:
            self.cb.on_llm_end(fake)
        except Exception as exc:
            self.fail(f"on_llm_end raised unexpectedly: {exc}")


# ─── CLI: squash audit subcommand ────────────────────────────────────────────

class TestAuditCli(unittest.TestCase):
    def _parser(self):
        from squish.squash.cli import _build_parser
        return _build_parser()

    def test_audit_subcommand_registered(self):
        p = self._parser()
        # Should not raise
        args = p.parse_args(["audit", "show"])
        self.assertEqual(args.command, "audit")

    def test_audit_show_default_n(self):
        p = self._parser()
        args = p.parse_args(["audit", "show"])
        self.assertIsInstance(args.n, int)
        self.assertGreater(args.n, 0)

    def test_audit_show_n_override(self):
        p = self._parser()
        args = p.parse_args(["audit", "show", "--n", "50"])
        self.assertEqual(args.n, 50)

    def test_audit_show_json_flag(self):
        p = self._parser()
        args = p.parse_args(["audit", "show", "--json"])
        self.assertTrue(args.json_output)

    def test_audit_verify_registered(self):
        p = self._parser()
        args = p.parse_args(["audit", "verify"])
        self.assertEqual(args.command, "audit")
        self.assertEqual(args.audit_command, "verify")

    def test_audit_show_log_override(self):
        p = self._parser()
        args = p.parse_args(["audit", "show", "--log", "/tmp/other.jsonl"])
        self.assertEqual(args.log, "/tmp/other.jsonl")

    def test_audit_verify_log_override(self):
        p = self._parser()
        args = p.parse_args(["audit", "verify", "--log", "/tmp/x.jsonl"])
        self.assertEqual(args.log, "/tmp/x.jsonl")

    def test_audit_cmd_show_exit_0_empty(self):
        """squash audit show on an empty log should exit 0."""
        from squish.squash.cli import _build_parser, _cmd_audit
        import argparse
        p = self._parser()
        with tempfile.TemporaryDirectory() as td:
            log_path = str(Path(td) / "a.jsonl")
            args = p.parse_args(["audit", "show", "--log", log_path])
            ret = _cmd_audit(args, quiet=True)
        self.assertEqual(ret, 0)

    def test_audit_cmd_verify_exit_0_missing(self):
        """squash audit verify on missing file should exit 0 (empty chain is valid)."""
        from squish.squash.cli import _build_parser, _cmd_audit
        with tempfile.TemporaryDirectory() as td:
            log_path = str(Path(td) / "nonexistent.jsonl")
            p = self._parser()
            args = p.parse_args(["audit", "verify", "--log", log_path])
            ret = _cmd_audit(args, quiet=True)
        self.assertEqual(ret, 0)

    def test_audit_cmd_verify_exit_2_tampered(self):
        """squash audit verify on tampered log should exit 2."""
        from squish.squash.governor import AgentAuditLogger
        from squish.squash.cli import _build_parser, _cmd_audit
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "audit.jsonl"
            logger = AgentAuditLogger(log_path=log_path)
            logger.append(event_type="llm_start", model_id="m")
            # tamper
            obj = json.loads(log_path.read_text().strip())
            obj["entry_hash"] = "00" * 32
            log_path.write_text(json.dumps(obj))
            p = self._parser()
            args = p.parse_args(["audit", "verify", "--log", str(log_path)])
            ret = _cmd_audit(args, quiet=True)
        self.assertEqual(ret, 2)


# ─── API: GET /audit/trail ────────────────────────────────────────────────────

class TestAuditApi(unittest.TestCase):
    def _client(self):
        from starlette.testclient import TestClient
        from squish.squash.api import app
        return TestClient(app)

    def test_get_audit_trail_route_registered(self):
        client = self._client()
        # Even on a fresh boot with no log, the route must respond (not 404)
        resp = client.get("/audit/trail")
        self.assertNotEqual(resp.status_code, 404)

    def test_get_audit_trail_returns_json_keys(self):
        client = self._client()
        resp = client.get("/audit/trail")
        if resp.status_code == 200:
            data = resp.json()
            self.assertIn("count", data)
            self.assertIn("log_path", data)
            self.assertIn("entries", data)

    def test_get_audit_trail_limit_param_accepted(self):
        client = self._client()
        resp = client.get("/audit/trail?limit=10")
        self.assertNotEqual(resp.status_code, 422)

    def test_get_audit_trail_log_param_accepted(self):
        client = self._client()
        with tempfile.TemporaryDirectory() as td:
            log = str(Path(td) / "t.jsonl")
            resp = client.get(f"/audit/trail?log={log}")
        self.assertNotEqual(resp.status_code, 422)

    def test_get_audit_trail_entries_len_bounded_by_limit(self):
        from squish.squash.governor import AgentAuditLogger
        client = self._client()
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "audit.jsonl"
            logger = AgentAuditLogger(log_path=log_path)
            for _ in range(20):
                logger.append(event_type="llm_start", model_id="m")
            resp = client.get(f"/audit/trail?limit=5&log={log_path}")
            if resp.status_code == 200:
                data = resp.json()
                self.assertLessEqual(len(data["entries"]), 5)
