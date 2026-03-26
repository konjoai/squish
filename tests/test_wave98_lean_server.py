"""tests/test_wave98_lean_server.py — Wave 98: Lean server memory + silent startup.

Covers:
- production_profiler.py: numpy NOT imported at module level
- production_profiler.py: record/stats/report still fully correct post-deferral
- production_profiler.py: to_json_dict, reset still work
- server.py: _ProductionProfiler is None at module level (deferred to main())
- server.py: _PROFILER_AVAILABLE is False at module level (deferred to main())
- server.py: _print_optimization_status() still exists (kept for --verbose use)
- server.py: "Optimization modules" section NOT in default startup path call
- CHANGELOG: Wave 98 entry present
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# 1.  production_profiler — numpy deferred to _compute_stats()
# ==============================================================================

class TestNumpyDeferral(unittest.TestCase):
    """Importing production_profiler must not load numpy.

    Python 3.12+ prohibits reloading C-extension modules within the same
    process (ImportError: cannot load module more than once per process).
    Both tests use a subprocess so the verification runs in a clean
    interpreter that has never loaded numpy, avoiding process poisoning.
    """

    _SCRIPT_NO_NUMPY = (
        "import sys; "
        "import squish.hardware.production_profiler; "
        "assert 'numpy' not in sys.modules, "
        "'import production_profiler must not eagerly import numpy'"
    )

    _SCRIPT_LAZY_NUMPY = (
        "import sys; "
        "import squish.hardware.production_profiler as pp; "
        "assert 'numpy' not in sys.modules, 'numpy loaded too early'; "
        "p = pp.ProductionProfiler(); p.record('x', 1.0); _ = p.stats('x'); "
        "assert 'numpy' in sys.modules, 'stats() must trigger numpy import'"
    )

    def _run_isolated(self, script: str) -> None:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Isolated check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_import_does_not_load_numpy(self):
        """Module-level import of production_profiler must not trigger numpy."""
        self._run_isolated(self._SCRIPT_NO_NUMPY)

    def test_numpy_loaded_when_stats_called(self):
        """numpy must be imported lazily when stats() is first called."""
        self._run_isolated(self._SCRIPT_LAZY_NUMPY)


# ==============================================================================
# 2.  production_profiler — correctness after deferral
# ==============================================================================

class TestProfilerCorrectness(unittest.TestCase):
    """Verify ProductionProfiler works identically after the numpy deferral."""

    def setUp(self):
        from squish.hardware.production_profiler import ProductionProfiler, ProfilerWindow
        self.profiler = ProductionProfiler(ProfilerWindow(window_size=200))

    def test_record_and_stats_single_value(self):
        self.profiler.record("op", 10.0)
        s = self.profiler.stats("op")
        self.assertEqual(s.name, "op")
        self.assertEqual(s.n_samples, 1)
        self.assertAlmostEqual(s.mean_ms, 10.0)
        self.assertAlmostEqual(s.p50_ms, 10.0)
        self.assertAlmostEqual(s.p99_ms, 10.0)
        self.assertAlmostEqual(s.min_ms, 10.0)
        self.assertAlmostEqual(s.max_ms, 10.0)

    def test_record_many_values_percentiles(self):
        for i in range(1, 101):
            self.profiler.record("lat", float(i))
        s = self.profiler.stats("lat")
        self.assertEqual(s.n_samples, 100)
        self.assertAlmostEqual(s.p50_ms, 50.5, delta=1.0)
        self.assertGreaterEqual(s.p99_ms, 98.0)

    def test_empty_window_returns_zeros(self):
        from squish.hardware.production_profiler import ProductionProfiler
        p = ProductionProfiler()
        p.record("x", 5.0)
        p.reset("x")
        s = p.stats("x")
        self.assertEqual(s.n_samples, 0)
        self.assertEqual(s.mean_ms, 0.0)
        self.assertEqual(s.p50_ms, 0.0)

    def test_report_returns_all_ops(self):
        self.profiler.record("prefill", 100.0)
        self.profiler.record("decode", 20.0)
        r = self.profiler.report()
        self.assertIn("prefill", r)
        self.assertIn("decode", r)

    def test_to_json_dict_structure(self):
        self.profiler.record("gen", 8.0)
        j = self.profiler.to_json_dict()
        self.assertIn("gen", j)
        row = j["gen"]
        for key in ("n_samples", "mean_ms", "p50_ms", "p99_ms", "p999_ms", "min_ms", "max_ms"):
            self.assertIn(key, row)
        self.assertEqual(row["n_samples"], 1)

    def test_reset_specific_op(self):
        self.profiler.record("a", 1.0)
        self.profiler.record("b", 2.0)
        self.profiler.reset("a")
        s = self.profiler.stats("a")
        self.assertEqual(s.n_samples, 0)
        self.assertEqual(self.profiler.stats("b").n_samples, 1)

    def test_reset_all(self):
        self.profiler.record("a", 1.0)
        self.profiler.record("b", 2.0)
        self.profiler.reset()
        self.assertEqual(self.profiler.operations, [])

    def test_operations_sorted(self):
        self.profiler.record("z", 1.0)
        self.profiler.record("a", 2.0)
        self.profiler.record("m", 3.0)
        self.assertEqual(self.profiler.operations, ["a", "m", "z"])

    def test_negative_latency_raises(self):
        with self.assertRaises(ValueError):
            self.profiler.record("bad", -1.0)

    def test_stats_unknown_op_raises(self):
        with self.assertRaises(KeyError):
            self.profiler.stats("nonexistent")

    def test_duplicate_op_records_accumulate(self):
        self.profiler.record("req", 5.0)
        self.profiler.record("req", 15.0)
        s = self.profiler.stats("req")
        self.assertEqual(s.n_samples, 2)
        self.assertAlmostEqual(s.mean_ms, 10.0)


# ==============================================================================
# 3.  server.py — production_profiler deferred at module level
# ==============================================================================

class TestServerProfilerDeferred(unittest.TestCase):
    """_ProductionProfiler and _PROFILER_AVAILABLE must be falsy at module level."""

    def test_production_profiler_is_none_at_module_level(self):
        import squish.server as srv
        self.assertIsNone(srv._ProductionProfiler,
                          "_ProductionProfiler must be None before main() is called")

    def test_profiler_available_is_false_at_module_level(self):
        import squish.server as srv
        self.assertFalse(srv._PROFILER_AVAILABLE,
                         "_PROFILER_AVAILABLE must be False before main() is called")

    def test_print_optimization_status_still_exists(self):
        """The function must remain for --verbose / future use."""
        import squish.server as srv
        self.assertTrue(
            callable(getattr(srv, "_print_optimization_status", None)),
            "_print_optimization_status() must still exist on squish.server",
        )


# ==============================================================================
# 4.  server.py — startup path does NOT call _print_optimization_status
# ==============================================================================

class TestOptimizationTableRemoved(unittest.TestCase):
    """Default startup must not invoke the 7-row optimization table."""

    def test_optimization_table_not_called_in_startup_path(self):
        """Grep server.py source: _print_optimization_status is not called
        from the startup-output section of main()."""
        import squish.server as srv
        import inspect
        source = inspect.getsource(srv.main)
        # The optimization table should NOT be called unconditionally —
        # it may appear in the source only in a comment or a guarded block
        # that requires _auto_prof to be set AND the model to be loaded.
        # Verify it does not appear as a bare/else call.
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "_print_optimization_status()" in stripped:
                # It should NOT appear as a bare call or inside an else clause
                # without further guarding. Check it's not in an else: block
                # (the change removed the else: _print_optimization_status() pair).
                self.fail(
                    f"_print_optimization_status() found in main() at line {i}:\n"
                    f"  {line}\n"
                    "This should have been removed from the default startup path."
                )

    def test_optimization_table_not_in_default_import_source(self):
        """Verify the removed else: block is truly gone from the startup section."""
        server_path = Path(ROOT / "squish" / "server.py")
        src = server_path.read_text()
        # The old pattern was:
        #     else:
        #         _print_optimization_status()
        # This exact pattern must no longer appear in the file.
        self.assertNotIn(
            "else:\n        _print_optimization_status()",
            src,
            "The 'else: _print_optimization_status()' block must be removed",
        )


# ==============================================================================
# 5.  CHANGELOG includes Wave 98
# ==============================================================================

class TestChangelog98(unittest.TestCase):

    def test_wave98_in_changelog(self):
        changelog = (ROOT / "CHANGELOG.md").read_text()
        self.assertIn("Wave 98", changelog)

    def test_v71_in_changelog(self):
        changelog = (ROOT / "CHANGELOG.md").read_text()
        self.assertIn("71.0.0", changelog)


if __name__ == "__main__":
    unittest.main()
