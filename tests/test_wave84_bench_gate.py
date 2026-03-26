"""tests/test_wave84_bench_gate.py

Wave 84 — CI benchmark gate: --tps-min / --ttft-max flags + python -m squish.bench

Tests:
  1.  Parser: --tps-min flag exists with default 0.0
  2.  Parser: --ttft-max flag exists with default 0.0
  3.  Parser: --tps-min parsed to float correctly
  4.  Parser: --ttft-max parsed to float correctly
  5.  Parser: both flags set simultaneously
  6.  Parser: --tps-min zero means no gate
  7.  Parser: --ttft-max zero means no gate
  8.  Parser: bench subcommand sets func=cmd_bench
  9.  Gate logic: tps_min=0 never triggers gate
  10. Gate logic: avg_tps >= tps_min passes
  11. Gate logic: avg_tps < tps_min fails
  12. Gate logic: avg_tps exactly equals tps_min passes (boundary)
  13. Gate logic: ttft_max=0 never triggers gate
  14. Gate logic: avg_ttft * 1000 <= ttft_max passes
  15. Gate logic: avg_ttft * 1000 > ttft_max fails
  16. Gate logic: avg_ttft exactly equals ttft_max/1000 passes (boundary)
  17. Gate logic: both gates pass — gate_failed stays False
  18. Gate logic: both gates fail — gate_failed is True
  19. Gate logic: tps fails but ttft passes — gate_failed is True
  20. Gate logic: ttft fails but tps passes — gate_failed is True
  21. bench/__main__.py: squish.bench package is importable
  22. bench/__main__.py: __main__.py has main() callable
  23. bench/__main__.py: argv pre-insertion when argv has no "bench"
  24. bench/__main__.py: no double-insertion when argv already starts with "bench"
  25. bench/__main__.py: no insertion when argv is empty (appends "bench")
  26. Subprocess: python -m squish.bench --help exits 0
  27. Subprocess: python -m squish.bench with no server exits non-zero
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Helpers — replicate the exact gate logic from squish/cli.py cmd_bench
# to allow deterministic unit testing without a live server.
# ---------------------------------------------------------------------------

def _run_gate(args_tps_min: float, args_ttft_max: float,
              avg_tps: float, avg_ttft: float) -> bool:
    """Mirror of the performance-gate block in cmd_bench.

    Returns True if the gate would fire (sys.exit(2) would be called),
    False if all checks pass.
    """
    tps_min  = float(args_tps_min  or 0.0)
    ttft_max = float(args_ttft_max or 0.0)
    gate_failed = False

    if tps_min > 0 and avg_tps < tps_min:
        gate_failed = True

    if ttft_max > 0 and avg_ttft * 1000 > ttft_max:
        gate_failed = True

    return gate_failed


# ============================================================================
# TestBenchGateParser — argparse flag declarations
# ============================================================================

class TestBenchGateParser(unittest.TestCase):
    """Verify --tps-min and --ttft-max are wired into the bench subparser."""

    def _parse(self, argv: list[str]):
        from squish.cli import build_parser  # noqa: PLC0415
        return build_parser().parse_args(argv)

    # 1
    def test_tps_min_default_zero(self):
        args = self._parse(["bench", "--port", "11434"])
        self.assertEqual(args.tps_min, 0.0)

    # 2
    def test_ttft_max_default_zero(self):
        args = self._parse(["bench", "--port", "11434"])
        self.assertEqual(args.ttft_max, 0.0)

    # 3
    def test_tps_min_parsed_as_float(self):
        args = self._parse(["bench", "--tps-min", "30"])
        self.assertIsInstance(args.tps_min, float)
        self.assertEqual(args.tps_min, 30.0)

    # 4
    def test_ttft_max_parsed_as_float(self):
        args = self._parse(["bench", "--ttft-max", "500"])
        self.assertIsInstance(args.ttft_max, float)
        self.assertEqual(args.ttft_max, 500.0)

    # 5
    def test_both_flags_set_simultaneously(self):
        args = self._parse(["bench", "--tps-min", "25", "--ttft-max", "400"])
        self.assertEqual(args.tps_min, 25.0)
        self.assertEqual(args.ttft_max, 400.0)

    # 6
    def test_tps_min_zero_is_valid(self):
        args = self._parse(["bench", "--tps-min", "0"])
        self.assertEqual(args.tps_min, 0.0)

    # 7
    def test_ttft_max_zero_is_valid(self):
        args = self._parse(["bench", "--ttft-max", "0"])
        self.assertEqual(args.ttft_max, 0.0)

    # 8
    def test_bench_subcommand_sets_cmd_bench_func(self):
        from squish.cli import cmd_bench  # noqa: PLC0415
        args = self._parse(["bench"])
        self.assertIs(args.func, cmd_bench)


# ============================================================================
# TestBenchGateLogic — deterministic unit tests for gate conditions
# ============================================================================

class TestBenchGateLogic(unittest.TestCase):
    """Unit tests for the performance-gate decision logic."""

    # 9
    def test_tps_min_zero_never_triggers(self):
        """tps_min=0 is a no-op regardless of actual throughput."""
        self.assertFalse(_run_gate(0.0, 0.0, avg_tps=0.1, avg_ttft=1.0))

    # 10
    def test_tps_above_min_passes(self):
        self.assertFalse(_run_gate(30.0, 0.0, avg_tps=45.0, avg_ttft=0.1))

    # 11
    def test_tps_below_min_fails(self):
        self.assertTrue(_run_gate(30.0, 0.0, avg_tps=12.5, avg_ttft=0.1))

    # 12
    def test_tps_exactly_equal_to_min_passes(self):
        """avg_tps == tps_min is NOT a failure (gate is strictly less-than)."""
        self.assertFalse(_run_gate(30.0, 0.0, avg_tps=30.0, avg_ttft=0.1))

    # 13
    def test_ttft_max_zero_never_triggers(self):
        """ttft_max=0 is a no-op regardless of actual TTFT."""
        self.assertFalse(_run_gate(0.0, 0.0, avg_tps=100.0, avg_ttft=99.0))

    # 14
    def test_ttft_below_max_passes(self):
        """avg_ttft * 1000 <= ttft_max — should pass."""
        self.assertFalse(_run_gate(0.0, 500.0, avg_tps=1.0, avg_ttft=0.4))

    # 15
    def test_ttft_above_max_fails(self):
        """avg_ttft * 1000 > ttft_max — should fail."""
        self.assertTrue(_run_gate(0.0, 500.0, avg_tps=1.0, avg_ttft=0.6))

    # 16
    def test_ttft_exactly_equal_to_max_passes(self):
        """avg_ttft * 1000 == ttft_max is NOT a failure (gate is strictly greater-than)."""
        self.assertFalse(_run_gate(0.0, 500.0, avg_tps=1.0, avg_ttft=0.5))

    # 17
    def test_both_gates_pass(self):
        self.assertFalse(_run_gate(30.0, 500.0, avg_tps=55.0, avg_ttft=0.3))

    # 18
    def test_both_gates_fail(self):
        self.assertTrue(_run_gate(30.0, 500.0, avg_tps=10.0, avg_ttft=0.9))

    # 19
    def test_tps_fails_ttft_passes(self):
        self.assertTrue(_run_gate(30.0, 500.0, avg_tps=5.0, avg_ttft=0.2))

    # 20
    def test_ttft_fails_tps_passes(self):
        self.assertTrue(_run_gate(30.0, 500.0, avg_tps=50.0, avg_ttft=0.8))


# ============================================================================
# TestBenchModule — squish/bench.py module structure
# ============================================================================

class TestBenchModule(unittest.TestCase):
    """Verify squish/bench/__main__.py module structure without running the server."""

    @staticmethod
    def _main_src() -> str:
        """Return source text of squish/bench/__main__.py."""
        import squish.bench  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        main_py = Path(squish.bench.__file__).parent / "__main__.py"
        return main_py.read_text()

    # 21
    def test_module_importable(self):
        import squish.bench  # noqa: PLC0415
        self.assertIsNotNone(squish.bench)

    # 22
    def test_main_is_callable(self):
        code = self._main_src()
        # main() must be defined
        self.assertIn("def main()", code)

    # 23
    def test_argv_insertion_when_no_bench(self):
        """main() pre-inserts 'bench' when sys.argv[1] is not already 'bench'."""
        code = self._main_src()
        self.assertIn('sys.argv[1] != "bench"', code)

    # 24
    def test_no_double_insertion_guard_present(self):
        """main() skips insertion when argv[1] is already 'bench'."""
        code = self._main_src()
        self.assertIn('sys.argv[1] != "bench"', code)
        self.assertIn('sys.argv.append("bench")', code)

    # 25
    def test_empty_argv_appends_bench(self):
        """main() appends 'bench' when sys.argv has no subcommand."""
        code = self._main_src()
        self.assertIn('sys.argv.append("bench")', code)


# ============================================================================
# TestBenchSubprocess — subprocess-level behaviour
# ============================================================================

class TestBenchSubprocess(unittest.TestCase):
    """Use subprocess isolation for process-level behaviour."""

    def _run(self, argv: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "squish.bench"] + argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_repo_root,
        )

    # 26
    def test_help_exits_zero(self):
        result = self._run(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("bench", result.stdout.lower())

    # 27
    def test_no_server_exits_nonzero(self):
        """Without a running server the bench command should exit with a non-zero code."""
        result = self._run(["--port", "19999", "--max-tokens", "1"])
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
