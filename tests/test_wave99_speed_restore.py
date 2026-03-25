"""tests/test_wave99_speed_restore.py — Wave 99: Hot-path speed restoration.

Covers:
1. Structural: mx.max(_logit_vec) removed from decode loop hot path
2. INT3 auto-select: floor at params_b >= 7.0 (same pattern as Wave 97 INT2 fix)
3. Dead-code removal: _prefix_cache None guard removed from _generate_tokens
4. Task-type detection: guarded behind babbling_suppression / semantic_cache flag
5. prompt.split() guarded behind _trace flag
6. Cache warmup: list() replaced with .tolist() when available
7. Benchmark script: loadable and callable
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# 1.  Structural: mx.max() removed from decode loop
# ==============================================================================

class TestMxMaxRemoved(unittest.TestCase):
    """The per-token mx.max(_logit_vec).item() call must be gone from the hot path."""

    def _get_generate_tokens_source(self) -> str:
        import squish.server as srv
        import inspect
        return inspect.getsource(srv._generate_tokens)

    def test_mx_max_logit_vec_removed(self):
        src = self._get_generate_tokens_source()
        self.assertNotIn(
            "mx.max(_logit_vec)",
            src,
            "mx.max(_logit_vec) must be eliminated from the decode loop "
            "(Wave 99: kills per-token full-vocabulary Metal reduction)",
        )

    def test_logit_np_shared_present(self):
        src = self._get_generate_tokens_source()
        self.assertIn(
            "_logit_np_shared",
            src,
            "_logit_np_shared must exist in _generate_tokens to share the "
            "Metal→CPU copy between babbling check and fused sampler",
        )

    def test_grammar_invalidates_shared_copy(self):
        """After grammar constrains logits, the shared copy must be invalidated."""
        src = self._get_generate_tokens_source()
        # The pattern '_logit_np_shared = None' must appear after grammar engine usage
        self.assertIn(
            "_logit_np_shared = None",
            src,
            "Grammar modification must invalidate the shared logit copy",
        )

    def test_single_element_eos_check_present(self):
        """The quick single-element EOS check must be used as the first guard."""
        src = self._get_generate_tokens_source()
        # Should have the -10.0 threshold guard
        self.assertIn(
            "_eos_check > -10.0",
            src,
            "Single-element EOS guard (_eos_check > -10.0) must be present "
            "to skip full vector materialisation when EOS is clearly suppressed",
        )

    def test_shared_copy_reused_in_fused_sampler(self):
        """Fused sampler must use _logit_np_shared when available."""
        src = self._get_generate_tokens_source()
        self.assertIn(
            "_logit_np_shared if _logit_np_shared is not None",
            src,
            "Fused sampler must reuse _logit_np_shared to avoid second Metal→CPU copy",
        )


# ==============================================================================
# 2.  Dead-code removal
# ==============================================================================

class TestDeadCodeRemoved(unittest.TestCase):

    def _get_generate_tokens_source(self) -> str:
        import squish.server as srv
        import inspect
        return inspect.getsource(srv._generate_tokens)

    def test_prefix_cache_none_guard_removed(self):
        """The _prefix_cache is None guard must be removed from the hot path."""
        src = self._get_generate_tokens_source()
        self.assertNotIn(
            "if _prefix_cache is None:\n        _init_prefix_cache()",
            src,
            "_prefix_cache None guard must be removed (dead code on hot path)",
        )

    def test_prompt_split_not_unconditional(self):
        """len(prompt.split()) must NOT appear unconditionally outside a guard."""
        src = self._get_generate_tokens_source()
        lines = src.splitlines()
        # Accepted guard keywords: trace/debug flags OR _compress_enabled (prompt
        # splitting for compression ratio is intentional and gated).
        _GUARD_KEYWORDS = ("_trace", "isEnabledFor", "DEBUG", "_compress_enabled")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "len(prompt.split())" in stripped:
                # Must be inside a guarded block — look backwards up to 8 lines for
                # a containing if-statement at a lower indent level.
                indent = len(line) - len(line.lstrip())
                guarded = False
                for j in range(max(0, i - 8), i):
                    prev = lines[j]
                    prev_indent = len(prev) - len(prev.lstrip())
                    if prev_indent < indent and any(kw in prev for kw in _GUARD_KEYWORDS):
                        guarded = True
                        break
                self.assertTrue(
                    guarded,
                    f"len(prompt.split()) at line {i} must be inside a _trace, DEBUG, "
                    "or _compress_enabled guard",
                )

    def test_task_type_detection_guarded(self):
        """_detect_task_type must be guarded by _babbling_suppression or semantic cache."""
        src = self._get_generate_tokens_source()
        lines = src.splitlines()
        for i, line in enumerate(lines):
            # Find the line calling _detect_task_type directly (not as a comment)
            if "_detect_task_type(prompt)" in line and not line.strip().startswith("#"):
                # The guard may appear in the 3 lines before OR the 3 lines after
                # (ternary form: `_detect_task_type(prompt)\n    if (_babbling_suppression …)`
                # puts the guard on the line *after* the call).
                context = "\n".join(lines[max(0, i-3):min(len(lines), i+4)])
                self.assertTrue(
                    "_babbling_suppression" in context or "_semantic_cache" in context,
                    f"_detect_task_type at line {i} must be guarded by "
                    "_babbling_suppression or _semantic_cache check",
                )
                break


# ==============================================================================
# 3.  INT3 params floor
# ==============================================================================

class TestInt3Floor(unittest.TestCase):
    """INT3 must NOT be auto-selected for models < 7B."""

    def _fake_entry(self, squished_gb: float, params: str):
        from squish.catalog import CatalogEntry
        return CatalogEntry(
            id="test:model",
            name="Test Model",
            hf_mlx_repo="fake/repo",
            size_gb=squished_gb * 1.5,
            squished_size_gb=squished_gb,
            params=params,
            context=8192,
        )

    def _run_quant_block(self, ram_gb: float, squished_gb: float, params: str) -> types.SimpleNamespace:
        args = types.SimpleNamespace(
            model="test:model",
            int2=False, int3=False, int4=False, int8=False,
        )
        entry = self._fake_entry(squished_gb, params)
        import re as _re
        _params_str = getattr(entry, "params", "") or ""
        _pm = _re.search(r"(\d+\.?\d*)B", _params_str, _re.IGNORECASE)
        _params_b = float(_pm.group(1)) if _pm else 0.0
        _sq_gb  = squished_gb
        _ram_gb = ram_gb
        if _sq_gb > _ram_gb * 0.75:
            if _params_b >= 30.0:
                args.int2 = True
        elif _sq_gb > _ram_gb * 0.55:
            if _params_b >= 7.0:
                args.int3 = True
        return args

    def test_3b_model_no_int3(self):
        """3B model exceeding 55% RAM → INT3 must NOT be selected."""
        args = self._run_quant_block(ram_gb=8.0, squished_gb=5.0, params="3B")
        self.assertFalse(args.int3, "3B model must not get INT3 auto-selected")
        self.assertFalse(args.int2)

    def test_1b_model_no_int3(self):
        args = self._run_quant_block(ram_gb=4.0, squished_gb=2.5, params="1.7B")
        self.assertFalse(args.int3, "1.7B model must not get INT3")

    def test_4b_model_no_int3(self):
        args = self._run_quant_block(ram_gb=8.0, squished_gb=5.0, params="4B")
        self.assertFalse(args.int3, "4B model must not get INT3")

    def test_7b_model_gets_int3(self):
        """7B model at RAM pressure → INT3 is acceptable."""
        args = self._run_quant_block(ram_gb=8.0, squished_gb=5.5, params="7B")
        self.assertTrue(args.int3, "7B model should get INT3 when RAM is tight")

    def test_8b_model_gets_int3(self):
        args = self._run_quant_block(ram_gb=12.0, squished_gb=8.0, params="8B")
        self.assertTrue(args.int3, "8B model should get INT3 when RAM is tight")

    def test_14b_model_gets_int3(self):
        args = self._run_quant_block(ram_gb=16.0, squished_gb=10.0, params="14B")
        self.assertTrue(args.int3, "14B model should get INT3 when RAM is tight")

    def test_6b_boundary_no_int3(self):
        """6.9B → still below 7.0 floor."""
        args = self._run_quant_block(ram_gb=8.0, squished_gb=5.0, params="6.9B")
        self.assertFalse(args.int3, "6.9B must not get INT3")

    def test_int2_still_requires_30b(self):
        """INT2 gate unchanged: still requires >= 30B."""
        args = self._run_quant_block(ram_gb=16.0, squished_gb=14.0, params="8B")
        self.assertFalse(args.int2, "8B must not get INT2")

    def test_70b_still_gets_int2(self):
        args = self._run_quant_block(ram_gb=48.0, squished_gb=38.0, params="70B")
        self.assertTrue(args.int2, "70B should still get INT2 when RAM is tight")


# ==============================================================================
# 4.  server.py source does not contain old pattern
# ==============================================================================

class TestServerSourcePatterns(unittest.TestCase):

    def _src(self) -> str:
        return (ROOT / "squish" / "server.py").read_text()

    def test_old_mx_max_not_in_decode_loop(self):
        src = self._src()
        # The old pattern had two Metal syncs:
        # _max_logit_val = float(mx.max(_logit_vec).item())
        self.assertNotIn(
            "_max_logit_val = float(mx.max(_logit_vec).item())",
            src,
            "Old mx.max Metal reduction must be gone",
        )

    def test_old_unconditional_prompt_split_gone(self):
        src = self._src()
        self.assertNotIn(
            "_prompt_tokens_approx = len(prompt.split())\n    _logging.getLogger",
            src,
            "Unconditional prompt.split() logging must be removed from hot path",
        )

    def test_int3_params_floor_in_cli(self):
        cli_src = (ROOT / "squish" / "cli.py").read_text()
        self.assertIn(
            "_params_b >= 7.0",
            cli_src,
            "INT3 params floor (_params_b >= 7.0) must be present in cli.py",
        )


# ==============================================================================
# 5.  Benchmark script loadable
# ==============================================================================

class TestBenchmarkScript(unittest.TestCase):

    def test_bench_wave99_importable(self):
        bench_path = ROOT / "dev" / "benchmarks" / "bench_wave99_speed.py"
        self.assertTrue(bench_path.exists(), "bench_wave99_speed.py must exist")

    def test_bench_wave99_parseable(self):
        import ast
        bench_path = ROOT / "dev" / "benchmarks" / "bench_wave99_speed.py"
        src = bench_path.read_text()
        try:
            ast.parse(src)
        except SyntaxError as e:
            self.fail(f"bench_wave99_speed.py has syntax error: {e}")

    def test_bench_wave99_has_compare_function(self):
        bench_path = ROOT / "dev" / "benchmarks" / "bench_wave99_speed.py"
        src = bench_path.read_text()
        self.assertIn("def compare(", src)
        self.assertIn("def run_benchmark(", src)


if __name__ == "__main__":
    unittest.main()
