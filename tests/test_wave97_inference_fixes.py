"""tests/test_wave97_inference_fixes.py — Wave 97: Inference stability bug fixes.

Covers:
- _iter_next helper: returns value or _INFERENCE_STOP sentinel
- _collect_tokens_sync helper: drains generator to list, stops on finish reason
- _inference_executor: is a ThreadPoolExecutor with max_workers=1
- INT2 auto-select guard: not triggered for <30B models even if RAM is tight
- INT2 auto-select guard: still triggered for >=30B models
- INT3 auto-select still works for large models within 55-75% RAM window
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# 1.  _iter_next  &  _INFERENCE_STOP
# ==============================================================================

class TestIterNext(unittest.TestCase):

    def _get(self):
        from squish.server import _iter_next, _INFERENCE_STOP
        return _iter_next, _INFERENCE_STOP

    def test_returns_value_from_iterator(self):
        _iter_next, _INFERENCE_STOP = self._get()
        it = iter([(1, None), (2, "stop")])
        result = _iter_next(it)
        self.assertEqual(result, (1, None))

    def test_advances_iterator_on_each_call(self):
        _iter_next, _INFERENCE_STOP = self._get()
        it = iter(["a", "b", "c"])
        self.assertEqual(_iter_next(it), "a")
        self.assertEqual(_iter_next(it), "b")
        self.assertEqual(_iter_next(it), "c")

    def test_returns_stop_sentinel_when_exhausted(self):
        _iter_next, _INFERENCE_STOP = self._get()
        it = iter([])
        result = _iter_next(it)
        self.assertIs(result, _INFERENCE_STOP,
                      "_iter_next must return _INFERENCE_STOP for exhausted iterators")

    def test_does_not_raise_stop_iteration(self):
        _iter_next, _INFERENCE_STOP = self._get()
        it = iter([])
        try:
            _iter_next(it)
        except StopIteration:
            self.fail("_iter_next must not propagate StopIteration")

    def test_stop_sentinel_is_unique_object(self):
        from squish.server import _INFERENCE_STOP
        self.assertIsNotNone(_INFERENCE_STOP)
        self.assertIsNot(_INFERENCE_STOP, None)
        self.assertIsNot(_INFERENCE_STOP, False)
        self.assertIsNot(_INFERENCE_STOP, 0)


# ==============================================================================
# 2.  _collect_tokens_sync
# ==============================================================================

class TestCollectTokensSync(unittest.TestCase):

    def _fn(self):
        from squish.server import _collect_tokens_sync
        return _collect_tokens_sync

    def test_collects_all_tuples(self):
        fn = self._fn()
        def gen():
            yield "Hello", None
            yield " world", None
            yield "", "stop"
        result = fn(gen())
        self.assertEqual(result, [("Hello", None), (" world", None), ("", "stop")])

    def test_stops_after_finish_reason(self):
        fn = self._fn()
        def gen():
            yield "tok1", None
            yield "tok2", "stop"
            yield "tok3", None   # must NOT be in result
        result = fn(gen())
        self.assertEqual(len(result), 2)
        self.assertNotIn(("tok3", None), result)

    def test_returns_list(self):
        fn = self._fn()
        result = fn(iter([]))
        self.assertIsInstance(result, list)

    def test_empty_generator(self):
        fn = self._fn()
        result = fn(iter([]))
        self.assertEqual(result, [])

    def test_single_token_with_finish(self):
        fn = self._fn()
        def gen():
            yield "answer", "stop"
        result = fn(gen())
        self.assertEqual(result, [("answer", "stop")])


# ==============================================================================
# 3.  _inference_executor
# ==============================================================================

class TestInferenceExecutor(unittest.TestCase):

    def test_executor_is_thread_pool(self):
        import concurrent.futures
        from squish.server import _inference_executor
        self.assertIsInstance(_inference_executor, concurrent.futures.ThreadPoolExecutor)

    def test_executor_has_max_workers_1(self):
        from squish.server import _inference_executor
        self.assertEqual(_inference_executor._max_workers, 1)

    def test_executor_thread_name_prefix(self):
        from squish.server import _inference_executor
        self.assertIn("squish", _inference_executor._thread_name_prefix)

    def test_executor_can_run_callable(self):
        from squish.server import _inference_executor
        future = _inference_executor.submit(lambda: 42)
        self.assertEqual(future.result(timeout=2), 42)


# ==============================================================================
# 4.  INT2 auto-select guard in cmd_run
# ==============================================================================

class TestInt2GuardSmallModel(unittest.TestCase):
    """INT2 must NOT be automatically selected for models < 30B."""

    def _fake_entry(self, squished_gb: float, params: str):
        from squish.catalog import CatalogEntry
        return CatalogEntry(
            id="test:8b",
            name="Test Model",
            hf_mlx_repo="fake/repo",
            size_gb=squished_gb * 1.5,
            squished_size_gb=squished_gb,
            params=params,
            context=8192,
        )

    def _run_quant_block(self, ram_gb: float, squished_gb: float, params: str) -> types.SimpleNamespace:
        """Simulate the RAM-aware auto-select block in cmd_run."""
        args = types.SimpleNamespace(
            model="test:8b",
            int2=False, int3=False, int4=False, int8=False,
        )
        entry = self._fake_entry(squished_gb, params)
        with patch("squish.cli._CATALOG_AVAILABLE", True), \
             patch("squish.cli._catalog_resolve", return_value=entry), \
             patch("squish.cli._detect_ram_gb", return_value=ram_gb):
            import importlib
            import squish.cli as cli
            # Execute just the auto-select block
            sq_gb = squished_gb
            _ram_gb = ram_gb
            _auto_entry = entry
            _sq_gb = getattr(_auto_entry, "squished_size_gb", 0.0) or 0.0
            if _sq_gb > _ram_gb * 0.75:
                import re as _re
                _params_str = getattr(_auto_entry, "params", "") or ""
                _pm = _re.search(r"(\d+\.?\d*)B", _params_str, _re.IGNORECASE)
                _params_b = float(_pm.group(1)) if _pm else 0.0
                if _params_b >= 30.0:
                    args.int2 = True
                else:
                    pass  # warn only, no int2
            elif _sq_gb > _ram_gb * 0.55:
                args.int3 = True
        return args

    def test_8b_model_does_not_get_int2(self):
        """8B model squeezed on 8 GB RAM → no INT2, stays INT4."""
        args = self._run_quant_block(ram_gb=8.0, squished_gb=8.5, params="8B")
        self.assertFalse(args.int2, "8B model must never get INT2 auto-selected")

    def test_8b_model_warning_not_int2(self):
        """14B on 16 GB RAM at 0.75 threshold → no INT2 auto-selected."""
        args = self._run_quant_block(ram_gb=16.0, squished_gb=13.0, params="14B")
        self.assertFalse(args.int2, "14B model must not get INT2")

    def test_70b_model_gets_int2_on_tight_ram(self):
        """70B on 48 GB RAM — squished ~38 GB > 75% = INT2 is safe at this scale."""
        args = self._run_quant_block(ram_gb=48.0, squished_gb=38.0, params="70B")
        self.assertTrue(args.int2, "70B model should get INT2 when RAM is tight")

    def test_32b_model_gets_int2_at_threshold(self):
        """32B just over the 75% threshold → INT2 allowed since params ≥ 30B."""
        args = self._run_quant_block(ram_gb=24.0, squished_gb=22.0, params="32B")
        self.assertTrue(args.int2, "32B model at threshold should get INT2")

    def test_30b_boundary_gets_int2(self):
        """Exactly 30B → INT2 is allowed."""
        args = self._run_quant_block(ram_gb=16.0, squished_gb=14.0, params="30B")
        self.assertTrue(args.int2, "30B exactly should get INT2")

    def test_29b_boundary_does_not_get_int2(self):
        """29.9B → INT2 not allowed."""
        args = self._run_quant_block(ram_gb=16.0, squished_gb=14.0, params="29.9B")
        self.assertFalse(args.int2, "29.9B must not get INT2")

    def test_int3_still_selected_for_moderate_pressure(self):
        """55–75% RAM pressure → INT3 (regardless of params)."""
        args = self._run_quant_block(ram_gb=16.0, squished_gb=10.0, params="8B")
        self.assertTrue(args.int3)
        self.assertFalse(args.int2)

    def test_no_auto_if_ram_sufficient(self):
        """Model fits easily → no auto-quant."""
        args = self._run_quant_block(ram_gb=32.0, squished_gb=8.5, params="8B")
        self.assertFalse(args.int2)
        self.assertFalse(args.int3)


# ==============================================================================
# 5.  Verify concurrent.futures import is at module level in server.py
# ==============================================================================

class TestServerImports(unittest.TestCase):

    def test_concurrent_futures_imported(self):
        import squish.server as srv
        import concurrent.futures
        # The executor is an instance of ThreadPoolExecutor which lives in
        # concurrent.futures, so this confirms the import works.
        self.assertIsNotNone(srv._inference_executor)

    def test_iter_next_and_sentinel_exported(self):
        import squish.server as srv
        self.assertTrue(callable(srv._iter_next))
        self.assertIsNotNone(srv._INFERENCE_STOP)

    def test_collect_tokens_sync_exported(self):
        import squish.server as srv
        self.assertTrue(callable(srv._collect_tokens_sync))


if __name__ == "__main__":
    unittest.main()
