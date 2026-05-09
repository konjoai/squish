"""tests/test_wave108_calculator.py — Wave 108: dashboard v2 savings calculator.

Covers:
- demo.server._arch_for_params:        bracket lookup is monotone + total
- demo.server._bits_to_mode:            valid bits map → modes; invalid raises
- demo.server._calculate:               pure-arithmetic correctness vs
                                        estimate_kv_memory; weights formula;
                                        FP16 baseline; total_compression_ratio.
- /api/calculate endpoint:              200 path returns the right keys + values
- /api/calculate endpoint:              4xx on bad params (negative ctx,
                                        missing model_params_b, bad bits).
- Architecture lookup vs JS table:      _ARCH_TABLE in server.py matches the
                                        ARCH_TABLE in demo/index.html exactly.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo import server as demo_server                              # noqa: E402

from squish.kv.kv_cache import estimate_kv_memory                   # noqa: E402


# ==============================================================================
# 1.  _arch_for_params — bracket lookup
# ==============================================================================


class TestArchLookup(unittest.TestCase):
    """The arch table must be total, monotone, and pin to known architectures."""

    def test_known_models_map_to_realistic_arch(self):
        # Qwen2.5-1.5B: lands in the 1.5B–2B bracket.
        a15 = demo_server._arch_for_params(1.5)
        self.assertEqual(a15["n_layers"],   28)
        self.assertEqual(a15["n_kv_heads"],  2)
        self.assertEqual(a15["head_dim"],  128)

        # Qwen2.5-7B: lands in the 7B–8B bracket.
        a7 = demo_server._arch_for_params(7.0)
        self.assertEqual(a7["n_layers"],   32)
        self.assertEqual(a7["n_kv_heads"],  8)
        self.assertEqual(a7["head_dim"],  128)

        # 70B: lands in the 70B bracket.
        a70 = demo_server._arch_for_params(70.0)
        self.assertEqual(a70["n_layers"],   80)
        self.assertEqual(a70["n_kv_heads"],  8)

    def test_arch_table_is_total_over_supported_range(self):
        # No bracket should ever raise on an in-range params_b.
        for b in (0.05, 0.5, 1.0, 1.5, 3.0, 7.0, 13.0, 30.0, 70.0, 200.0, 999.9):
            arch = demo_server._arch_for_params(b)
            self.assertIn("n_layers",   arch)
            self.assertIn("n_kv_heads", arch)
            self.assertIn("head_dim",   arch)
            self.assertIn("label",      arch)

    def test_arch_n_layers_is_monotone_non_decreasing(self):
        # Doubling params should never decrease layer count — sanity bound.
        last = 0
        for b in (0.05, 0.5, 1.0, 2.0, 4.0, 8.0, 13.0, 30.0, 70.0, 999.0):
            n = demo_server._arch_for_params(b)["n_layers"]
            self.assertGreaterEqual(n, last, f"non-monotone at {b}B → {n} layers")
            last = n


# ==============================================================================
# 2.  _bits_to_mode
# ==============================================================================


class TestBitsToMode(unittest.TestCase):

    def test_valid_bits(self):
        self.assertEqual(demo_server._bits_to_mode(16), "fp16")
        self.assertEqual(demo_server._bits_to_mode( 8), "int8")
        self.assertEqual(demo_server._bits_to_mode( 4), "int4")
        self.assertEqual(demo_server._bits_to_mode( 2), "int2")

    def test_invalid_bits_raise(self):
        for bad in (0, 1, 3, 5, 6, 7, 9, 12, 32):
            with self.assertRaises(ValueError):
                demo_server._bits_to_mode(bad)


# ==============================================================================
# 3.  _calculate — closed-form vs estimate_kv_memory + weights formula
# ==============================================================================


class TestCalculate(unittest.TestCase):
    """The endpoint's math must match the production formula bit-for-bit.

    estimate_kv_memory is what production callers use via make_kv_cache —
    the public-facing calculator must agree with it exactly, not just
    "approximately".  Any drift means the dashboard would lie.
    """

    def test_kv_arithmetic_matches_production_formula(self):
        # 7B-class arch · 32K context · INT4 (sweet spot for this length).
        result = demo_server._calculate(
            model_params_b=7.0, context_len=32_000, precision_bits=4,
        )
        arch = result["arch"]
        ref  = estimate_kv_memory(
            n_layers=arch["n_layers"], n_kv_heads=arch["n_kv_heads"],
            head_dim=arch["head_dim"], context_tokens=32_000, mode="int4",
        )
        self.assertEqual(result["kv_cache_bytes"],      ref.total_bytes)
        self.assertEqual(result["kv_cache_fp16_bytes"], ref.fp16_baseline_bytes)
        self.assertAlmostEqual(
            result["kv_compression_ratio"], ref.compression_ratio, places=6,
        )

    def test_weights_bytes_formula(self):
        # weights_bytes == params * bits / 8  (integer division).
        for params_b, bits in [(7.0, 4), (1.5, 8), (70.0, 2), (3.0, 16)]:
            r = demo_server._calculate(params_b, 1_024, bits)
            params = int(params_b * 1e9)
            self.assertEqual(r["weights_bytes"], params * bits // 8)
            self.assertEqual(r["weights_fp16_bytes"], params * 2)

    def test_total_and_savings_invariants(self):
        r = demo_server._calculate(7.0, 16_384, 2)
        self.assertEqual(
            r["total_bytes"],
            r["weights_bytes"] + r["kv_cache_bytes"],
        )
        self.assertEqual(
            r["fp16_total_bytes"],
            r["weights_fp16_bytes"] + r["kv_cache_fp16_bytes"],
        )
        # INT2 must save memory vs FP16 — never zero, never negative.
        self.assertGreater(r["savings_bytes"], 0)
        self.assertEqual(
            r["savings_bytes"], r["fp16_total_bytes"] - r["total_bytes"],
        )
        self.assertGreater(r["total_compression_ratio"], 1.0)

    def test_fp16_returns_neutral_compression(self):
        # FP16 vs FP16 baseline must have zero savings and ratio=1.
        r = demo_server._calculate(7.0, 8_192, 16)
        self.assertEqual(r["savings_bytes"], 0)
        self.assertEqual(r["weights_bytes"], r["weights_fp16_bytes"])
        self.assertEqual(r["kv_cache_bytes"], r["kv_cache_fp16_bytes"])
        self.assertAlmostEqual(r["total_compression_ratio"], 1.0, places=6)

    def test_compression_ratio_ordering_int8_lt_int4_lt_int2(self):
        # By construction (more bits saved → bigger ratio): int2 > int4 > int8.
        r8 = demo_server._calculate(7.0, 8_192, 8)
        r4 = demo_server._calculate(7.0, 8_192, 4)
        r2 = demo_server._calculate(7.0, 8_192, 2)
        self.assertLess(
            r8["total_compression_ratio"],
            r4["total_compression_ratio"],
            "int8 must compress less than int4",
        )
        self.assertLess(
            r4["total_compression_ratio"],
            r2["total_compression_ratio"],
            "int4 must compress less than int2",
        )

    def test_calculate_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            demo_server._calculate(0.0, 1024, 4)            # too small
        with self.assertRaises(ValueError):
            demo_server._calculate(1e9, 1024, 4)            # absurd
        with self.assertRaises(ValueError):
            demo_server._calculate(7.0, 0, 4)               # zero ctx
        with self.assertRaises(ValueError):
            demo_server._calculate(7.0, -1, 4)              # negative ctx
        with self.assertRaises(ValueError):
            demo_server._calculate(7.0, 1024, 5)            # bad bits


# ==============================================================================
# 4.  /api/calculate — full HTTP round-trip on a real ephemeral server
# ==============================================================================


class _ServerThread:
    """Spin demo_server.Handler on a free port for the duration of one test."""

    def __init__(self):
        from http.server import HTTPServer
        self.httpd = HTTPServer(("127.0.0.1", 0), demo_server.Handler)
        self.port  = self.httpd.server_address[1]
        self.t     = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestCalculateEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = _ServerThread()

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()

    def _post(self, path: str, body: dict):
        c = HTTPConnection("127.0.0.1", self.srv.port, timeout=5)
        c.request(
            "POST", path, body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        r = c.getresponse()
        data = r.read().decode("utf-8")
        c.close()
        return r.status, json.loads(data) if data else {}

    def test_happy_path(self):
        status, body = self._post(
            "/api/calculate",
            {"model_params_b": 7.0, "context_len": 32_000, "precision_bits": 4},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["live"], True)
        self.assertEqual(body["mode"], "int4")
        self.assertEqual(body["precision_bits"], 4)
        for key in (
            "weights_bytes", "weights_fp16_bytes",
            "kv_cache_bytes", "kv_cache_fp16_bytes",
            "total_bytes", "fp16_total_bytes",
            "savings_bytes", "total_compression_ratio",
            "arch",
        ):
            self.assertIn(key, body, f"missing key {key!r}")
        self.assertEqual(
            body["total_bytes"],
            body["weights_bytes"] + body["kv_cache_bytes"],
        )

    def test_missing_params_returns_400(self):
        status, body = self._post(
            "/api/calculate", {"context_len": 1024, "precision_bits": 4},
        )
        self.assertEqual(status, 400)
        self.assertIn("model_params_b", body["error"])

    def test_bad_bits_returns_400(self):
        status, body = self._post(
            "/api/calculate",
            {"model_params_b": 7.0, "context_len": 1024, "precision_bits": 5},
        )
        self.assertEqual(status, 400)
        self.assertIn("precision_bits", body["error"])

    def test_negative_context_returns_400(self):
        status, body = self._post(
            "/api/calculate",
            {"model_params_b": 7.0, "context_len": -1, "precision_bits": 4},
        )
        self.assertEqual(status, 400)

    def test_alias_path_calculate_works(self):
        # /calculate is an alias of /api/calculate (per W108 spec).
        status, body = self._post(
            "/calculate",
            {"model_params_b": 1.5, "context_len": 4_096, "precision_bits": 8},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["mode"], "int8")


# ==============================================================================
# 5.  Architecture lookup parity — server.py table must match index.html JS
# ==============================================================================


class TestArchTableJsParity(unittest.TestCase):
    """The JS ARCH_TABLE in demo/index.html must mirror server._ARCH_TABLE.

    If they drift, the projected numbers shown before the user types into
    the form (client-side) won't match what the server returns, which makes
    the dashboard look broken.
    """

    def test_index_html_arch_rows_match_server(self):
        html = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
        # Pull every JS arch row.  Format is:
        #     { cap: <num>, n_layers: <int>, n_kv_heads: <int>, head_dim: <int>, label: '<...>' },
        pattern = re.compile(
            r"\{\s*cap:\s*(?P<cap>[\d.]+|Infinity)\s*,"
            r"\s*n_layers:\s*(?P<n_layers>\d+)\s*,"
            r"\s*n_kv_heads:\s*(?P<n_kv_heads>\d+)\s*,"
            r"\s*head_dim:\s*(?P<head_dim>\d+)\s*,"
            r"\s*label:\s*'(?P<label>[^']+)'"
        )
        js_rows = [m.groupdict() for m in pattern.finditer(html)]
        self.assertEqual(
            len(js_rows), len(demo_server._ARCH_TABLE),
            "JS arch row count does not match server _ARCH_TABLE",
        )
        for js, py in zip(js_rows, demo_server._ARCH_TABLE):
            py_cap, py_layers, py_heads, py_dim, py_label = py
            js_cap = float("inf") if js["cap"] == "Infinity" else float(js["cap"])
            self.assertEqual(js_cap,                py_cap)
            self.assertEqual(int(js["n_layers"]),   py_layers)
            self.assertEqual(int(js["n_kv_heads"]), py_heads)
            self.assertEqual(int(js["head_dim"]),   py_dim)
            self.assertEqual(js["label"],           py_label)


if __name__ == "__main__":
    unittest.main()
