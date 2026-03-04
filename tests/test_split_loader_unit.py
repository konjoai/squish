"""
tests/test_split_loader_unit.py

Unit tests for pure helpers in squish/split_loader.py.
Covers _total_ram_bytes (hardware query, pure Python — no MLX required).
"""
from __future__ import annotations


from squish.split_loader import _total_ram_bytes


class TestTotalRamBytes:
    def test_returns_positive_int(self):
        result = _total_ram_bytes()
        assert isinstance(result, int)
        assert result > 0

    def test_at_least_one_gb(self):
        result = _total_ram_bytes()
        assert result >= 1 * 1024 ** 3  # at least 1 GB RAM

    def test_reasonable_upper_bound(self):
        result = _total_ram_bytes()
        # No machine has more than 16 TB RAM in 2025
        assert result < 16 * 1024 ** 4

    def test_consistent_results(self):
        result1 = _total_ram_bytes()
        result2 = _total_ram_bytes()
        assert result1 == result2, f"Expected consistent results, got {result1} and {result2}"
    
    def test_non_integer_result(self):
        result = _total_ram_bytes()
        assert isinstance(result, int), f"Expected int result, got {type(result)}"

    def test_non_positive_result(self):
        result = _total_ram_bytes()
        assert result > 0, f"Expected positive RAM bytes, got {result}"

    def test_extremely_large_result(self):
        result = _total_ram_bytes()
        assert result < 128 * 1024 ** 4, f"Expected less than 128 TB RAM, got {result}"
