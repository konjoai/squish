"""
tests/test_overnight_bench_unit.py
Pure unit + integration tests for dev/benchmarks/run_overnight_bench.py.

Test taxonomy:
  - Pure unit  — no I/O, deterministic.  Tests: _infer_quant_dir, MODEL_PLAN,
                 _BENCH_MODEL_NAME, TASK_NAMES, _TABLE_ORDER invariants.
  - Integration — uses tmp_path, tearDown implicit via pytest.  Tests:
                  _is_mlx_format, _dir_gb, load_scores_from_dir,
                  build_comparison_table.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_overnight_bench as rob


# ── Pure unit tests ──────────────────────────────────────────────────────────

class TestInferQuantDir:
    """_infer_quant_dir strips known BF16/FP16 suffixes and appends -int{bits}."""

    def test_strips_bf16(self):
        p = Path("/models/Qwen3-0.6B-bf16")
        assert rob._infer_quant_dir(p, 4).name == "Qwen3-0.6B-int4"

    def test_strips_fp16(self):
        p = Path("/models/Llama-3.2-1B-fp16")
        assert rob._infer_quant_dir(p, 3).name == "Llama-3.2-1B-int3"

    def test_strips_bf16_instruct(self):
        p = Path("/models/Qwen2.5-1.5B-Instruct-bf16")
        assert rob._infer_quant_dir(p, 2).name == "Qwen2.5-1.5B-Instruct-int2"

    def test_preserves_parent(self):
        p = Path("/my/models/gemma-3-1b-it-bf16")
        result = rob._infer_quant_dir(p, 4)
        assert result.parent == Path("/my/models")
        assert result.name == "gemma-3-1b-it-int4"

    def test_strips_bit_suffix(self):
        # handles -4bit or similar suffixes
        p = Path("/models/SomeModel-4bit")
        result = rob._infer_quant_dir(p, 3)
        assert result.name == "SomeModel-int3"

    def test_no_suffix_unchanged(self):
        # dir without recognised suffix — base stays as-is
        p = Path("/models/Qwen3-8B")
        result = rob._infer_quant_dir(p, 2)
        assert result.name == "Qwen3-8B-int2"


class TestModelPlan:
    """MODEL_PLAN structural invariants — no I/O."""

    def test_all_families_covered(self):
        families = {row[0] for row in rob.MODEL_PLAN}
        assert families == {"Qwen3-0.6B", "Llama-3.2-1B", "gemma-3-1b", "Qwen2.5-1.5B", "Qwen3-8B"}

    def test_qwen3_8b_no_int4(self):
        row = next(r for r in rob.MODEL_PLAN if r[0] == "Qwen3-8B")
        assert 4 not in row[2], "Qwen3-8B INT4 should be excluded (OOM guard)"

    def test_small_models_have_int4(self):
        for name, _, bits, _ in rob.MODEL_PLAN:
            if name != "Qwen3-8B":
                assert 4 in bits, f"{name} should have INT4 in plan"

    def test_all_have_int2_int3(self):
        for name, _, bits, _ in rob.MODEL_PLAN:
            assert 2 in bits and 3 in bits, f"{name} missing INT2 or INT3"

    def test_bf16_eval_flags(self):
        # Small models get BF16 eval, Qwen3-8B does not (too slow for BF16 run)
        for name, _, _, run_bf16 in rob.MODEL_PLAN:
            if name != "Qwen3-8B":
                assert run_bf16, f"{name}: expected run_bf16=True"

    def test_bf16_dirs_use_bf16_suffix(self):
        for _, bf16_dir, _, _ in rob.MODEL_PLAN:
            assert bf16_dir.endswith("-bf16"), f"Expected -bf16 suffix: {bf16_dir}"


class TestBenchModelName:
    """_BENCH_MODEL_NAME maps (family, bits) → correct registry display names."""

    def test_all_plan_bits_covered(self):
        for name, _, bits, run_bf16 in rob.MODEL_PLAN:
            for b in bits:
                key = (name, b)
                assert key in rob._BENCH_MODEL_NAME, f"Missing key {key} in _BENCH_MODEL_NAME"
            if run_bf16:
                assert (name, "bf16") in rob._BENCH_MODEL_NAME

    def test_qwen3_0_6b_int4(self):
        assert rob._BENCH_MODEL_NAME[("Qwen3-0.6B", 4)] == "Qwen3-0.6B-int4"

    def test_llama_int4_dir_differs(self):
        # Registry display name differs from dir-name pattern (Instruct vs non-Instruct)
        assert rob._BENCH_MODEL_NAME[("Llama-3.2-1B", 4)] == "Llama-3.2-1B-int4"

    def test_no_qwen3_8b_int4(self):
        assert ("Qwen3-8B", 4) not in rob._BENCH_MODEL_NAME

    def test_int2_entries_present(self):
        for name, _, bits, _ in rob.MODEL_PLAN:
            if 2 in bits:
                assert (name, 2) in rob._BENCH_MODEL_NAME


class TestTableOrder:
    """_TABLE_ORDER invariants — all display names must appear in _BENCH_MODEL_NAME values."""

    def test_no_unknown_names(self):
        all_values = set(rob._BENCH_MODEL_NAME.values())
        # BF16 display names not in _BENCH_MODEL_NAME values end in -bf16
        for entry in rob._TABLE_ORDER:
            if not entry.endswith("-bf16"):
                assert entry in all_values, f"_TABLE_ORDER entry '{entry}' not in _BENCH_MODEL_NAME"

    def test_no_duplicates(self):
        seen = set()
        for e in rob._TABLE_ORDER:
            assert e not in seen, f"Duplicate in _TABLE_ORDER: {e}"
            seen.add(e)

    def test_tasks_list(self):
        assert "arc_easy" in rob.TASK_NAMES
        assert "arc_challenge" in rob.TASK_NAMES
        assert len(rob.TASK_NAMES) == 6


# ── Integration tests (use tmp_path, no external services) ───────────────────

class TestIsMlxFormat:
    """_is_mlx_format detects mlx safetensors layout."""

    def test_detects_single_safetensors(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert rob._is_mlx_format(tmp_path) is True

    def test_detects_sharded_index(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_bytes(b"{}")
        assert rob._is_mlx_format(tmp_path) is True

    def test_false_for_empty_dir(self, tmp_path):
        assert rob._is_mlx_format(tmp_path) is False

    def test_false_for_squish_format(self, tmp_path):
        # squish format has manifest.json + tensors/*.npy
        (tmp_path / "manifest.json").write_bytes(b"{}")
        tensors = tmp_path / "tensors"
        tensors.mkdir()
        (tensors / "layer0.npy").write_bytes(b"")
        assert rob._is_mlx_format(tmp_path) is False

    def test_false_for_nonexistent(self, tmp_path):
        assert rob._is_mlx_format(tmp_path / "no_such_dir") is False


class TestDirGb:
    """_dir_gb sums file sizes recursively."""

    def test_empty_dir(self, tmp_path):
        assert rob._dir_gb(tmp_path) == pytest.approx(0.0)

    def test_single_file(self, tmp_path):
        data = b"x" * 1_000_000  # 1 MB
        (tmp_path / "file.bin").write_bytes(data)
        expected = 1_000_000 / 1e9
        assert rob._dir_gb(tmp_path) == pytest.approx(expected, rel=1e-6)

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.bin").write_bytes(b"a" * 500_000)
        (sub / "b.bin").write_bytes(b"b" * 500_000)
        expected = 1_000_000 / 1e9
        assert rob._dir_gb(tmp_path) == pytest.approx(expected, rel=1e-6)


class TestLoadScoresFromDir:
    """load_scores_from_dir reads lmeval_*.json files and returns {model: scores}."""

    def _write_lmeval(self, path: Path, model: str, scores: dict) -> None:
        path.write_text(json.dumps({"model": model, "scores": scores}))

    def test_single_file(self, tmp_path):
        self._write_lmeval(
            tmp_path / "lmeval_Qwen3-0.6B-int4_20260101T000000.json",
            "Qwen3-0.6B-int4",
            {"arc_easy": 37.6, "hellaswag": 28.1},
        )
        result = rob.load_scores_from_dir(tmp_path)
        assert "Qwen3-0.6B-int4" in result
        assert result["Qwen3-0.6B-int4"]["arc_easy"] == pytest.approx(37.6)

    def test_multiple_models(self, tmp_path):
        self._write_lmeval(
            tmp_path / "lmeval_ModelA_001.json", "ModelA", {"arc_easy": 50.0}
        )
        self._write_lmeval(
            tmp_path / "lmeval_ModelB_002.json", "ModelB", {"arc_easy": 60.0}
        )
        result = rob.load_scores_from_dir(tmp_path)
        assert set(result.keys()) == {"ModelA", "ModelB"}

    def test_ignores_non_lmeval_json(self, tmp_path):
        (tmp_path / "other_file.json").write_text(json.dumps({"model": "X", "scores": {}}))
        result = rob.load_scores_from_dir(tmp_path)
        assert result == {}

    def test_skips_corrupt_json(self, tmp_path):
        (tmp_path / "lmeval_broken.json").write_text("{invalid json")
        result = rob.load_scores_from_dir(tmp_path)
        assert result == {}

    def test_empty_dir(self, tmp_path):
        assert rob.load_scores_from_dir(tmp_path) == {}

    def test_later_file_wins_for_same_model(self, tmp_path):
        # Files sorted alphabetically; last one (ts=002) overwrites ts=001
        self._write_lmeval(
            tmp_path / "lmeval_ModelA_001.json", "ModelA", {"arc_easy": 40.0}
        )
        self._write_lmeval(
            tmp_path / "lmeval_ModelA_002.json", "ModelA", {"arc_easy": 45.0}
        )
        result = rob.load_scores_from_dir(tmp_path)
        assert result["ModelA"]["arc_easy"] == pytest.approx(45.0)


class TestBuildComparisonTable:
    """build_comparison_table writes a valid BENCHMARK_TABLE.md."""

    _SCORES = {
        "Qwen3-0.6B-bf16":  {"arc_easy": 50.0, "arc_challenge": 30.0, "hellaswag": 45.0,
                              "winogrande": 52.0, "piqa": 60.0, "openbookqa": 35.0},
        "Qwen3-0.6B-int4":  {"arc_easy": 48.0, "arc_challenge": 28.0, "hellaswag": 43.0,
                              "winogrande": 50.0, "piqa": 58.0, "openbookqa": 33.0},
        "Qwen3-0.6B-int2":  {"arc_easy": 25.0, "arc_challenge": 22.0, "hellaswag": 27.0,
                              "winogrande": 48.0, "piqa": 50.0, "openbookqa": 25.0},
    }
    _PLAT = {"mlx_lm": "0.30.7", "lm_eval": "0.4.11", "platform": "Darwin", "python": "3.12.8", "processor": "arm"}

    def test_creates_file(self, tmp_path):
        md = rob.build_comparison_table(self._SCORES, self._PLAT, 500, tmp_path)
        assert md.exists()
        assert md.name == "BENCHMARK_TABLE.md"

    def test_contains_model_names(self, tmp_path):
        md = rob.build_comparison_table(self._SCORES, self._PLAT, 500, tmp_path)
        content = md.read_text()
        assert "Qwen3-0.6B-bf16" in content
        assert "Qwen3-0.6B-int4" in content
        assert "Qwen3-0.6B-int2" in content

    def test_contains_delta_rows(self, tmp_path):
        md = rob.build_comparison_table(self._SCORES, self._PLAT, 500, tmp_path)
        content = md.read_text()
        assert "Δ vs BF16" in content

    def test_contains_task_headers(self, tmp_path):
        md = rob.build_comparison_table(self._SCORES, self._PLAT, 500, tmp_path)
        content = md.read_text()
        for task in rob.TASK_NAMES:
            assert task in content

    def test_platform_info_included(self, tmp_path):
        md = rob.build_comparison_table(self._SCORES, self._PLAT, 500, tmp_path)
        content = md.read_text()
        assert "0.30.7" in content  # mlx_lm version

    def test_limit_note_included(self, tmp_path):
        md = rob.build_comparison_table(self._SCORES, self._PLAT, 250, tmp_path)
        content = md.read_text()
        assert "limit=250" in content

    def test_no_delta_for_bf16_row(self, tmp_path):
        md = rob.build_comparison_table(self._SCORES, self._PLAT, 500, tmp_path)
        lines = md.read_text().splitlines()
        # The BF16 row itself should have no Δ line immediately after it
        for i, line in enumerate(lines):
            if "Qwen3-0.6B-bf16" in line and i + 1 < len(lines):
                # Next line should NOT be another entry for same model with "(+"
                assert "Δ vs BF16" not in lines[i + 1], "BF16 row should not have a delta row"
                break

    def test_empty_scores_returns_file(self, tmp_path):
        md = rob.build_comparison_table({}, self._PLAT, 500, tmp_path)
        assert md.exists()
        assert md.name == "BENCHMARK_TABLE.md"
        content = md.read_text()
        assert "Squish Quantization Accuracy Benchmark" in content
