"""
tests/test_phase_e_cli.py

Coverage tests for Phase E2/E3 additions to squish/cli.py:
  - _apply_dare_sparsification
  - _find_adapter_safetensors
  - cmd_train_adapter
  - cmd_merge_model
  - train-adapter and merge-model subparser wiring
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _import_cli():
    import squish.cli as cli  # noqa: PLC0415
    return cli


# ── _apply_dare_sparsification ────────────────────────────────────────────────


class TestApplyDareSparsification:
    def test_no_safetensors_warns_and_returns(self, tmp_path: Path, capsys):
        """When safetensors is not available, prints warning and returns early."""
        cli = _import_cli()
        with patch.dict(sys.modules, {"safetensors": None, "safetensors.numpy": None}):
            cli._apply_dare_sparsification(tmp_path)
        out = capsys.readouterr().out
        assert "safetensors not available" in out

    def test_no_matching_files_no_op(self, tmp_path: Path):
        """Empty directory: loop body never executes, no load/save calls."""
        cli = _import_cli()
        mock_stn = types.ModuleType("safetensors.numpy")
        mock_stn.load_file = MagicMock()
        mock_stn.save_file = MagicMock()
        # Ensure the module is seen when imported from inside the function
        with patch.dict(sys.modules, {"safetensors": MagicMock(), "safetensors.numpy": mock_stn}):
            cli._apply_dare_sparsification(tmp_path)
        mock_stn.load_file.assert_not_called()
        mock_stn.save_file.assert_not_called()

    def test_processes_matching_files(self, tmp_path: Path, capsys):
        """Files matching adapter_model*.safetensors are sparsified and saved."""
        st_file = tmp_path / "adapter_model.safetensors"
        st_file.write_bytes(b"x" * 200)

        arr = np.ones((8, 8), dtype=np.float32)
        mock_stn = types.ModuleType("safetensors.numpy")
        mock_stn.load_file = MagicMock(return_value={"w": arr})
        mock_stn.save_file = MagicMock()

        with patch.dict(sys.modules, {"safetensors": MagicMock(), "safetensors.numpy": mock_stn}):
            cli = _import_cli()
            cli._apply_dare_sparsification(tmp_path, sparsity_ratio=0.9)

        mock_stn.save_file.assert_called_once()
        out = capsys.readouterr().out
        assert "DARE" in out


# ── _find_adapter_safetensors ─────────────────────────────────────────────────


class TestFindAdapterSafetensors:
    def test_path_is_safetensors_file(self, tmp_path: Path):
        """If the path itself is a .safetensors file, return it."""
        cli = _import_cli()
        st = tmp_path / "adapter.safetensors"
        st.write_bytes(b"x")
        result = cli._find_adapter_safetensors(st)
        assert result == st

    def test_path_is_dir_returns_first(self, tmp_path: Path):
        """If the path is a directory, return the first (sorted) .safetensors file."""
        cli = _import_cli()
        (tmp_path / "b.safetensors").write_bytes(b"b")
        (tmp_path / "a.safetensors").write_bytes(b"a")
        result = cli._find_adapter_safetensors(tmp_path)
        assert result.name == "a.safetensors"

    def test_no_files_raises(self, tmp_path: Path):
        """Directory with no .safetensors files raises FileNotFoundError."""
        cli = _import_cli()
        with pytest.raises(FileNotFoundError, match="No .safetensors files found"):
            cli._find_adapter_safetensors(tmp_path)


# ── cmd_train_adapter ─────────────────────────────────────────────────────────


def _train_ns(tmp_path: Path, dataset_exists: bool = True):
    if dataset_exists:
        ds = tmp_path / "train.jsonl"
        ds.write_text("data")
    else:
        ds = tmp_path / "missing.jsonl"
    return argparse.Namespace(
        model="qwen3:8b",
        dataset=str(ds),
        domain="legal",
        rank=8,
        epochs=3,
        output_dir=str(tmp_path / "out"),
    )


class TestCmdTrainAdapter:
    def test_dataset_not_found_exits(self, tmp_path: Path):
        cli = _import_cli()
        ns = _train_ns(tmp_path, dataset_exists=False)
        with pytest.raises(SystemExit) as exc:
            cli.cmd_train_adapter(ns)
        assert exc.value.code == 1

    def test_mlx_lm_unavailable_exits(self, tmp_path: Path):
        cli = _import_cli()
        ns = _train_ns(tmp_path)
        with patch.dict(sys.modules, {"mlx_lm": None}):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_train_adapter(ns)
        assert exc.value.code == 1

    def test_train_raises_exits(self, tmp_path: Path):
        cli = _import_cli()
        ns = _train_ns(tmp_path)
        mock_mlx = MagicMock()
        mock_mlx.lora.train.side_effect = RuntimeError("GPU OOM")
        with patch.dict(sys.modules, {"mlx_lm": mock_mlx}):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_train_adapter(ns)
        assert exc.value.code == 1

    def test_success_calls_train_and_dare(self, tmp_path: Path, capsys):
        cli = _import_cli()
        ns = _train_ns(tmp_path)
        mock_mlx = MagicMock()
        mock_mlx.lora.train.return_value = None
        # Let _apply_dare_sparsification run naturally (output_dir is empty → no-op)
        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "safetensors": None, "safetensors.numpy": None}):
            cli.cmd_train_adapter(ns)
        mock_mlx.lora.train.assert_called_once()
        out = capsys.readouterr().out
        assert "Training LoRA" in out
        assert "Adapter saved to" in out

    def test_output_dir_created(self, tmp_path: Path):
        cli = _import_cli()
        output_dir = tmp_path / "nested" / "out"
        assert not output_dir.exists()
        ds = tmp_path / "train.jsonl"
        ds.write_text("data")
        ns = argparse.Namespace(
            model="qwen3:8b",
            dataset=str(ds),
            domain="code",
            rank=4,
            epochs=1,
            output_dir=str(output_dir),
        )
        mock_mlx = MagicMock()
        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "safetensors": None, "safetensors.numpy": None}):
            cli.cmd_train_adapter(ns)
        assert output_dir.exists()


# ── cmd_merge_model ───────────────────────────────────────────────────────────


def _merge_ns(adapters, method="dare-ties", output_path=None, base_model="qwen3:8b"):
    return argparse.Namespace(
        base_model=base_model,
        adapters=adapters,
        method=method,
        output_path=str(output_path),
    )


def _make_mock_st_numpy(weights_per_adapter=None):
    """Return a mock safetensors.numpy module where load_file returns given dicts."""
    mock_stn = types.ModuleType("safetensors.numpy")
    if weights_per_adapter is None:
        weights_per_adapter = [{"w": np.ones((4, 4), dtype=np.float32)}]
    call_count = [0]

    def fake_load_file(path):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(weights_per_adapter):
            return weights_per_adapter[idx]
        return weights_per_adapter[-1]

    mock_stn.load_file = MagicMock(side_effect=fake_load_file)
    mock_stn.save_file = MagicMock()
    return mock_stn


class TestCmdMergeModel:
    def test_bad_adapter_spec_exits(self, tmp_path: Path):
        """Adapter spec without colon → SystemExit(1)."""
        cli = _import_cli()
        ns = _merge_ns(["no-colon-adapter"], output_path=tmp_path / "out")
        with pytest.raises(SystemExit) as exc:
            cli.cmd_merge_model(ns)
        assert exc.value.code == 1

    def test_adapter_path_not_found_exits(self, tmp_path: Path):
        """Adapter path does not exist → SystemExit(1)."""
        cli = _import_cli()
        ns = _merge_ns(
            [f"legal:{tmp_path / 'nonexistent'}"],
            output_path=tmp_path / "out",
        )
        with pytest.raises(SystemExit) as exc:
            cli.cmd_merge_model(ns)
        assert exc.value.code == 1

    def test_safetensors_unavailable_exits(self, tmp_path: Path):
        """safetensors not installed → SystemExit(1)."""
        cli = _import_cli()
        adapter_dir = tmp_path / "legal"
        adapter_dir.mkdir()
        ns = _merge_ns([f"legal:{adapter_dir}"], output_path=tmp_path / "out")
        with patch.dict(sys.modules, {"safetensors": None, "safetensors.numpy": None}):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_merge_model(ns)
        assert exc.value.code == 1

    def test_load_adapter_fails_exits(self, tmp_path: Path):
        """load_file raises → _die → SystemExit(1)."""
        cli = _import_cli()
        adapter_dir = tmp_path / "legal"
        adapter_dir.mkdir()
        ns = _merge_ns([f"legal:{adapter_dir}"], output_path=tmp_path / "out")

        mock_stn = types.ModuleType("safetensors.numpy")
        mock_stn.load_file = MagicMock(side_effect=RuntimeError("corrupt"))
        mock_stn.save_file = MagicMock()

        with patch.dict(sys.modules, {"safetensors": MagicMock(), "safetensors.numpy": mock_stn}):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_merge_model(ns)
        assert exc.value.code == 1

    def _run_merge(self, tmp_path, method, num_adapters=1, capsys=None):
        """Helper: create adapters, run merge, return (mock_stn, capsys_out)."""
        cli = _import_cli()
        adapter_dirs = []
        for i in range(num_adapters):
            ad = tmp_path / f"ad{i}"
            ad.mkdir()
            st = ad / "adapter_model.safetensors"
            st.write_bytes(b"x")
            adapter_dirs.append(ad)

        adapters = [f"d{i}:{adapter_dirs[i]}" for i in range(num_adapters)]
        ns = _merge_ns(adapters, method=method, output_path=tmp_path / "out")

        arr = np.array([[1.0, -2.0], [3.0, -1.0]], dtype=np.float32)
        weights_list = [{"layer.w": arr} for _ in range(num_adapters)]
        mock_stn = _make_mock_st_numpy(weights_list)

        with patch.dict(sys.modules, {"safetensors": MagicMock(), "safetensors.numpy": mock_stn}):
            cli.cmd_merge_model(ns)

        return mock_stn

    def test_dare_method_single_adapter(self, tmp_path: Path, capsys):
        mock_stn = self._run_merge(tmp_path, method="dare", num_adapters=1)
        mock_stn.save_file.assert_called_once()

    def test_ties_method_two_adapters(self, tmp_path: Path):
        """ties with 2 adapters hits the TIES majority-sign branch."""
        mock_stn = self._run_merge(tmp_path, method="ties", num_adapters=2)
        mock_stn.save_file.assert_called_once()

    def test_dare_ties_method_two_adapters(self, tmp_path: Path):
        """dare-ties with 2 adapters hits both DARE and TIES branches."""
        mock_stn = self._run_merge(tmp_path, method="dare-ties", num_adapters=2)
        mock_stn.save_file.assert_called_once()

    def test_merge_produces_output_message(self, tmp_path: Path, capsys):
        cli = _import_cli()
        adapter_dir = tmp_path / "legal"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x")

        ns = _merge_ns(
            [f"legal:{adapter_dir}"],
            method="dare",
            output_path=tmp_path / "out",
        )
        arr = np.ones((2, 2), dtype=np.float32)
        mock_stn = _make_mock_st_numpy([{"w": arr}])
        with patch.dict(sys.modules, {"safetensors": MagicMock(), "safetensors.numpy": mock_stn}):
            cli.cmd_merge_model(ns)

        out = capsys.readouterr().out
        assert "Merged adapter saved to" in out or "adapter_model.safetensors" in out

    def test_ties_single_adapter_no_ties_branch(self, tmp_path: Path):
        """ties with 1 adapter: TIES branch skipped (len(deltas)==1)."""
        mock_stn = self._run_merge(tmp_path, method="ties", num_adapters=1)
        mock_stn.save_file.assert_called_once()

    def test_empty_weights_skips_conflict_report(self, tmp_path: Path, capsys):
        """Adapters with no keys: total_keys=0, conflict-rate line not printed."""
        cli = _import_cli()
        adapter_dir = tmp_path / "legal"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x")

        ns = _merge_ns(
            [f"legal:{adapter_dir}"],
            method="dare",
            output_path=tmp_path / "out",
        )
        mock_stn = types.ModuleType("safetensors.numpy")
        mock_stn.load_file = MagicMock(return_value={})  # empty weights
        mock_stn.save_file = MagicMock()
        with patch.dict(sys.modules, {"safetensors": MagicMock(), "safetensors.numpy": mock_stn}):
            cli.cmd_merge_model(ns)
        out = capsys.readouterr().out
        assert "Sign conflict" not in out


# ── Subparser wiring ──────────────────────────────────────────────────────────


class TestTrainAdapterSubparser:
    def test_help_exits_zero(self):
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "train-adapter", "--help"]):
                cli.main()
        assert exc.value.code == 0

    def test_missing_required_args_exits(self):
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "train-adapter"]):
                cli.main()
        assert exc.value.code != 0


class TestMergeModelSubparser:
    def test_help_exits_zero(self):
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "merge-model", "--help"]):
                cli.main()
        assert exc.value.code == 0

    def test_missing_required_args_exits(self):
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "merge-model"]):
                cli.main()
        assert exc.value.code != 0
