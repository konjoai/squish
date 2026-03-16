"""
tests/integration/test_model_pipeline_unit.py

Unit tests for dev/scripts/model_pipeline.py.

All tests in this file must pass without HF_TOKEN or real models.
Dry-run mode is the primary test vector.

Run with:
    /Users/wscholl/.pyenv/versions/3.12.7/envs/squish/bin/pytest \
        tests/integration/test_model_pipeline_unit.py -v --tb=short
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure dev/scripts is importable
_PIPELINE_PATH = Path(__file__).resolve().parent.parent.parent / "dev" / "scripts"
if str(_PIPELINE_PATH) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_PATH))

from model_pipeline import (  # noqa: E402
    _KNOWN_ARCHITECTURES,
    _SYNTHETIC_CANDIDATES,
    CompressJob,
    ModelCandidate,
    PipelineConfig,
    PublishJob,
    WatchJob,
    main,
)

# ── TestModelCandidate ────────────────────────────────────────────────────────

class TestModelCandidate:
    """Verify ModelCandidate dataclass stores fields correctly."""

    def test_fields_stored(self):
        c = ModelCandidate(
            name="Qwen2.5-1.5B-Instruct",
            hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
            size_gb=3.1,
            architecture="Qwen",
            priority="P0",
        )
        assert c.name == "Qwen2.5-1.5B-Instruct"
        assert c.hf_repo == "Qwen/Qwen2.5-1.5B-Instruct"
        assert c.size_gb == 3.1
        assert c.architecture == "Qwen"
        assert c.priority == "P0"

    def test_dataclass_equality(self):
        c1 = ModelCandidate("A", "org/A", 2.0, "Llama", "P1")
        c2 = ModelCandidate("A", "org/A", 2.0, "Llama", "P1")
        assert c1 == c2

    def test_dataclass_mutation(self):
        c = ModelCandidate("A", "org/A", 2.0, "Llama", "P1")
        c.priority = "P0"
        assert c.priority == "P0"


# ── TestPipelineConfig ────────────────────────────────────────────────────────

class TestPipelineConfig:
    """Verify PipelineConfig defaults are set correctly."""

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.dry_run is False
        assert cfg.validate is False
        assert cfg.output_dir == Path.home() / ".cache" / "squish" / "pipeline"
        assert cfg.hf_token is None

    def test_custom_values(self):
        cfg = PipelineConfig(
            dry_run=True,
            validate=True,
            output_dir=Path("/tmp/test"),
            hf_token="hf_test",
        )
        assert cfg.dry_run is True
        assert cfg.validate is True
        assert cfg.output_dir == Path("/tmp/test")
        assert cfg.hf_token == "hf_test"


# ── TestWatchJobDryRun ────────────────────────────────────────────────────────

class TestWatchJobDryRun:
    """--dry-run returns exactly 3 candidates without any external calls."""

    def test_returns_exactly_three(self):
        job = WatchJob()
        cfg = PipelineConfig(dry_run=True)
        candidates = job.run(cfg)
        assert len(candidates) == 3

    def test_no_external_calls(self):
        """No huggingface_hub calls should be made in dry-run mode."""
        job = WatchJob()
        cfg = PipelineConfig(dry_run=True)
        # Patch huggingface_hub to fail — dry-run must not reach it
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            candidates = job.run(cfg)
        assert len(candidates) == 3

    def test_synthetic_candidates_are_independent_copies(self):
        """Each call returns a fresh list."""
        job = WatchJob()
        cfg = PipelineConfig(dry_run=True)
        c1 = job.run(cfg)
        c2 = job.run(cfg)
        assert c1 == c2
        assert c1 is not c2


# ── TestWatchJobCandidateFields ───────────────────────────────────────────────

class TestWatchJobCandidateFields:
    """Each synthetic candidate has valid field values."""

    @pytest.fixture
    def candidates(self):
        job = WatchJob()
        cfg = PipelineConfig(dry_run=True)
        return job.run(cfg)

    def test_all_have_name(self, candidates):
        for c in candidates:
            assert c.name and isinstance(c.name, str)

    def test_all_have_hf_repo(self, candidates):
        for c in candidates:
            assert c.hf_repo and "/" in c.hf_repo

    def test_all_have_positive_size_gb(self, candidates):
        for c in candidates:
            assert c.size_gb > 0

    def test_all_architectures_known(self, candidates):
        for c in candidates:
            assert c.architecture in _KNOWN_ARCHITECTURES

    def test_all_priorities_valid(self, candidates):
        for c in candidates:
            assert c.priority in {"P0", "P1", "P2"}


# ── TestCompressJobDryRun ─────────────────────────────────────────────────────

class TestCompressJobDryRun:
    """Dry-run prints commands without executing subprocess."""

    def test_dry_run_no_subprocess(self, tmp_path, capsys):
        cfg = PipelineConfig(dry_run=True, output_dir=tmp_path)
        candidates = [ModelCandidate("TestModel", "org/TestModel", 3.0, "Qwen", "P0")]
        job = CompressJob()

        with patch("subprocess.run") as mock_run:
            job.run(candidates, cfg)
            mock_run.assert_not_called()

    def test_dry_run_prints_command(self, tmp_path, capsys):
        cfg = PipelineConfig(dry_run=True, output_dir=tmp_path)
        candidates = [ModelCandidate("TestModel", "org/TestModel", 3.0, "Qwen", "P0")]
        job = CompressJob()
        job.run(candidates, cfg)
        out = capsys.readouterr().out
        assert "DRY-RUN" in out

    def test_dry_run_returns_output_dirs(self, tmp_path):
        cfg = PipelineConfig(dry_run=True, output_dir=tmp_path)
        candidates = [ModelCandidate("TestModel", "org/TestModel", 3.0, "Qwen", "P0")]
        job = CompressJob()
        result = job.run(candidates, cfg)
        assert len(result) == 1
        assert result[0] == tmp_path / "TestModel"


# ── TestCompressJobCommandBuild ───────────────────────────────────────────────

class TestCompressJobCommandBuild:
    """Command contains squish compress, --int4, and model name."""

    def test_command_contains_squish_compress(self):
        job = CompressJob()
        c = ModelCandidate("M", "org/M", 3.0, "Llama", "P1")
        cmd = job._build_command(c, Path("/tmp/M"))
        assert "squish" in cmd
        assert "compress" in cmd

    def test_command_contains_int4(self):
        job = CompressJob()
        c = ModelCandidate("M", "org/M", 3.0, "Llama", "P1")
        cmd = job._build_command(c, Path("/tmp/M"))
        assert "--int4" in cmd

    def test_command_contains_model_name(self):
        job = CompressJob()
        c = ModelCandidate("MyModel", "org/MyModel", 3.0, "Mistral", "P0")
        cmd = job._build_command(c, Path("/tmp/MyModel"))
        assert "org/MyModel" in cmd

    def test_command_contains_output_dir(self):
        job = CompressJob()
        c = ModelCandidate("M", "org/M", 3.0, "Phi", "P2")
        out = Path("/tmp/output/M")
        cmd = job._build_command(c, out)
        assert "--output-dir" in cmd
        assert str(out) in cmd


# ── TestPublishJobDryRun ──────────────────────────────────────────────────────

class TestPublishJobDryRun:
    """Dry-run doesn't call publish_hf.py subprocess."""

    def test_no_subprocess_in_dry_run(self, tmp_path, capsys):
        cfg = PipelineConfig(dry_run=True)
        job = PublishJob()

        with patch("subprocess.run") as mock_run:
            job.run([tmp_path / "TestModel"], cfg)
            mock_run.assert_not_called()

    def test_dry_run_prints_publish_command(self, tmp_path, capsys):
        cfg = PipelineConfig(dry_run=True)
        job = PublishJob()
        job.run([tmp_path / "TestModel"], cfg)
        out = capsys.readouterr().out
        assert "DRY-RUN" in out

    def test_dry_run_uses_squish_community_org(self, tmp_path, capsys):
        cfg = PipelineConfig(dry_run=True)
        job = PublishJob()
        job.run([tmp_path / "TestModel"], cfg)
        out = capsys.readouterr().out
        assert "squish-community" in out


# ── TestPipelineE2E ───────────────────────────────────────────────────────────

class TestPipelineE2E:
    """watch --dry-run feeds into compress --dry-run without error."""

    def test_watch_into_compress_dry_run(self, tmp_path):
        cfg = PipelineConfig(dry_run=True, output_dir=tmp_path)

        watch = WatchJob()
        candidates = watch.run(cfg)
        assert len(candidates) == 3

        compress = CompressJob()
        with patch("subprocess.run") as mock_run:
            output_dirs = compress.run(candidates, cfg)
            mock_run.assert_not_called()

        assert len(output_dirs) == 3

    def test_full_pipeline_dry_run(self, tmp_path):
        """Full watch → compress → publish pipeline in dry-run."""
        cfg = PipelineConfig(dry_run=True, output_dir=tmp_path)

        candidates = WatchJob().run(cfg)
        output_dirs = CompressJob().run(candidates, cfg)

        with patch("subprocess.run") as mock_run:
            PublishJob().run(output_dirs, cfg)
            mock_run.assert_not_called()


# ── TestPipelineMainArgs ──────────────────────────────────────────────────────

class TestPipelineMainArgs:
    """CLI --job watch and --job compress --validate --dry-run don't raise."""

    def test_job_watch_dry_run(self, tmp_path):
        output_file = tmp_path / "models.json"
        main([
            "--job", "watch",
            "--dry-run",
            "--output", str(output_file),
        ])
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_job_compress_validate_dry_run(self, tmp_path):
        # Create models.json first
        models_file = tmp_path / "models.json"
        data = [
            {
                "name": "Qwen2.5-1.5B-Instruct",
                "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct",
                "size_gb": 3.1,
                "architecture": "Qwen",
                "priority": "P0",
            }
        ]
        models_file.write_text(json.dumps(data))

        # Should not raise even with --validate in dry-run mode
        main([
            "--job", "compress",
            "--validate",
            "--dry-run",
            "--models", str(models_file),
            "--output-dir", str(tmp_path / "output"),
        ])


# ── TestPipelineInvalidJob ────────────────────────────────────────────────────

class TestPipelineInvalidJob:
    """Unknown job name raises SystemExit or ValueError."""

    def test_unknown_job_raises(self):
        with pytest.raises(SystemExit):
            main(["--job", "nonexistent_job_xyz"])
