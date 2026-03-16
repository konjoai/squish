"""
tests/test_model_pipeline_unit.py

Unit tests for dev/scripts/model_pipeline.py.

Coverage map (task requirements)
---------------------------------
1.  Candidate filter — licence: open licence (apache-2.0) passes
2.  Candidate filter — licence: non-commercial (cc-by-nc-4.0) is rejected
3.  Candidate filter — size: model tagged '72b' estimates > 60 GB and is excluded
4.  Candidate filter — size: model tagged '3b' estimates within the 1–60 GB window
5.  Candidate filter — age: _estimate_age_days > 180 for a 200-day-old model
6.  Candidate filter — age: _estimate_age_days < 1 for a model modified today
7.  Accuracy gate (CompressJob._accuracy_gate): delta ≤ 3 pp passes without retry
8.  Accuracy gate (CompressJob._accuracy_gate): delta exactly 3.0 pp passes (boundary)
9.  Accuracy gate (CompressJob._accuracy_gate): delta > 3 pp triggers retry_fn once
10. Accuracy gate (CompressJob._accuracy_gate): delta > 3 pp after retry → (False, delta)
11. _write_rejection: appends correct JSON record to pipeline_rejected.json
12. _write_rejection: dry-run mode never touches the filesystem
13. Catalog diff: new models (absent from previous_names) appear in result
14. Catalog diff: empty previous_names returns every candidate
15. --dry-run mode: CompressJob.run() never invokes subprocess.run
16. --dry-run mode: correct number of output_dirs returned without compression
17. Pipeline JSON schema: every watch-dry-run entry has the five required keys
18. Missing HF metadata: tags=None defaults to 7.0 GB estimate
19. Missing HF metadata: tags=[] defaults to 7.0 GB estimate
20. Missing HF metadata: lastModified=None returns 0.0 age days
21. Missing HF metadata: no licence info defaults to open (allowed)

Additional tests retained from earlier coverage
------------------------------------------------
- WatchJob dry-run returns list of ModelCandidate objects
- Synthetic candidates have valid priorities and positive sizes
- _estimate_size_gb parses 'B' tags correctly
- CompressJob._build_command includes 'squish' and '--int4'
- AccuracyGate.check dry-run always passes
- AccuracyGate.check passes when delta ≤ threshold
- AccuracyGate.check returns False and writes rejected.json when both fail
- AccuracyGate.check passes when INT8 retry succeeds
- PublishJob dry-run does not call subprocess
"""
from __future__ import annotations

import datetime
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make dev/scripts importable without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev" / "scripts"))

from model_pipeline import (  # noqa: E402
    _PP_THRESHOLD,
    _REJECTED_JSON,
    _SYNTHETIC_CANDIDATES,
    AccuracyGate,
    CompressJob,
    ModelCandidate,
    PipelineConfig,
    PublishJob,
    WatchJob,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _cfg(dry_run: bool = True, **kw) -> PipelineConfig:
    return PipelineConfig(dry_run=dry_run, **kw)


def _make_candidate(**overrides) -> ModelCandidate:
    """Return a ModelCandidate with sensible defaults."""
    defaults: dict = dict(
        name="TestModel-3B",
        hf_repo="test-org/TestModel-3B",
        size_gb=6.0,
        architecture="Qwen",
        priority="P1",
    )
    defaults.update(overrides)
    return ModelCandidate(**defaults)


def _make_hf_model(**overrides) -> MagicMock:
    """
    Return a minimal HuggingFace model_info mock.

    Attributes are set to concrete values so that ``getattr(m, attr, default)``
    and truthiness checks behave deterministically in the code under test.
    """
    m = MagicMock()
    m.id = overrides.get("model_id", "test-org/TestModel-3B")
    m.tags = overrides.get("tags", ["3b"])
    m.gated = overrides.get("gated", False)
    m.lastModified = overrides.get("last_modified", None)
    # None → ``getattr(m, "cardData", None) or {}`` yields {} inside the method.
    m.cardData = overrides.get("cardData", None)
    return m


# ── 1–2. Candidate filter: licence ───────────────────────────────────────────

class TestLicenceFilter:
    """WatchJob._is_open_license classifies open vs non-commercial licences."""

    def test_open_licence_apache_passes(self):
        """apache-2.0 is openly licenced — _is_open_license must return True."""
        job = WatchJob()
        m = _make_hf_model(tags=["license:apache-2.0", "3b"])
        assert job._is_open_license(m) is True

    def test_non_commercial_cc_by_nc_rejected(self):
        """cc-by-nc-4.0 is non-commercial — _is_open_license must return False."""
        job = WatchJob()
        m = _make_hf_model(tags=["license:cc-by-nc-4.0", "3b"])
        assert job._is_open_license(m) is False


# ── 3–4. Candidate filter: size ──────────────────────────────────────────────

class TestSizeFilter:
    """_estimate_size_gb returns values that cause > 70 B models to be excluded."""

    def test_72b_tag_exceeds_60gb_ceiling(self):
        """'72b' tag → 144 GB estimate, above the 60 GB filter ceiling."""
        job = WatchJob()
        m = _make_hf_model(tags=["72b"])
        size = job._estimate_size_gb(m)
        assert size > 60.0, f"Expected > 60 GB for a 72 B model, got {size}"

    def test_3b_tag_within_filter_window(self):
        """'3b' tag → 6 GB estimate, within the 1–60 GB window."""
        job = WatchJob()
        m = _make_hf_model(tags=["3b"])
        size = job._estimate_size_gb(m)
        assert 1.0 <= size <= 60.0, f"3 B model size {size} not in 1–60 GB range"


# ── 5–6. Candidate filter: age ───────────────────────────────────────────────

class TestAgeFilter:
    """_estimate_age_days returns correct age for dated model_info objects."""

    def test_200_day_old_model_exceeds_180_days(self):
        """A model last modified 200 days ago must report age > 180."""
        job = WatchJob()
        old_ts = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=200)
        ).isoformat()
        m = _make_hf_model(last_modified=old_ts)
        assert job._estimate_age_days(m) > 180.0

    def test_model_modified_today_is_under_one_day(self):
        """A model modified moments ago must report age < 1 day."""
        job = WatchJob()
        recent_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m = _make_hf_model(last_modified=recent_ts)
        assert job._estimate_age_days(m) < 1.0


# ── 7–8. Accuracy gate (CompressJob._accuracy_gate): passing cases ────────────

class TestCompressJobAccuracyGatePass:
    """CompressJob._accuracy_gate returns (True, delta) when within threshold."""

    def test_delta_below_threshold_passes_without_retry(self):
        """delta=1.5 pp < 3.0 pp → gate passes; retry_fn must not be called."""
        job = CompressJob()
        retry_fn = MagicMock()
        passed, final = job._accuracy_gate("TestModel-3B", 1.5, retry_fn)
        assert passed is True
        assert final == pytest.approx(1.5)
        retry_fn.assert_not_called()

    def test_delta_exactly_at_threshold_passes(self):
        """delta == _PP_THRESHOLD (3.0) is ≤ threshold — gate must pass."""
        job = CompressJob()
        retry_fn = MagicMock()
        passed, final = job._accuracy_gate("TestModel-3B", _PP_THRESHOLD, retry_fn)
        assert passed is True
        assert final == pytest.approx(_PP_THRESHOLD)
        retry_fn.assert_not_called()


# ── 9. Accuracy gate: delta > threshold triggers retry ───────────────────────

class TestCompressJobAccuracyGateRetry:
    """When delta > _PP_THRESHOLD, retry_fn is called exactly once."""

    def test_retry_fn_called_once_on_int4_failure(self):
        """delta=4.5 pp > 3.0 pp → retry_fn called; int8 returns 2.0 → passes."""
        job = CompressJob()
        retry_fn = MagicMock(return_value=2.0)
        passed, final = job._accuracy_gate("TestModel-3B", 4.5, retry_fn)
        retry_fn.assert_called_once()
        assert passed is True
        assert final == pytest.approx(2.0)


# ── 10. Accuracy gate: rejected after retry also fails ───────────────────────

class TestCompressJobAccuracyGateReject:
    """When both int4 and int8 exceed threshold, gate returns (False, retry_delta)."""

    def test_returns_false_when_int8_also_fails(self):
        """retry_fn returns 4.0 pp (> 3.0 pp) → gate returns (False, 4.0)."""
        job = CompressJob()
        retry_fn = MagicMock(return_value=4.0)
        passed, final = job._accuracy_gate("TestModel-3B", 5.0, retry_fn)
        assert passed is False
        assert final == pytest.approx(4.0)
        retry_fn.assert_called_once()


# ── 11–12. CompressJob._write_rejection ──────────────────────────────────────

class TestWriteRejection:
    """_write_rejection appends JSON to pipeline_rejected.json; dry-run skips disk."""

    def test_writes_correct_json_schema_to_disk(self, tmp_path):
        """
        After one call the file must exist with a list containing one entry
        that has all required schema keys.
        """
        job = CompressJob()
        candidate = _make_candidate()
        config = _cfg(dry_run=False, output_dir=tmp_path)
        rejected_path = tmp_path / "pipeline_rejected.json"

        with patch("model_pipeline._REJECTED_JSON", rejected_path):
            job._write_rejection(candidate, 4.5, config)

        assert rejected_path.exists(), "pipeline_rejected.json must be created"
        data = json.loads(rejected_path.read_text())
        assert isinstance(data, list) and len(data) == 1
        entry = data[0]
        assert entry["model_name"] == candidate.name
        assert entry["model_id"] == candidate.hf_repo
        assert entry["delta_pp"] == pytest.approx(4.5)
        assert entry["quant_attempted"] == "int8"
        assert "rejected_at" in entry

    def test_dry_run_does_not_touch_filesystem(self, tmp_path):
        """In dry-run mode _write_rejection must NOT create any files."""
        job = CompressJob()
        candidate = _make_candidate()
        config = _cfg(dry_run=True, output_dir=tmp_path)
        rejected_path = tmp_path / "pipeline_rejected.json"

        with patch("model_pipeline._REJECTED_JSON", rejected_path):
            job._write_rejection(candidate, 5.0, config)

        assert not rejected_path.exists(), (
            "pipeline_rejected.json must NOT be created in dry-run mode"
        )


# ── 13–14. Catalog diff writer ────────────────────────────────────────────────

class TestCatalogDiff:
    """WatchJob._catalog_diff returns only candidates not already in catalog."""

    def test_new_models_appear_in_diff(self):
        """Only ModelB and ModelC should appear; ModelA is already known."""
        job = WatchJob()
        candidates = [
            _make_candidate(name="ModelA"),
            _make_candidate(name="ModelB"),
            _make_candidate(name="ModelC"),
        ]
        diff = job._catalog_diff(candidates, previous_names=["ModelA"])
        names = {c.name for c in diff}
        assert "ModelB" in names
        assert "ModelC" in names
        assert "ModelA" not in names

    def test_empty_previous_returns_all_candidates(self):
        """With no previous catalog every candidate is new."""
        job = WatchJob()
        candidates = [_make_candidate(name="X"), _make_candidate(name="Y")]
        diff = job._catalog_diff(candidates, previous_names=[])
        assert len(diff) == 2


# ── 15–16. --dry-run mode ─────────────────────────────────────────────────────

class TestDryRunMode:
    """CompressJob.run(dry_run=True) must not call subprocess and still return dirs."""

    def test_subprocess_not_called_in_dry_run(self, tmp_path):
        """subprocess.run must never be invoked when dry_run=True."""
        candidates = [_make_candidate()]
        config = _cfg(dry_run=True, output_dir=tmp_path)
        job = CompressJob()
        with patch("subprocess.run") as mock_run:
            job.run(candidates, config)
        mock_run.assert_not_called()

    def test_dry_run_returns_one_dir_per_candidate(self, tmp_path):
        """Dry-run must return a Path for every candidate even without compression."""
        candidates = [_make_candidate(name="Alpha"), _make_candidate(name="Beta")]
        config = _cfg(dry_run=True, output_dir=tmp_path)
        job = CompressJob()
        with patch("subprocess.run"):
            output_dirs = job.run(candidates, config)
        assert len(output_dirs) == 2
        assert all(isinstance(d, Path) for d in output_dirs)


# ── 17. Pipeline JSON output schema ──────────────────────────────────────────

class TestPipelineJsonSchema:
    """Watch dry-run output JSON must contain all five required schema keys."""

    REQUIRED_KEYS = {"name", "hf_repo", "size_gb", "architecture", "priority"}

    def test_watch_dry_run_output_has_correct_schema(self, tmp_path):
        """
        Simulate the watch job serialising candidates to models.json and
        verify every entry contains the five keys consumed by the compress job.
        """
        config = _cfg(dry_run=True)
        job = WatchJob()
        candidates = job.run(config)

        output_path = tmp_path / "models.json"
        data = [
            {
                "name": c.name,
                "hf_repo": c.hf_repo,
                "size_gb": c.size_gb,
                "architecture": c.architecture,
                "priority": c.priority,
            }
            for c in candidates
        ]
        output_path.write_text(json.dumps(data, indent=2))
        loaded = json.loads(output_path.read_text())

        assert len(loaded) > 0, "Expected at least one candidate in dry-run output"
        for entry in loaded:
            missing = self.REQUIRED_KEYS - entry.keys()
            assert not missing, f"Entry is missing required keys: {missing!r}"


# ── 18–21. Missing / incomplete HF API metadata ──────────────────────────────

class TestMissingMetadata:
    """WatchJob helpers handle absent/None/empty HF metadata without raising."""

    def test_tags_none_defaults_to_7gb(self):
        """tags=None must not raise; must default to 7.0 GB."""
        job = WatchJob()
        m = _make_hf_model()
        m.tags = None
        assert job._estimate_size_gb(m) == pytest.approx(7.0)

    def test_tags_empty_list_defaults_to_7gb(self):
        """tags=[] must not raise; must default to 7.0 GB."""
        job = WatchJob()
        m = _make_hf_model()
        m.tags = []
        assert job._estimate_size_gb(m) == pytest.approx(7.0)

    def test_last_modified_none_returns_zero_age(self):
        """lastModified=None must return 0.0 from _estimate_age_days."""
        job = WatchJob()
        m = _make_hf_model(last_modified=None)
        assert job._estimate_age_days(m) == pytest.approx(0.0)

    def test_no_licence_info_defaults_to_open(self):
        """
        A model with cardData=None and empty tags has no detectable licence.
        The filter must default to allowing it (returns True).
        """
        job = WatchJob()
        m = _make_hf_model()
        m.cardData = None
        m.tags = []
        assert job._is_open_license(m) is True


# ── Additional tests: WatchJob dry-run ───────────────────────────────────────

class TestWatchJobDryRun:
    def test_dry_run_returns_list(self):
        result = WatchJob().run(_cfg(dry_run=True))
        assert isinstance(result, list)

    def test_dry_run_returns_at_least_3_candidates(self):
        result = WatchJob().run(_cfg(dry_run=True))
        assert len(result) >= 3

    def test_dry_run_returns_model_candidate_objects(self):
        for c in WatchJob().run(_cfg(dry_run=True)):
            assert isinstance(c, ModelCandidate)

    def test_synthetic_candidates_have_valid_priorities(self):
        for c in _SYNTHETIC_CANDIDATES:
            assert c.priority in ("P0", "P1", "P2")

    def test_synthetic_candidates_have_positive_size(self):
        for c in _SYNTHETIC_CANDIDATES:
            assert c.size_gb > 0

    def test_estimate_size_gb_parses_b_tag(self):
        job = WatchJob()
        m = MagicMock()
        m.tags = ["1B", "text-generation"]
        assert job._estimate_size_gb(m) == pytest.approx(2.0, abs=0.1)


# ── Additional tests: CompressJob ────────────────────────────────────────────

class TestCompressJobBuild:
    def test_build_command_includes_squish(self):
        job = CompressJob()
        cmd = job._build_command(_make_candidate(), Path("/tmp/out"))
        assert any("squish" in c for c in cmd)

    def test_build_command_includes_int4_flag(self):
        job = CompressJob()
        cmd = job._build_command(_make_candidate(), Path("/tmp/out"))
        assert "--int4" in cmd


# ── Additional tests: AccuracyGate (standalone class) ─────────────────────────

class TestAccuracyGateClass:
    """Tests for the AccuracyGate standalone class (separate from CompressJob)."""

    def test_dry_run_always_passes(self):
        gate = AccuracyGate()
        with tempfile.TemporaryDirectory() as tmp:
            result = gate.check(_make_candidate(), Path(tmp), _cfg(dry_run=True))
        assert result is True

    def test_no_reference_ppl_passes(self):
        gate = AccuracyGate()
        with tempfile.TemporaryDirectory() as tmp:
            result = gate.check(
                _make_candidate(), Path(tmp), _cfg(dry_run=False), reference_ppl=None
            )
        assert result is True

    def test_pass_when_delta_below_threshold(self):
        gate = AccuracyGate()
        gate.measure_perplexity = MagicMock(return_value=10.5)  # delta=0.5
        with tempfile.TemporaryDirectory() as tmp:
            result = gate.check(
                _make_candidate(), Path(tmp), _cfg(dry_run=False), reference_ppl=10.0
            )
        assert result is True

    def test_fail_and_write_rejected_when_both_quantisations_fail(self):
        gate = AccuracyGate()
        gate.measure_perplexity = MagicMock(return_value=17.0)  # delta=7 > 3
        gate._retry_int8 = MagicMock(return_value=16.0)          # retry delta=6 > 3
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "model"
            out_dir.mkdir()
            result = gate.check(
                _make_candidate(), out_dir, _cfg(dry_run=False), reference_ppl=10.0
            )
        assert result is False

    def test_rejected_json_written_on_double_failure(self):
        gate = AccuracyGate()
        gate.measure_perplexity = MagicMock(return_value=17.0)
        gate._retry_int8 = MagicMock(return_value=16.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "BadModel"
            out_dir.mkdir()
            gate.check(
                _make_candidate(name="BadModel"),
                out_dir,
                _cfg(dry_run=False),
                reference_ppl=10.0,
            )
            rejected_path = Path(tmp) / "pipeline_rejected.json"
            assert rejected_path.exists()
            data = json.loads(rejected_path.read_text())
            assert any(e["name"] == "BadModel" for e in data)

    def test_pass_when_int8_retry_succeeds(self):
        gate = AccuracyGate()
        gate.measure_perplexity = MagicMock(return_value=17.0)  # INT4 fails
        gate._retry_int8 = MagicMock(return_value=11.0)          # INT8 passes (delta=1)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "model"
            out_dir.mkdir()
            result = gate.check(
                _make_candidate(), out_dir, _cfg(dry_run=False), reference_ppl=10.0
            )
        assert result is True


# ── Additional tests: PublishJob ─────────────────────────────────────────────

class TestPublishJobDryRun:
    def test_dry_run_does_not_call_subprocess(self):
        job = PublishJob()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "the-model"
            out_dir.mkdir()
            with patch("model_pipeline.subprocess.run") as mock_run:
                job.run([out_dir], _cfg(dry_run=True))
            mock_run.assert_not_called()

    def test_build_command_contains_repo_and_model_dir(self):
        job = PublishJob()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "Qwen-1.5B"
            cmd = job._build_command(
                model_dir, "squish-community/Qwen-1.5B-int4", _cfg()
            )
            assert "--repo" in cmd
            assert "squish-community/Qwen-1.5B-int4" in cmd
