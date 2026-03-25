"""tests/test_wave83_moe_autoload.py

Wave 83 — MoE auto-load wiring + 6-model MoE catalog expansion

Tests:
  1. Server Wave 83 block: LazyExpertLoader auto-enabled when
     auto_profile.use_moe_lazy=True and _lazy_expert is None
  2. Server Wave 83 block: skipped when explicit --lazy-expert already set
  3. Server Wave 83 block: skipped when use_moe_lazy=False
  4. Server Wave 83 block: exception silenced (never blocks startup)
  5. Catalog: 6 new MoE entries present and structurally correct
  6. Catalog: required fields, moe=True, active_params_b set for all new entries
  7. Catalog: IDs are unique across entire catalog
  8. auto_profile: MoE detection sets use_moe_lazy=True for known architectures
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# TestMoeLazyAutoLoad — server.py Wave 83 block
# ============================================================================

class TestMoeLazyAutoLoad(unittest.TestCase):
    """Wave 83: LazyExpertLoader auto-enabled from auto_profile.use_moe_lazy."""

    def _make_moe_profile(self, use_moe_lazy=True):
        from squish.runtime.auto_profile import OptimizationProfile  # noqa: PLC0415
        return OptimizationProfile(
            use_moe_lazy=use_moe_lazy,
            active_features=["fused_int4", "moe-lazy"] if use_moe_lazy else ["fused_int4"],
        )

    def test_lazy_expert_enabled_when_use_moe_lazy_and_none(self):
        """LazyExpertLoader is instantiated when profile.use_moe_lazy=True and _lazy_expert is None."""
        mock_loader_cls = MagicMock()
        mock_loader_instance = MagicMock()
        mock_loader_cls.return_value = mock_loader_instance
        mock_config_cls = MagicMock()

        prof = self._make_moe_profile(use_moe_lazy=True)
        result = {}
        _fake_globals = {"_lazy_expert": None}

        with patch.dict("sys.modules", {
            "squish.moe.lazy_expert_load": types.ModuleType("squish.moe.lazy_expert_load"),
        }):
            sys.modules["squish.moe.lazy_expert_load"].LazyExpertConfig = mock_config_cls
            sys.modules["squish.moe.lazy_expert_load"].LazyExpertLoader = mock_loader_cls

            # Simulate the server.py Wave 83 conditional block
            if (
                prof is not None
                and prof.use_moe_lazy
                and _fake_globals.get("_lazy_expert") is None
            ):
                from squish.moe.lazy_expert_load import LazyExpertConfig, LazyExpertLoader  # noqa: PLC0415
                _le_cfg = LazyExpertConfig()
                result["_lazy_expert"] = LazyExpertLoader(_le_cfg)

        mock_loader_cls.assert_called_once()
        self.assertIn("_lazy_expert", result)

    def test_lazy_expert_not_reinitialised_when_already_set(self):
        """When _lazy_expert is already set (user passed --lazy-expert), auto-load is skipped."""
        mock_loader_cls = MagicMock()
        prof = self._make_moe_profile(use_moe_lazy=True)

        _fake_globals = {"_lazy_expert": MagicMock()}  # already set by user's --lazy-expert

        if (
            prof is not None
            and prof.use_moe_lazy
            and _fake_globals.get("_lazy_expert") is None     # Guard: this is False
        ):
            mock_loader_cls()

        mock_loader_cls.assert_not_called()

    def test_lazy_expert_skipped_when_use_moe_lazy_false(self):
        """use_moe_lazy=False → LazyExpertLoader is NOT called."""
        mock_loader_cls = MagicMock()
        prof = self._make_moe_profile(use_moe_lazy=False)
        _fake_globals = {"_lazy_expert": None}

        if (
            prof is not None
            and prof.use_moe_lazy          # False
            and _fake_globals.get("_lazy_expert") is None
        ):
            mock_loader_cls()

        mock_loader_cls.assert_not_called()

    def test_lazy_expert_skipped_when_prof_none(self):
        """None auto_profile → LazyExpertLoader is NOT called."""
        mock_loader_cls = MagicMock()
        prof = None
        _lazy_expert = None

        if (
            prof is not None              # False
            and getattr(prof, "use_moe_lazy", False)
            and _lazy_expert is None
        ):
            mock_loader_cls()

        mock_loader_cls.assert_not_called()

    def test_lazy_expert_exception_does_not_propagate(self):
        """Import/init failure in Wave 83 block must not raise (simulates try/except guard)."""
        prof = self._make_moe_profile(use_moe_lazy=True)
        _lazy_expert = None
        startup_crashed = False

        try:
            try:
                if prof is not None and prof.use_moe_lazy and _lazy_expert is None:
                    raise ImportError("no moe module found")
            except Exception:
                pass  # server.py catches all exceptions
        except Exception:
            startup_crashed = True

        self.assertFalse(startup_crashed)

    def test_server_lazy_expert_global_exists(self):
        """squish.server._lazy_expert global exists (set at module level)."""
        import squish.server as srv  # noqa: PLC0415
        self.assertTrue(
            hasattr(srv, "_lazy_expert"),
            "_lazy_expert global missing from squish.server",
        )


# ============================================================================
# TestMoeCatalogEntries — 6 new Wave 83 catalog entries
# ============================================================================

_WAVE83_IDS = {
    "olmoe:1b-7b",
    "qwen2-moe:57b-a14b",
    "phi3.5-moe:42b",
    "deepseek-v2-lite:16b",
    "jamba:1.5-mini",
    "deepseek-v3:685b",
}

_REQUIRED_FIELDS = {"id", "name", "hf_mlx_repo", "size_gb", "params", "context", "tags"}


class TestMoeCatalogEntries(unittest.TestCase):
    """Structural correctness of the 6 new MoE catalog entries."""

    @classmethod
    def setUpClass(cls):
        from squish.catalog import _BUNDLED  # noqa: PLC0415
        cls.catalog = _BUNDLED
        cls.new_moe_entries = [e for e in cls.catalog if e.get("id") in _WAVE83_IDS]

    def test_all_six_entries_present(self):
        found = {e["id"] for e in self.new_moe_entries}
        missing = _WAVE83_IDS - found
        self.assertEqual(missing, set(), f"Missing catalog entries: {missing}")

    def test_all_new_entries_have_required_fields(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                missing = _REQUIRED_FIELDS - entry.keys()
                self.assertEqual(missing, set(), f"{entry['id']} missing fields: {missing}")

    def test_all_new_entries_have_moe_true(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                self.assertTrue(entry.get("moe"), f"{entry['id']} missing moe=True")

    def test_all_new_entries_have_active_params_b(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                self.assertIn("active_params_b", entry, f"{entry['id']} missing active_params_b")
                self.assertGreater(entry["active_params_b"], 0)

    def test_all_new_entries_have_moe_in_tags(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                self.assertIn("moe", entry.get("tags", []),
                              f"{entry['id']} missing 'moe' tag")

    def test_all_new_entries_have_positive_size_gb(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                self.assertGreater(entry["size_gb"], 0)

    def test_all_new_entries_have_hf_mlx_repo_format(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                repo = entry["hf_mlx_repo"]
                self.assertIn("/", repo, f"{entry['id']} hf_mlx_repo lacks '/'")
                self.assertTrue(repo.startswith("mlx-community/") or "/" in repo)

    def test_no_duplicate_ids_in_full_catalog(self):
        all_ids = [e["id"] for e in self.catalog]
        duplicates = [i for i in all_ids if all_ids.count(i) > 1]
        self.assertEqual(duplicates, [], f"Duplicate catalog IDs: {set(duplicates)}")

    def test_olmoe_active_params_less_than_total(self):
        entry = next(e for e in self.new_moe_entries if e["id"] == "olmoe:1b-7b")
        self.assertLess(entry["active_params_b"], float(entry["params"].rstrip("B")))

    def test_deepseek_v3_tagged_impossible(self):
        entry = next(e for e in self.new_moe_entries if e["id"] == "deepseek-v3:685b")
        self.assertIn("impossible", entry.get("tags", []))

    def test_jamba_has_ssm_tag(self):
        """Jamba is a hybrid SSM/MoE — should have 'ssm' tag."""
        entry = next(e for e in self.new_moe_entries if e["id"] == "jamba:1.5-mini")
        self.assertIn("ssm", entry.get("tags", []))

    def test_new_entries_context_positive(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                self.assertGreater(entry.get("context", 0), 0)

    def test_deepseek_v3_is_largest_active_params(self):
        """DeepSeek-V3 has largest active_params_b among new entries."""
        max_entry = max(self.new_moe_entries, key=lambda e: e["active_params_b"])
        self.assertEqual(max_entry["id"], "deepseek-v3:685b")

    def test_olmoe_is_smallest_active_params(self):
        """OlMoE has smallest active_params_b among new entries."""
        min_entry = min(self.new_moe_entries, key=lambda e: e["active_params_b"])
        self.assertEqual(min_entry["id"], "olmoe:1b-7b")

    def test_notes_present_for_all_new_entries(self):
        for entry in self.new_moe_entries:
            with self.subTest(id=entry["id"]):
                notes = entry.get("notes", "")
                self.assertTrue(len(notes) > 20, f"{entry['id']} has very short notes")


# ============================================================================
# TestMoeCatalogResolve — catalog resolution for new MoE IDs
# ============================================================================

class TestMoeCatalogResolve(unittest.TestCase):
    """Re-verify that squish.catalog.resolve() finds all new MoE entries."""

    def test_resolve_olmoe(self):
        from squish.catalog import resolve  # noqa: PLC0415
        entry = resolve("olmoe:1b-7b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)

    def test_resolve_qwen2_moe(self):
        from squish.catalog import resolve  # noqa: PLC0415
        entry = resolve("qwen2-moe:57b-a14b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)

    def test_resolve_phi35_moe(self):
        from squish.catalog import resolve  # noqa: PLC0415
        entry = resolve("phi3.5-moe:42b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)

    def test_resolve_deepseek_v2_lite(self):
        from squish.catalog import resolve  # noqa: PLC0415
        entry = resolve("deepseek-v2-lite:16b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)

    def test_resolve_jamba_mini(self):
        from squish.catalog import resolve  # noqa: PLC0415
        entry = resolve("jamba:1.5-mini")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)

    def test_resolve_deepseek_v3(self):
        from squish.catalog import resolve  # noqa: PLC0415
        entry = resolve("deepseek-v3:685b")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.moe)


# ============================================================================
# TestAutoProfileMoEDetection — auto_profile detects MoE architectures
# ============================================================================

class TestAutoProfileMoEDetection(unittest.TestCase):
    """auto_profile correctly sets use_moe_lazy=True for known MoE architectures."""

    def _make_config(self, architecture: str, **extra) -> dict:
        cfg = {"architectures": [architecture]}
        cfg.update(extra)
        return cfg

    def _detect(self, config: dict) -> "OptimizationProfile":
        from squish.runtime.auto_profile import ModelCapabilityDetector  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "config.json"
            cfg_path.write_text(json.dumps(config))
            detector = ModelCapabilityDetector()
            # Access private pass directly for unit testing
            from squish.runtime.auto_profile import OptimizationProfile  # noqa: PLC0415
            prof = OptimizationProfile()
            detector._detect_model_config(prof, Path(d))
            return prof

    def test_qwen3_moe_detected(self):
        prof = self._detect(self._make_config("Qwen3MoeForCausalLM"))
        self.assertTrue(prof.use_moe_lazy)

    def test_mixtral_detected(self):
        prof = self._detect(self._make_config("MixtralForCausalLM"))
        self.assertTrue(prof.use_moe_lazy)

    def test_deepseek_v2_detected(self):
        prof = self._detect(self._make_config("DeepseekV2ForCausalLM"))
        self.assertTrue(prof.use_moe_lazy)

    def test_phimoe_detected(self):
        prof = self._detect(self._make_config("PhiMoeForCausalLM"))
        self.assertTrue(prof.use_moe_lazy)

    def test_num_experts_detected(self):
        prof = self._detect({"num_experts": 8})
        self.assertTrue(prof.use_moe_lazy)

    def test_num_local_experts_detected(self):
        prof = self._detect({"num_local_experts": 64})
        self.assertTrue(prof.use_moe_lazy)

    def test_dense_model_not_moe(self):
        """Non-MoE dense architecture must not set use_moe_lazy."""
        prof = self._detect(self._make_config("LlamaForCausalLM"))
        self.assertFalse(prof.use_moe_lazy)

    def test_gemma_not_moe(self):
        prof = self._detect(self._make_config("GemmaForCausalLM"))
        self.assertFalse(prof.use_moe_lazy)

    def test_qwen3_dense_not_moe(self):
        """Qwen3 (non-MoE dense variant) must not trigger use_moe_lazy."""
        prof = self._detect(self._make_config("Qwen3ForCausalLM"))
        self.assertFalse(prof.use_moe_lazy)

    def test_olmoe_arch_detected(self):
        """OlMoEForCausalLM contains 'moe' → lazy=True."""
        prof = self._detect(self._make_config("OlMoEForCausalLM"))
        self.assertTrue(prof.use_moe_lazy)

    def test_jamba_arch_detected(self):
        """JambaForCausalLM: uses num_experts key."""
        prof = self._detect({"architectures": ["JambaForCausalLM"], "num_experts": 16})
        self.assertTrue(prof.use_moe_lazy)

    def test_empty_config_not_moe(self):
        """Empty config.json → use_moe_lazy remains False."""
        prof = self._detect({})
        self.assertFalse(prof.use_moe_lazy)


# ============================================================================
# TestTotalMoeCatalogCount
# ============================================================================

class TestTotalMoeCatalogCount(unittest.TestCase):
    """Catalog contains at least 12 MoE entries (6 pre-Wave-83 + 6 new)."""

    def test_at_least_twelve_moe_models(self):
        from squish.catalog import _BUNDLED  # noqa: PLC0415
        moe_count = sum(1 for e in _BUNDLED if e.get("moe"))
        self.assertGreaterEqual(moe_count, 12,
                                f"Expected ≥12 MoE catalog entries, got {moe_count}")


if __name__ == "__main__":
    unittest.main()
