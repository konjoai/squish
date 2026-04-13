"""tests/test_squash_model_card.py — Wave 57: ModelCardGenerator test suite.

Test taxonomy:
  - Pure unit:        ModelCardConfig, ModelCardSection, ModelCard.render()
  - Integration:      ModelCardGenerator loading artifacts from temp dirs,
                      generate() writing files, CLI via subprocess
  - Failure cases:    unknown format, missing model_dir, bad artifact JSON

All tests that create files use temporary directories (cleaned up in tearDown).
No in-process sys.path / sys.modules mutations.  No mocking of the generator.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# ── Unit: ModelCardConfig ─────────────────────────────────────────────────────


class TestModelCardConfig(unittest.TestCase):
    def test_required_field_model_dir(self) -> None:
        from squish.squash.model_card import ModelCardConfig

        cfg = ModelCardConfig(model_dir=Path("/tmp/model"))
        self.assertEqual(cfg.model_dir, Path("/tmp/model"))

    def test_defaults(self) -> None:
        from squish.squash.model_card import ModelCardConfig

        cfg = ModelCardConfig(model_dir=Path("/tmp/model"))
        self.assertEqual(cfg.model_id, "")
        self.assertEqual(cfg.model_name, "")
        self.assertEqual(cfg.model_version, "")
        self.assertEqual(cfg.language, ["en"])
        self.assertEqual(cfg.license, "apache-2.0")
        self.assertEqual(cfg.tags, [])
        self.assertIsNone(cfg.output_dir)

    def test_custom_values(self) -> None:
        from squish.squash.model_card import ModelCardConfig

        cfg = ModelCardConfig(
            model_dir=Path("/tmp/m"),
            model_id="acme/qwen-1.5b-int4",
            model_name="Qwen 1.5B INT4",
            model_version="1.2.0",
            language=["en", "fr"],
            license="mit",
            tags=["quantized"],
        )
        self.assertEqual(cfg.model_id, "acme/qwen-1.5b-int4")
        self.assertEqual(cfg.model_version, "1.2.0")
        self.assertEqual(cfg.language, ["en", "fr"])
        self.assertIn("quantized", cfg.tags)


# ── Unit: ModelCardSection ────────────────────────────────────────────────────


class TestModelCardSection(unittest.TestCase):
    def test_default_level(self) -> None:
        from squish.squash.model_card import ModelCardSection

        sec = ModelCardSection(title="Intro", content="# hello")
        self.assertEqual(sec.level, 2)

    def test_custom_level(self) -> None:
        from squish.squash.model_card import ModelCardSection

        sec = ModelCardSection(title="Sub", content="body", level=3)
        self.assertEqual(sec.level, 3)

    def test_fields_preserved(self) -> None:
        from squish.squash.model_card import ModelCardSection

        sec = ModelCardSection(title="T", content="C", level=4)
        self.assertEqual(sec.title, "T")
        self.assertEqual(sec.content, "C")
        self.assertEqual(sec.level, 4)


# ── Unit: ModelCard.render() ──────────────────────────────────────────────────


class TestModelCardRender(unittest.TestCase):
    def _make_card(self, fmt: str = "hf") -> "ModelCard":  # type: ignore[name-defined]
        from squish.squash.model_card import ModelCard, ModelCardConfig, ModelCardSection

        config = ModelCardConfig(model_dir=Path("/tmp/m"))
        return ModelCard(
            config=config,
            fmt=fmt,
            yaml_frontmatter={"model_id": "test/model", "license": "apache-2.0"},
            sections=[
                ModelCardSection("Introduction", "Hello world.", level=1),
                ModelCardSection("Details", "Some details here.", level=2),
            ],
            generated_at="2025-01-01T00:00:00+00:00",
        )

    def test_render_starts_with_yaml_fence(self) -> None:
        card = self._make_card()
        rendered = card.render()
        self.assertTrue(rendered.startswith("---"))

    def test_render_has_closing_yaml_fence(self) -> None:
        card = self._make_card()
        rendered = card.render()
        lines = rendered.splitlines()
        # First line is "---", second closing fence appears somewhere before content
        fence_count = sum(1 for l in lines if l == "---")
        self.assertGreaterEqual(fence_count, 2)

    def test_render_contains_frontmatter_key(self) -> None:
        card = self._make_card()
        self.assertIn("model_id:", card.render())

    def test_render_contains_h1_section(self) -> None:
        card = self._make_card()
        self.assertIn("# Introduction", card.render())

    def test_render_contains_h2_section(self) -> None:
        card = self._make_card()
        self.assertIn("## Details", card.render())

    def test_render_contains_section_body(self) -> None:
        card = self._make_card()
        self.assertIn("Hello world.", card.render())

    def test_render_list_frontmatter(self) -> None:
        from squish.squash.model_card import ModelCard, ModelCardConfig

        config = ModelCardConfig(model_dir=Path("/tmp/m"))
        card = ModelCard(
            config=config,
            fmt="hf",
            yaml_frontmatter={"language": ["en", "fr"], "tags": ["quantized"]},
            sections=[],
        )
        rendered = card.render()
        self.assertIn("language:", rendered)
        self.assertIn("  - en", rendered)
        self.assertIn("  - fr", rendered)

    def test_render_bool_frontmatter(self) -> None:
        from squish.squash.model_card import ModelCard, ModelCardConfig

        config = ModelCardConfig(model_dir=Path("/tmp/m"))
        card = ModelCard(
            config=config,
            fmt="eu-ai-act",
            yaml_frontmatter={"squash_attested": True},
            sections=[],
        )
        self.assertIn("squash_attested: true", card.render())


# ── Unit: ModelCard.write() ───────────────────────────────────────────────────


class TestModelCardWrite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.tmp = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_write_hf_returns_correct_filename(self) -> None:
        from squish.squash.model_card import ModelCard, ModelCardConfig

        config = ModelCardConfig(model_dir=self.tmp)
        card = ModelCard(
            config=config,
            fmt="hf",
            yaml_frontmatter={"model_id": "test"},
            sections=[],
        )
        path = card.write()
        self.assertEqual(path.name, "squash-model-card-hf.md")

    def test_write_euaiact_filename(self) -> None:
        from squish.squash.model_card import ModelCard, ModelCardConfig

        config = ModelCardConfig(model_dir=self.tmp)
        card = ModelCard(
            config=config,
            fmt="eu-ai-act",
            yaml_frontmatter={"regulation": "EU AI Act"},
            sections=[],
        )
        path = card.write()
        self.assertEqual(path.name, "squash-model-card-euaiact.md")

    def test_write_iso42001_filename(self) -> None:
        from squish.squash.model_card import ModelCard, ModelCardConfig

        config = ModelCardConfig(model_dir=self.tmp)
        card = ModelCard(
            config=config,
            fmt="iso-42001",
            yaml_frontmatter={"standard": "ISO/IEC 42001:2023"},
            sections=[],
        )
        path = card.write()
        self.assertEqual(path.name, "squash-model-card-iso42001.md")

    def test_write_creates_file(self) -> None:
        from squish.squash.model_card import ModelCard, ModelCardConfig

        config = ModelCardConfig(model_dir=self.tmp)
        card = ModelCard(
            config=config,
            fmt="hf",
            yaml_frontmatter={"model_id": "test"},
            sections=[],
        )
        path = card.write()
        self.assertTrue(path.exists())

    def test_write_output_dir_override(self) -> None:
        from squish.squash.model_card import ModelCard, ModelCardConfig

        config = ModelCardConfig(model_dir=self.tmp)
        card = ModelCard(
            config=config,
            fmt="hf",
            yaml_frontmatter={},
            sections=[],
        )
        out_dir = self.tmp / "out"
        path = card.write(output_dir=out_dir)
        self.assertTrue((out_dir / "squash-model-card-hf.md").exists())
        self.assertEqual(path, out_dir / "squash-model-card-hf.md")


# ── Integration: ModelCardGenerator (no artifacts) ───────────────────────────


class TestModelCardGeneratorEmpty(unittest.TestCase):
    """Generator degrades gracefully when no squash artifacts are present."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.tmp = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_construct_empty_dir(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        self.assertEqual(gen.model_dir, self.tmp)

    def test_model_id_falls_back_to_dirname(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        self.assertEqual(gen._model_id(), self.tmp.name)

    def test_scan_summary_no_scan(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        self.assertIn("No security findings", gen._scan_summary())

    def test_policy_summary_no_policies(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        self.assertIn("No policy evaluations", gen._policy_summary())

    def test_vex_summary_no_vex(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        self.assertIn("No CVEs", gen._vex_summary())

    def test_generate_hf_default(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("hf")
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].exists())
        self.assertEqual(paths[0].name, "squash-model-card-hf.md")

    def test_generate_writes_yaml_frontmatter(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("hf")
        content = paths[0].read_text()
        self.assertTrue(content.startswith("---"))

    def test_generate_euaiact(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("eu-ai-act")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "squash-model-card-euaiact.md")
        content = paths[0].read_text()
        self.assertIn("EU AI Act", content)
        self.assertIn("Art. 13", content)

    def test_generate_iso42001(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("iso-42001")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "squash-model-card-iso42001.md")
        content = paths[0].read_text()
        self.assertIn("ISO/IEC 42001", content)

    def test_generate_all_writes_three_files(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("all")
        self.assertEqual(len(paths), 3)
        names = {p.name for p in paths}
        self.assertIn("squash-model-card-hf.md", names)
        self.assertIn("squash-model-card-euaiact.md", names)
        self.assertIn("squash-model-card-iso42001.md", names)

    def test_unknown_format_raises_value_error(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        with self.assertRaises(ValueError):
            gen.generate("banana")


# ── Integration: Generator with squish.json ───────────────────────────────────


class TestModelCardGeneratorWithSquishJson(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.tmp = Path(self._tmp)
        (self.tmp / "squish.json").write_text(
            json.dumps({"model_id": "acme/qwen2.5-1.5b-int4", "quant_format": "INT4"}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_model_id_from_squish_json(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        self.assertEqual(gen._model_id(), "acme/qwen2.5-1.5b-int4")

    def test_quant_format_from_squish_json(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        self.assertEqual(gen._quant_format(), "INT4")

    def test_hf_card_contains_model_id(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("hf")
        content = paths[0].read_text()
        self.assertIn("acme/qwen2.5-1.5b-int4", content)


# ── Integration: Generator with all artifacts ─────────────────────────────────


class TestModelCardGeneratorAllArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.tmp = Path(self._tmp)

        (self.tmp / "squish.json").write_text(
            json.dumps({"model_id": "acme/test-int4", "quant_format": "INT4"}),
            encoding="utf-8",
        )
        (self.tmp / "squash-scan.json").write_text(
            json.dumps(
                {
                    "findings": [
                        {"severity": "error", "message": "pickle found"},
                        {"severity": "warning", "message": "no hash"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (self.tmp / "squash-policy-eu.json").write_text(
            json.dumps({"passed": True, "policy": "EU-AIA-001"}),
            encoding="utf-8",
        )
        (self.tmp / "squash-policy-ntia.json").write_text(
            json.dumps({"passed": False, "policy": "NTIA-MIN"}),
            encoding="utf-8",
        )
        (self.tmp / "squash-vex-report.json").write_text(
            json.dumps(
                {
                    "statements": [
                        {"cve": "CVE-2024-1234", "status": "not_affected"},
                        {"cve": "CVE-2024-5678", "status": "affected"},
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_scan_summary_counts_findings(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        summary = gen._scan_summary()
        self.assertIn("2 finding(s)", summary)
        self.assertIn("error", summary)

    def test_policy_summary_counts_passing(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        summary = gen._policy_summary()
        self.assertIn("1/2", summary)

    def test_vex_summary_counts_statements(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        summary = gen._vex_summary()
        self.assertIn("2 CVE(s)", summary)
        self.assertIn("1 not affected", summary)

    def test_generate_all_with_artifacts(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("all")
        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(p.exists())
            content = p.read_text()
            self.assertGreater(len(content), 100)

    def test_hf_card_tags_include_int4(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("hf")
        content = paths[0].read_text()
        self.assertIn("int4", content.lower())

    def test_output_dir_override(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        out = self.tmp / "cards"
        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("hf", output_dir=out)
        self.assertEqual(paths[0].parent, out)
        self.assertTrue(paths[0].exists())


# ── Integration: bad JSON artifact ────────────────────────────────────────────


class TestModelCardGeneratorBadArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.tmp = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_bad_json_does_not_crash(self) -> None:
        from squish.squash.model_card import ModelCardGenerator

        (self.tmp / "squish.json").write_text("{not valid json!!!}", encoding="utf-8")
        # Should not raise — bad artifact is silently skipped
        gen = ModelCardGenerator(self.tmp)
        paths = gen.generate("hf")
        self.assertEqual(len(paths), 1)


# ── Integration: config overrides ─────────────────────────────────────────────


class TestModelCardConfigOverrides(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.tmp = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_model_id_override(self) -> None:
        from squish.squash.model_card import ModelCardConfig, ModelCardGenerator

        cfg = ModelCardConfig(model_dir=self.tmp, model_id="override/id")
        gen = ModelCardGenerator(self.tmp, config=cfg)
        self.assertEqual(gen._model_id(), "override/id")

    def test_model_name_override(self) -> None:
        from squish.squash.model_card import ModelCardConfig, ModelCardGenerator

        cfg = ModelCardConfig(model_dir=self.tmp, model_name="My Custom Model")
        gen = ModelCardGenerator(self.tmp, config=cfg)
        self.assertEqual(gen._model_name(), "My Custom Model")

    def test_license_appears_in_hf_card(self) -> None:
        from squish.squash.model_card import ModelCardConfig, ModelCardGenerator

        cfg = ModelCardConfig(model_dir=self.tmp, license="mit")
        gen = ModelCardGenerator(self.tmp, config=cfg)
        paths = gen.generate("hf")
        self.assertIn("mit", paths[0].read_text())


# ── Module: public API shape ──────────────────────────────────────────────────


class TestModelCardPublicAPI(unittest.TestCase):
    def test_known_formats_exported(self) -> None:
        from squish.squash.model_card import KNOWN_FORMATS

        self.assertIn("hf", KNOWN_FORMATS)
        self.assertIn("eu-ai-act", KNOWN_FORMATS)
        self.assertIn("iso-42001", KNOWN_FORMATS)
        self.assertIn("all", KNOWN_FORMATS)

    def test_exports_from_squash_package(self) -> None:
        from squish.squash import (
            MODEL_CARD_KNOWN_FORMATS,
            ModelCard,
            ModelCardConfig,
            ModelCardGenerator,
            ModelCardSection,
        )

        self.assertIsNotNone(ModelCard)
        self.assertIsNotNone(ModelCardConfig)
        self.assertIsNotNone(ModelCardGenerator)
        self.assertIsNotNone(ModelCardSection)
        self.assertIn("hf", MODEL_CARD_KNOWN_FORMATS)


# ── CLI smoke test ─────────────────────────────────────────────────────────────


class TestModelCardCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.tmp = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_cli_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "model-card", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("model-card", result.stdout + result.stderr)

    def test_cli_generates_hf_card(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "squish.squash.cli",
                "model-card", str(self.tmp), "--format", "hf",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue((self.tmp / "squash-model-card-hf.md").exists())

    def test_cli_missing_model_dir_exits_1(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "squish.squash.cli",
                "model-card", "/nonexistent/path/to/model",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)

    def test_cli_quiet_flag_suppresses_output(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "squish.squash.cli",
                "model-card", str(self.tmp), "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "")


# ── Module count gate ─────────────────────────────────────────────────────────


class TestModuleCountGate(unittest.TestCase):
    """Enforces the squash/ module count ceiling after adding model_card.py."""

    def test_squash_module_count_is_36(self) -> None:
        squash_dir = Path(__file__).parent.parent / "squish" / "squash"
        py_files = [
            f for f in squash_dir.rglob("*.py") if "__pycache__" not in str(f)
        ]
        count = len(py_files)
        # Exact gate: Wave 57 adds model_card.py → should be 36
        self.assertEqual(
            count,
            36,
            msg=f"squash/ has {count} Python files (expected 36 after Wave 57). "
                "If you added a file, delete one or file a written exception.",
        )


if __name__ == "__main__":
    unittest.main()
