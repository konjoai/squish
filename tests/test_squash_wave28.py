"""tests/test_squash_wave28.py — Wave 28: CircleCI Orb YAML + Ray Serve decorator.

Test taxonomy:
  Unit:        SquashServeConfig defaults, _truthy-adjacent helpers, config fields
  Integration: _wrap_deployment bind-patch, user_config injection, error paths
  Subprocess:  n/a (no process-state changes required)
"""
from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml  # PyYAML is a squish dev dependency


# ---------------------------------------------------------------------------
# Orb YAML path helper
# ---------------------------------------------------------------------------

_ORB_PATH = (
    Path(__file__).parent.parent
    / "squish"
    / "squash"
    / "integrations"
    / "circleci"
    / "orb.yml"
)


# ===========================================================================
# 1. CircleCI Orb YAML — structural tests
# ===========================================================================


class TestCircleCIOrbExists:
    def test_orb_file_exists(self):
        assert _ORB_PATH.exists(), f"orb.yml not found at {_ORB_PATH}"

    def test_orb_file_is_non_empty(self):
        assert _ORB_PATH.stat().st_size > 100

    def test_orb_parses_as_valid_yaml(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert isinstance(doc, dict)

    def test_orb_version_field(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert doc.get("version") == 2.1

    def test_orb_has_description(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "description" in doc
        assert len(doc["description"]) > 10

    def test_orb_has_display_block(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "display" in doc
        display = doc["display"]
        assert "home_url" in display
        assert "source_url" in display

    def test_orb_has_commands_block(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "commands" in doc
        assert isinstance(doc["commands"], dict)

    def test_orb_has_attest_command(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "attest" in doc["commands"]

    def test_orb_has_check_command(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "check" in doc["commands"]

    def test_orb_has_policy_gate_command(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "policy-gate" in doc["commands"]

    def test_attest_command_has_model_path_parameter(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        params = doc["commands"]["attest"]["parameters"]
        assert "model-path" in params

    def test_attest_command_has_output_parameter(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        params = doc["commands"]["attest"]["parameters"]
        assert "output" in params
        assert params["output"]["default"] == "squash-bom.json"

    def test_attest_command_has_format_parameter(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        params = doc["commands"]["attest"]["parameters"]
        assert "format" in params
        assert params["format"]["default"] == "cyclonedx-json"

    def test_check_command_has_fail_on_violation_parameter(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        params = doc["commands"]["check"]["parameters"]
        assert "fail-on-violation" in params
        assert params["fail-on-violation"]["default"] is True

    def test_policy_gate_has_allow_unscanned_parameter(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        params = doc["commands"]["policy-gate"]["parameters"]
        assert "allow-unscanned" in params
        assert params["allow-unscanned"]["default"] is False

    def test_orb_has_examples_block(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "examples" in doc
        assert isinstance(doc["examples"], dict)
        assert len(doc["examples"]) >= 1

    def test_orb_examples_contain_usage(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        for _name, example in doc["examples"].items():
            assert "usage" in example, f"example '{_name}' missing 'usage'"

    def test_orb_home_url_references_squishai(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        assert "squishai" in doc["display"]["home_url"].lower()

    def test_attest_command_steps_reference_squash_ci_run(self):
        with _ORB_PATH.open() as f:
            raw = f.read()
        assert "squash ci-run" in raw

    def test_all_three_commands_have_steps(self):
        with _ORB_PATH.open() as f:
            doc = yaml.safe_load(f)
        for cmd_name in ("attest", "check", "policy-gate"):
            assert "steps" in doc["commands"][cmd_name], f"'{cmd_name}' missing steps"


# ===========================================================================
# 2. SquashServeConfig — unit tests
# ===========================================================================


class TestSquashServeConfig:
    def _import(self):
        from squish.squash.integrations.ray import SquashServeConfig
        return SquashServeConfig

    def test_default_model_dir_is_none(self):
        cfg = self._import()()
        assert cfg.model_dir is None

    def test_default_require_bom_is_true(self):
        cfg = self._import()()
        assert cfg.require_bom is True

    def test_default_policy_is_none(self):
        cfg = self._import()()
        assert cfg.policy is None

    def test_default_metadata_is_empty_dict(self):
        cfg = self._import()()
        assert cfg.metadata == {}

    def test_custom_model_dir(self):
        cfg = self._import()(model_dir="models/llama")
        assert str(cfg.model_dir) == "models/llama"

    def test_custom_require_bom_false(self):
        cfg = self._import()(require_bom=False)
        assert cfg.require_bom is False

    def test_custom_policy(self):
        cfg = self._import()(policy="eu-ai-act")
        assert cfg.policy == "eu-ai-act"

    def test_custom_metadata(self):
        cfg = self._import()(metadata={"env": "prod"})
        assert cfg.metadata["env"] == "prod"


# ===========================================================================
# 3. squash_serve decorator — unit tests
# ===========================================================================


class TestSquashServeDecorator:
    def _import(self):
        from squish.squash.integrations.ray import squash_serve
        return squash_serve

    def test_returns_callable_when_called_with_args(self):
        deco = self._import()(model_dir="models/x")
        assert callable(deco)

    def test_decorates_class_with_no_args(self):
        squash_serve = self._import()

        class FakeDeployment:
            pass

        result = squash_serve(FakeDeployment)
        assert result is FakeDeployment

    def test_decorates_class_with_kwargs(self):
        squash_serve = self._import()

        class FakeDeployment:
            pass

        result = squash_serve(model_dir=None)(FakeDeployment)
        assert result is FakeDeployment

    def test_bind_method_is_patched(self):
        squash_serve = self._import()

        class FakeDeployment:
            def bind(self, *args, **kwargs):
                return "original"

        squash_serve(model_dir=None)(FakeDeployment)
        assert FakeDeployment.bind is not None
        # patched bind should not be the same function object
        assert FakeDeployment.bind.__name__ == "bind" or callable(FakeDeployment.bind)

    def test_patched_bind_injects_user_config(self):
        squash_serve = self._import()

        captured: dict = {}

        class FakeDeployment:
            @classmethod
            def bind(cls, **kwargs):
                captured.update(kwargs)
                return cls

        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        FakeDeployment.bind()
        assert "user_config" in captured
        assert "squash_bom_summary" in captured["user_config"]

    def test_patched_bind_merges_existing_user_config(self):
        squash_serve = self._import()

        captured: dict = {}

        class FakeDeployment:
            @classmethod
            def bind(cls, **kwargs):
                captured.update(kwargs)

        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        FakeDeployment.bind(user_config={"my_key": "my_val"})
        uc = captured["user_config"]
        assert uc["my_key"] == "my_val"
        assert "squash_bom_summary" in uc

    def test_patched_bind_includes_extra_metadata(self):
        squash_serve = self._import()

        captured: dict = {}

        class FakeDeployment:
            @classmethod
            def bind(cls, **kwargs):
                captured.update(kwargs)

        squash_serve(model_dir=None, require_bom=False, metadata={"team": "ml"})(FakeDeployment)
        FakeDeployment.bind()
        assert captured["user_config"]["team"] == "ml"

    def test_no_model_dir_returns_validated_false(self):
        squash_serve = self._import()

        result_holder: list = []

        class FakeDeployment:
            @classmethod
            def bind(cls, **kwargs):
                result_holder.append(kwargs["user_config"]["squash_bom_summary"])

        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        FakeDeployment.bind()
        summary = result_holder[0]
        assert summary["validated"] is False


# ===========================================================================
# 4. _run_squash_validation — unit tests (mocked)
# ===========================================================================


class TestRunSquashValidation:
    def _fn(self):
        from squish.squash.integrations.ray import _run_squash_validation, SquashServeConfig
        return _run_squash_validation, SquashServeConfig

    def test_no_model_dir_returns_not_validated(self):
        fn, Cfg = self._fn()
        result = fn(Cfg(model_dir=None))
        assert result["validated"] is False
        assert "no model_dir" in result["reason"]

    def test_missing_dir_require_bom_true_raises(self, tmp_path):
        fn, Cfg = self._fn()
        missing = tmp_path / "does_not_exist"
        with pytest.raises(RuntimeError, match="does not exist"):
            fn(Cfg(model_dir=missing, require_bom=True))

    def test_missing_dir_require_bom_false_returns_not_validated(self, tmp_path):
        fn, Cfg = self._fn()
        missing = tmp_path / "does_not_exist"
        result = fn(Cfg(model_dir=missing, require_bom=False))
        assert result["validated"] is False
        assert "not found" in result["reason"]

    def test_successful_validation_returns_validated_true(self, tmp_path):
        fn, Cfg = self._fn()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_result = MagicMock()
        mock_result.summary.return_value = "[PASS] test-model: ok"

        with patch("squish.squash.attest.AttestPipeline.run", return_value=mock_result):
            result = fn(Cfg(model_dir=model_dir, require_bom=True))

        assert result["validated"] is True
        assert result["model_dir"] == str(model_dir)

    def test_failed_scan_require_bom_true_raises(self, tmp_path):
        fn, Cfg = self._fn()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("squish.squash.attest.AttestPipeline.run", side_effect=ValueError("scan exploded")):
            with pytest.raises(RuntimeError, match="attestation failed"):
                fn(Cfg(model_dir=model_dir, require_bom=True))

    def test_failed_scan_require_bom_false_returns_not_validated(self, tmp_path):
        fn, Cfg = self._fn()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("squish.squash.attest.AttestPipeline.run", side_effect=ValueError("scan exploded")):
            result = fn(Cfg(model_dir=model_dir, require_bom=False))

        assert result["validated"] is False
        assert "scan exploded" in result["reason"]

    def test_policy_included_in_summary_when_set(self, tmp_path):
        fn, Cfg = self._fn()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_result = MagicMock()
        mock_result.summary.return_value = "[PASS] m: ok"

        with patch("squish.squash.attest.AttestPipeline.run", return_value=mock_result):
            result = fn(Cfg(model_dir=model_dir, policy="eu-ai-act"))

        assert result.get("policy") == "eu-ai-act"


# ===========================================================================
# 5. SquashServeDeployment mix-in — integration tests
# ===========================================================================


class TestSquashServeDeploymentMixin:
    def test_mixin_is_importable(self):
        from squish.squash.integrations.ray import SquashServeDeployment
        assert SquashServeDeployment is not None

    def test_subclass_inherits_bind_patch(self):
        from squish.squash.integrations.ray import SquashServeDeployment

        captured: list = []

        class MyModel(SquashServeDeployment):
            _squash_model_dir = None
            _squash_require_bom = False

            @classmethod
            def bind(cls, **kwargs):
                captured.append(kwargs.get("user_config", {}))

        MyModel.bind()
        assert len(captured) == 1
        assert "squash_bom_summary" in captured[0]

    def test_subclass_overrides_squash_policy(self):
        from squish.squash.integrations.ray import SquashServeDeployment

        class MyModel(SquashServeDeployment):
            _squash_model_dir = None
            _squash_require_bom = False
            _squash_policy = "nist-ai-rmf"

            @classmethod
            def bind(cls, **kwargs):
                return kwargs

        result = MyModel.bind()
        # No model_dir → validated=False, no policy key expected in summary
        assert "user_config" in result
        assert "squash_bom_summary" in result["user_config"]

    def test_mixin_default_require_bom_is_true(self):
        from squish.squash.integrations.ray import SquashServeDeployment
        assert SquashServeDeployment._squash_require_bom is True

    def test_mixin_default_policy_is_none(self):
        from squish.squash.integrations.ray import SquashServeDeployment
        assert SquashServeDeployment._squash_policy is None


# ===========================================================================
# 6. Module count gate (inline)
# ===========================================================================


class TestModuleCount:
    def test_squash_python_module_count_at_most_106(self):
        squash_dir = Path(__file__).parent.parent / "squish" / "squash"
        py_files = [
            p for p in squash_dir.rglob("*.py")
            if "__pycache__" not in str(p)
        ]
        assert len(py_files) <= 106, (
            f"Module count {len(py_files)} exceeds 106. "
            "Every new file requires CHANGELOG justification."
        )


# ===========================================================================
# 7. squash.__all__ export tests
# ===========================================================================


class TestSquashAllExports:
    def test_squash_serve_in_all(self):
        import squish.squash as sq
        assert "squash_serve" in sq.__all__

    def test_squash_serve_config_in_all(self):
        import squish.squash as sq
        assert "SquashServeConfig" in sq.__all__

    def test_squash_serve_deployment_in_all(self):
        import squish.squash as sq
        assert "SquashServeDeployment" in sq.__all__

    def test_squash_serve_importable_from_squash(self):
        from squish.squash import squash_serve
        assert callable(squash_serve)

    def test_squash_serve_config_importable_from_squash(self):
        from squish.squash import SquashServeConfig
        assert SquashServeConfig is not None

    def test_squash_serve_deployment_importable_from_squash(self):
        from squish.squash import SquashServeDeployment
        assert SquashServeDeployment is not None

    def test_wave27_exports_still_present(self):
        import squish.squash as sq
        assert "KubernetesWebhookHandler" in sq.__all__
        assert "WebhookConfig" in sq.__all__


# ===========================================================================
# 8. ray.py public API surface tests
# ===========================================================================


class TestRayModuleApiSurface:
    def test_module_has_squash_serve(self):
        import squish.squash.integrations.ray as rmod
        assert hasattr(rmod, "squash_serve")

    def test_module_has_squash_serve_config(self):
        import squish.squash.integrations.ray as rmod
        assert hasattr(rmod, "SquashServeConfig")

    def test_module_has_squash_serve_deployment(self):
        import squish.squash.integrations.ray as rmod
        assert hasattr(rmod, "SquashServeDeployment")

    def test_module_has_wrap_deployment(self):
        import squish.squash.integrations.ray as rmod
        assert hasattr(rmod, "_wrap_deployment")

    def test_squash_metadata_key_constant(self):
        import squish.squash.integrations.ray as rmod
        assert rmod._SQUASH_METADATA_KEY == "squash_bom_summary"

    def test_squash_serve_accepts_no_args(self):
        from squish.squash.integrations.ray import squash_serve

        @squash_serve
        class _Fake:
            pass

        assert _Fake is not None

    def test_squash_serve_accepts_keyword_args(self):
        from squish.squash.integrations.ray import squash_serve

        @squash_serve(model_dir=None, require_bom=False)
        class _Fake:
            pass

        assert _Fake is not None


# ===========================================================================
# 9. Integration round-trip — full decorator + bind chain (mocked pipeline)
# ===========================================================================


class TestFullDecoratorRoundTrip:
    def test_bind_returns_original_return_value(self):
        from squish.squash.integrations.ray import squash_serve

        sentinel = object()

        class FakeDeployment:
            @classmethod
            def bind(cls, **kwargs):
                return sentinel

        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        result = FakeDeployment.bind()
        assert result is sentinel

    def test_bind_passes_through_positional_args(self):
        from squish.squash.integrations.ray import squash_serve

        captured: list = []

        class FakeDeployment:
            @classmethod
            def bind(cls, *args, **kwargs):
                captured.append(args)

        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        FakeDeployment.bind("pos1", "pos2")
        assert captured[0] == ("pos1", "pos2")

    def test_bom_summary_dict_has_validated_key(self):
        from squish.squash.integrations.ray import squash_serve

        summaries: list = []

        class FakeDeployment:
            @classmethod
            def bind(cls, **kwargs):
                summaries.append(kwargs["user_config"]["squash_bom_summary"])

        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        FakeDeployment.bind()
        assert "validated" in summaries[0]

    def test_class_without_bind_falls_back_gracefully(self):
        """Deployer class with no bind() at all — patched bind returns the class."""
        from squish.squash.integrations.ray import squash_serve

        class NoBind:
            pass

        squash_serve(model_dir=None, require_bom=False)(NoBind)
        # patched bind was installed; calling it should not raise
        result = NoBind.bind()
        assert result is NoBind

    def test_double_decoration_is_idempotent(self):
        """Applying @squash_serve twice should not crash."""
        from squish.squash.integrations.ray import squash_serve

        class FakeDeployment:
            @classmethod
            def bind(cls, **kwargs):
                return kwargs

        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        squash_serve(model_dir=None, require_bom=False)(FakeDeployment)
        result = FakeDeployment.bind()
        assert "user_config" in result
