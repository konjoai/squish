"""tests/test_wave130_vlm_backend_resolver.py

Wave 130 — architecture resolver for the mlx_lm / mlx_vlm dual runtime split.

``mlx_lm`` only implements text-only decoder architectures; a growing set of
``mlx-community`` checkpoints (Gemma 4, VLM/omni families) declare a
``model_type`` it doesn't know. ``squish.runtime.arch_resolver`` decides,
from ``config.json`` alone, which runtime backend a model directory should
load through — without loading any weights.

This pins:
- a known mlx_lm model_type resolves to "mlx_lm" via the cheap importlib probe
- an unknown model_type with mlx_vlm installed falls back to "mlx_vlm"
- an unknown model_type with mlx_vlm NOT installed raises a clear, actionable
  UnsupportedArchitectureError (never a silent failure or a stack trace)
- the JSON sidecar caches the resolution so a second call doesn't re-probe
- a missing config.json / missing model_type field raises a clear error
- squish.quant.compressed_loader._instantiate_model dispatches on the
  resolver: known mlx_lm types build via the unchanged path; unknown types
  build a real mlx_vlm model skeleton (verified against the real installed
  mlx_vlm package, not a mock — a tiny gemma4_unified config, vision/audio
  towers disabled, keeps this fast)
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

# arch_resolver's mlx_lm probe needs a real, importable mlx_lm — in the VS
# Code sandbox `import mlx_lm.*` is blocked to prevent a Metal-init SIGABRT
# (see tests/conftest.py); skip this module there rather than fail. Run with
# CI=1 (or outside the sandbox) to exercise it, same convention as tests/kv,
# tests/quant, etc.
pytest.importorskip(
    "mlx_lm.models.llama",
    reason="mlx_lm blocked in sandbox — set CI=1 to enable",
    exc_type=ImportError,
)

from squish.runtime import arch_resolver


def _write_config(model_dir: Path, model_type: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))


class TestKnownMlxLmType:
    def test_resolves_to_mlx_lm(self, tmp_path):
        _write_config(tmp_path, "llama")
        assert arch_resolver.resolve_runtime(tmp_path, use_cache=False) == "mlx_lm"

    def test_remapped_type_resolves_to_mlx_lm(self, tmp_path):
        # "mistral" remaps to mlx_lm.models.llama per _MODEL_REMAPPING.
        _write_config(tmp_path, "mistral")
        assert arch_resolver.resolve_runtime(tmp_path, use_cache=False) == "mlx_lm"


class TestUnknownTypeFallback:
    def test_falls_back_to_mlx_vlm_when_installed(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "totally_unknown_arch_xyz")
        monkeypatch.setattr(arch_resolver, "_mlx_lm_supports", lambda _t: False)
        monkeypatch.setattr(arch_resolver, "_mlx_vlm_available", lambda: True)
        assert arch_resolver.resolve_runtime(tmp_path, use_cache=False) == "mlx_vlm"

    def test_raises_actionable_error_when_mlx_vlm_missing(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "totally_unknown_arch_xyz")
        monkeypatch.setattr(arch_resolver, "_mlx_lm_supports", lambda _t: False)
        monkeypatch.setattr(arch_resolver, "_mlx_vlm_available", lambda: False)
        with pytest.raises(arch_resolver.UnsupportedArchitectureError) as exc:
            arch_resolver.resolve_runtime(tmp_path, use_cache=False)
        assert "pip install" in str(exc.value)
        assert "multimodal" in str(exc.value)


class TestSidecarCache:
    def test_second_call_uses_cache_without_reprobing(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "llama")
        assert arch_resolver.resolve_runtime(tmp_path) == "mlx_lm"
        assert (tmp_path / arch_resolver._SIDECAR_NAME).exists()

        # If a cached call re-probed, this would flip the result to mlx_vlm.
        monkeypatch.setattr(arch_resolver, "_mlx_lm_supports", lambda _t: False)
        monkeypatch.setattr(arch_resolver, "_mlx_vlm_available", lambda: True)
        assert arch_resolver.resolve_runtime(tmp_path) == "mlx_lm"

    def test_use_cache_false_forces_reprobe(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "llama")
        assert arch_resolver.resolve_runtime(tmp_path) == "mlx_lm"

        monkeypatch.setattr(arch_resolver, "_mlx_lm_supports", lambda _t: False)
        monkeypatch.setattr(arch_resolver, "_mlx_vlm_available", lambda: True)
        assert arch_resolver.resolve_runtime(tmp_path, use_cache=False) == "mlx_vlm"

    def test_corrupt_sidecar_is_ignored_not_fatal(self, tmp_path):
        _write_config(tmp_path, "llama")
        (tmp_path / arch_resolver._SIDECAR_NAME).write_text("{not json")
        assert arch_resolver.resolve_runtime(tmp_path) == "mlx_lm"


class TestConfigErrors:
    def test_missing_config_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            arch_resolver.resolve_runtime(tmp_path, use_cache=False)

    def test_missing_model_type_field_raises(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 4096}))
        with pytest.raises(ValueError, match="model_type"):
            arch_resolver.resolve_runtime(tmp_path, use_cache=False)


class TestMlxVlmAvailabilityProbe:
    def test_reports_absence_cleanly_rather_than_raising(self, monkeypatch):
        # The multimodal extra is opt-in — whether or not it happens to be
        # installed in a given dev environment, the probe must never raise;
        # it's a pure availability check consumed by resolve_runtime's
        # fallback branch.
        real_import = importlib.import_module

        def _blocked(name, *a, **k):
            if name == "mlx_vlm":
                raise ImportError("simulated: mlx_vlm not installed")
            return real_import(name, *a, **k)

        monkeypatch.setattr(arch_resolver.importlib, "import_module", _blocked)
        assert arch_resolver._mlx_vlm_available() is False

    def test_reports_presence_when_importable(self):
        pytest.importorskip("mlx_vlm", reason="mlx-vlm extra not installed")
        assert arch_resolver._mlx_vlm_available() is True


# ── Wave 130 — compressed_loader._instantiate_model dispatch ─────────────────

_TINY_LLAMA_CONFIG = {
    "model_type": "llama",
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 32,
}

# vision_config/audio_config: None skips vision-tower/audio-tower construction
# entirely (mlx_vlm.models.gemma4_unified.Model.__init__ only builds them when
# not None) — keeps this a fast, genuinely text-only skeleton build.
_TINY_GEMMA4_CONFIG = {
    "model_type": "gemma4_unified",
    "vision_config": None,
    "audio_config": None,
    "hidden_size": 32,
    "vocab_size": 100,
    "text_config": {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "head_dim": 16,
        "global_head_dim": 16,
        "vocab_size": 100,
        "vocab_size_per_layer_input": 100,
        "num_key_value_heads": 1,
        "num_global_key_value_heads": 1,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
    },
}


def _write_json_config(model_dir: Path, config: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))


class TestInstantiateModelDispatch:
    def test_known_mlx_lm_type_builds_via_unchanged_path(self, tmp_path):
        from squish.quant.compressed_loader import _instantiate_model

        _write_json_config(tmp_path, _TINY_LLAMA_CONFIG)
        model, mlx_type = _instantiate_model(str(tmp_path))
        assert mlx_type == "llama"
        assert getattr(model, "__squish_runtime__", "mlx_lm") == "mlx_lm"

    def test_unknown_type_builds_real_mlx_vlm_skeleton(self, tmp_path):
        pytest.importorskip("mlx_vlm", reason="mlx-vlm extra not installed")
        from squish.quant.compressed_loader import _instantiate_model

        _write_json_config(tmp_path, _TINY_GEMMA4_CONFIG)
        model, mlx_type = _instantiate_model(str(tmp_path))
        assert mlx_type == "gemma4_unified"
        assert model.__squish_runtime__ == "mlx_vlm"
        assert model.vision_embedder is None  # text-only: vision tower skipped
        assert model.embed_audio is None  # text-only: audio tower skipped

    def test_unknown_type_without_mlx_vlm_raises_actionable_error(self, tmp_path, monkeypatch):
        import builtins

        from squish.quant.compressed_loader import _instantiate_model_mlx_vlm

        real_import = builtins.__import__

        def _blocked(name, *a, **k):
            if name == "mlx_vlm.utils" or name.startswith("mlx_vlm.utils"):
                raise ImportError("simulated: mlx_vlm not installed")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(ValueError, match="multimodal"):
            _instantiate_model_mlx_vlm(_TINY_GEMMA4_CONFIG)


class TestInstantiateModelMlxVlmSkeleton:
    def test_skeleton_has_expected_shapes(self, tmp_path):
        pytest.importorskip("mlx_vlm", reason="mlx-vlm extra not installed")
        from squish.quant.compressed_loader import _instantiate_model_mlx_vlm

        model, mlx_type = _instantiate_model_mlx_vlm(_TINY_GEMMA4_CONFIG)
        assert mlx_type == "gemma4_unified"
        assert model.vocab_size == 100
        assert len(model.language_model.layers) == 2

    def test_does_not_mutate_caller_config(self, tmp_path):
        pytest.importorskip("mlx_vlm", reason="mlx-vlm extra not installed")
        from squish.quant.compressed_loader import _instantiate_model_mlx_vlm

        config = json.loads(json.dumps(_TINY_GEMMA4_CONFIG))  # deep copy
        original = json.loads(json.dumps(config))
        _instantiate_model_mlx_vlm(config)
        assert config == original
