"""
tests/test_phase_g_diffusion.py

Coverage tests for squish/diffusion_draft.py — Phase G1.

``load()`` and ``generate_short()`` are hardware-bound and marked
``# pragma: no cover``.  All other methods are fully exercised here.
"""

from __future__ import annotations

import pytest

from squish.diffusion_draft import DiffusionDraftModel


class TestDiffusionDraftModelInit:
    def test_defaults(self):
        m = DiffusionDraftModel("/path/to/model")
        assert m.model_path() == "/path/to/model"
        assert m.confidence_threshold() == DiffusionDraftModel.DEFAULT_CONFIDENCE_THRESHOLD
        assert m.max_suitable_tokens() == DiffusionDraftModel.DEFAULT_MAX_SUITABLE_TOKENS
        assert m._model is None
        assert m._tokenizer is None

    def test_custom_params(self):
        m = DiffusionDraftModel("~/models/llada", confidence_threshold=0.85, max_suitable_tokens=32)
        assert m.confidence_threshold() == 0.85
        assert m.max_suitable_tokens() == 32

    def test_model_path_stringified(self):
        from pathlib import Path  # noqa: PLC0415
        m = DiffusionDraftModel(Path("/abs/path"))
        assert m.model_path() == "/abs/path"


class TestDiffusionDraftModelAvailability:
    def test_not_available_before_load(self):
        m = DiffusionDraftModel("/model")
        assert m.is_available() is False

    def test_available_after_setting_model(self):
        m = DiffusionDraftModel("/model")
        m._model = object()  # simulate loaded
        assert m.is_available() is True


class TestDiffusionDraftModelSuitability:
    def test_short_task_suitable(self):
        m = DiffusionDraftModel("/model", max_suitable_tokens=64)
        assert m.is_suitable_for_task(1) is True
        assert m.is_suitable_for_task(64) is True

    def test_long_task_not_suitable(self):
        m = DiffusionDraftModel("/model", max_suitable_tokens=64)
        assert m.is_suitable_for_task(65) is False
        assert m.is_suitable_for_task(512) is False

    def test_boundary_exactly_max(self):
        m = DiffusionDraftModel("/model", max_suitable_tokens=32)
        assert m.is_suitable_for_task(32) is True

    def test_boundary_above_max(self):
        m = DiffusionDraftModel("/model", max_suitable_tokens=32)
        assert m.is_suitable_for_task(33) is False


class TestDiffusionDraftModelGetters:
    def test_model_path_getter(self):
        m = DiffusionDraftModel("/my/model")
        assert m.model_path() == "/my/model"

    def test_confidence_threshold_getter(self):
        m = DiffusionDraftModel("/m", confidence_threshold=0.5)
        assert m.confidence_threshold() == 0.5

    def test_max_suitable_tokens_getter(self):
        m = DiffusionDraftModel("/m", max_suitable_tokens=128)
        assert m.max_suitable_tokens() == 128

    def test_class_constants(self):
        assert DiffusionDraftModel.DEFAULT_CONFIDENCE_THRESHOLD == 0.7
        assert DiffusionDraftModel.DEFAULT_MAX_SUITABLE_TOKENS == 64
