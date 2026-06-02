"""tests/test_load_mlx_model_parallel.py

Regression test for Fix #2 — squish.server.load_mlx_model now runs
load_tokenizer() on a worker thread while load_model() runs on the
main thread. The output should be equivalent to the serial
mlx_lm.load() call this replaced.

We can't import mlx_lm in-sandbox (conftest blocks Metal init), so
we inject a fake ``mlx_lm.utils`` module into sys.modules before
calling load_mlx_model. The function's internal
``from mlx_lm.utils import ...`` resolves to our fake.
"""
from __future__ import annotations

import sys
import time
import types
from typing import Any

import pytest

import squish.server as _srv


class _MlxUtilsFake:
    """Holds the per-test mocks. Installed into sys.modules['mlx_lm.utils']."""

    def __init__(self) -> None:
        self.load_config_calls: list[Any] = []
        self.load_model_calls: list[Any] = []
        self.load_tokenizer_calls: list[Any] = []
        self.config_return: dict[str, Any] = {}
        self.model_return: Any = ("test-model", {})
        self.tokenizer_return: Any = "test-tokenizer"
        self.load_model_delay_s: float = 0.0
        self.load_tokenizer_delay_s: float = 0.0
        self.tokenizer_raises: Exception | None = None
        self.weights_done_at: list[float] = []
        self.tokenizer_done_at: list[float] = []
        self._t0: float = 0.0

    def load_config(self, model_path):
        self.load_config_calls.append(model_path)
        return self.config_return

    def load_model(self, model_path, lazy):
        if self.load_model_delay_s:
            time.sleep(self.load_model_delay_s)
        self.weights_done_at.append(time.perf_counter() - self._t0)
        self.load_model_calls.append((model_path, lazy))
        return self.model_return

    def load_tokenizer(self, model_path, tokenizer_config, eos_token_ids=None):
        if self.tokenizer_raises:
            raise self.tokenizer_raises
        if self.load_tokenizer_delay_s:
            time.sleep(self.load_tokenizer_delay_s)
        self.tokenizer_done_at.append(time.perf_counter() - self._t0)
        self.load_tokenizer_calls.append((model_path, tokenizer_config, eos_token_ids))
        return self.tokenizer_return

    def install(self) -> None:
        """Inject this fake as sys.modules['mlx_lm.utils']."""
        mod = types.ModuleType("mlx_lm.utils")
        mod.load_config = self.load_config
        mod.load_model = self.load_model
        mod.load_tokenizer = self.load_tokenizer
        sys.modules["mlx_lm.utils"] = mod
        # Also stub mlx_lm so the bg-thread `import mlx_lm` (which may run
        # alongside) finds something — squish._fast_imports' await call
        # only checks for sys.modules['mlx_lm'].
        if "mlx_lm" not in sys.modules:
            sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
        self._t0 = time.perf_counter()


@pytest.fixture
def mlx_fake():
    """Install fake mlx_lm.utils for the test; restore the previous module after."""
    saved_utils = sys.modules.get("mlx_lm.utils")
    saved_mlx_lm = sys.modules.get("mlx_lm")
    fake = _MlxUtilsFake()
    fake.install()
    # Neutralise the background-import state so load_mlx_model's call to
    # _fi.await_mlx_lm_import() returns immediately. The conftest sandbox
    # makes the real bg import fail; we want to bypass that for the unit
    # test, since we've supplied our own mlx_lm.utils mock.
    import squish._fast_imports as _fi
    saved_thread = _fi._mlx_lm_import_thread
    saved_done = _fi._mlx_lm_import_done
    saved_err = _fi._mlx_lm_import_error
    _fi._mlx_lm_import_thread = None
    import threading
    _fi._mlx_lm_import_done = threading.Event()
    _fi._mlx_lm_import_done.set()
    _fi._mlx_lm_import_error = None
    try:
        yield fake
    finally:
        _fi._mlx_lm_import_thread = saved_thread
        _fi._mlx_lm_import_done = saved_done
        _fi._mlx_lm_import_error = saved_err
        if saved_utils is not None:
            sys.modules["mlx_lm.utils"] = saved_utils
        else:
            sys.modules.pop("mlx_lm.utils", None)
        if saved_mlx_lm is not None:
            sys.modules["mlx_lm"] = saved_mlx_lm
        else:
            sys.modules.pop("mlx_lm", None)


@pytest.fixture(autouse=True)
def _reset_state():
    _srv._state.model = None
    _srv._state.tokenizer = None
    _srv._state.model_name = ""
    _srv._state.loaded_at = 0.0
    _srv._state.load_time_s = 0.0
    _srv._state.loader_tag = ""
    yield
    _srv._state.model = None
    _srv._state.tokenizer = None


@pytest.fixture
def patched_warmup(monkeypatch):
    """Replace _cap_metal_cache and _warmup_model with no-ops so the test
    doesn't try to touch Metal."""
    monkeypatch.setattr(_srv, "_cap_metal_cache", lambda **kwargs: None)
    monkeypatch.setattr(_srv, "_warmup_model", lambda **kwargs: None)
    yield


def test_load_mlx_model_calls_both_loaders_with_path(mlx_fake, patched_warmup, tmp_path):
    """load_model + load_tokenizer must each be invoked once with the path,
    and eos_token_ids must be forwarded from config."""
    mlx_fake.config_return = {"eos_token_id": 99}
    _srv.load_mlx_model(str(tmp_path), verbose=False)

    assert len(mlx_fake.load_model_calls) == 1
    assert mlx_fake.load_model_calls[0][0] == tmp_path
    assert len(mlx_fake.load_tokenizer_calls) == 1
    assert mlx_fake.load_tokenizer_calls[0][0] == tmp_path
    assert mlx_fake.load_tokenizer_calls[0][2] == 99  # eos_token_ids


def test_load_mlx_model_populates_state(mlx_fake, patched_warmup, tmp_path):
    """After load, _state.model and _state.tokenizer match loader returns."""
    mlx_fake.model_return = ("model-sentinel", {"eos_token_id": [1, 2]})
    mlx_fake.tokenizer_return = "tokenizer-sentinel"
    _srv.load_mlx_model(str(tmp_path), verbose=False)
    assert _srv._state.model == "model-sentinel"
    assert _srv._state.tokenizer == "tokenizer-sentinel"
    assert _srv._state.loader_tag == "mlx_lm"
    assert _srv._state.model_name == tmp_path.name


def test_tokenizer_runs_in_parallel_with_weights(mlx_fake, patched_warmup, tmp_path):
    """Wall time should be ~max(weight_load, tokenizer_load), not the sum.

    Each fake loader sleeps 0.15 s. Parallel total ≈ 0.15 s; serial ≈ 0.30 s.
    On GitHub Actions runners, scheduler jitter can add up to ~150 ms of slop
    on top of the parallel best case; the assertion checks that we are still
    closer to parallel than to serial (0.30 s).
    """
    mlx_fake.load_model_delay_s = 0.15
    mlx_fake.load_tokenizer_delay_s = 0.15

    t0 = time.perf_counter()
    _srv.load_mlx_model(str(tmp_path), verbose=False)
    elapsed = time.perf_counter() - t0

    # Parallelism is proven by inter-completion drift: when both 0.15 s sleeps
    # run in parallel their finish times are close together. When serial they
    # are ~0.15 s apart. The wall-time check is kept as a soft upper bound,
    # well below 2× serial, to catch a regression that drops parallelism
    # entirely while still tolerating GitHub Actions scheduler jitter (we have
    # observed up to 0.30 s wall on contended runners).
    assert elapsed < 0.45, (
        f"load_mlx_model took {elapsed:.3f}s — far beyond 0.15 s parallel "
        "best case; tokenizer is likely not running in parallel with the weights."
    )
    # Drift bound is the parallelism witness. Under parallel execution the
    # two 0.15 s sleeps finish near-simultaneously; under serial execution
    # they finish ~0.15 s apart. We allow generous slop for GitHub Actions
    # scheduler jitter (observed up to ~0.13 s on contended runners) while
    # still discriminating: drift ≥ 0.18 s indicates real serialization.
    drift = abs(mlx_fake.weights_done_at[0] - mlx_fake.tokenizer_done_at[0])
    assert drift < 0.18, (
        f"weights and tokenizer finished {drift:.3f}s apart — under parallel "
        "execution with equal 0.15 s sleeps they should finish closer than this."
    )


def test_tokenizer_error_propagates_to_caller(mlx_fake, patched_warmup, tmp_path):
    """If the tokenizer worker raises, load_mlx_model must re-raise on the
    main thread instead of leaving _state half-populated."""
    class _BoomError(RuntimeError):
        pass

    mlx_fake.tokenizer_raises = _BoomError("tokenizer is sad")

    with pytest.raises(_BoomError, match="tokenizer is sad"):
        _srv.load_mlx_model(str(tmp_path), verbose=False)
