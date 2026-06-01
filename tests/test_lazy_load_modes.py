"""tests/test_lazy_load_modes.py

Regression tests for the --lazy and --preload-async serving modes added
alongside the Ollama fair-comparison benchmark.

What's covered
──────────────
 1. Eager (default) mode:
    * `_LOAD_MODE == "eager"` by default
    * /health reports model_loaded=True after the eager path has loaded
    * /model/status agrees with /health
 2. Lazy mode:
    * Server module imports without loading a model
    * /health responds with status="ready" and model_loaded=False
      before any inference request
    * The first chat-completions request triggers _do_model_load
    * Subsequent requests skip the load (idempotent under _LOAD_LOCK)
 3. Preload-async mode:
    * Background thread drives the load while /health remains responsive
    * /health flips to model_loaded=True once the thread completes

We don't actually load a 7B model in these tests — that's both slow and
not testing what these tests aim to verify. Instead we install a stub
load function that flips `_state.model` to a sentinel object and sets
`_LOAD_COMPLETE`. The contract under test is the *control-flow* of the
load modes, not the model loader itself (which is covered elsewhere).
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import squish.server as _srv


# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeModel:
    """Minimal stand-in for a loaded mlx_lm model."""
    name = "test-model"


class _FakeTokenizer:
    """Minimal stand-in for a tokenizer."""
    pass


def _install_fast_loader(load_delay_s: float = 0.0):
    """Replace _do_model_load's underlying loaders with a fast stub.

    Returns a context manager-ish tuple (load_called_event, cleanup_fn).
    Tests call cleanup_fn() at teardown to restore the real loaders.
    """
    load_called = threading.Event()

    def fake_load_mlx_model(mlx_model_dir, verbose=False):
        if load_delay_s:
            time.sleep(load_delay_s)
        _srv._state.model = _FakeModel()
        _srv._state.tokenizer = _FakeTokenizer()
        _srv._state.model_name = "test-model"
        _srv._state.loaded_at = time.time()
        _srv._state.load_time_s = load_delay_s
        _srv._state.loader_tag = "test-stub"
        load_called.set()

    def fake_load_model(model_dir, compressed_dir, verbose=False):
        fake_load_mlx_model(model_dir, verbose=verbose)

    orig_load_mlx = _srv.load_mlx_model
    orig_load = _srv.load_model
    _srv.load_mlx_model = fake_load_mlx_model
    _srv.load_model = fake_load_model

    def cleanup() -> None:
        _srv.load_mlx_model = orig_load_mlx
        _srv.load_model = orig_load

    return load_called, cleanup


def _reset_server_state() -> None:
    """Reset module-level load state between tests."""
    _srv._state.model = None
    _srv._state.tokenizer = None
    _srv._state.model_name = ""
    _srv._state.loaded_at = 0.0
    _srv._state.load_time_s = 0.0
    _srv._state.loader_tag = ""
    _srv._LOAD_MODE = "eager"
    _srv._LOAD_ARGS = None
    _srv._LOAD_ERROR = None
    _srv._LOAD_COMPLETE.clear()


@pytest.fixture(autouse=True)
def _isolate_load_state():
    """Each test starts with a clean global load state and gets it reset after."""
    _reset_server_state()
    yield
    _reset_server_state()


@pytest.fixture()
def client():
    """TestClient with no API-key auth."""
    orig_key = _srv._API_KEY
    _srv._API_KEY = None
    c = TestClient(_srv.app, raise_server_exceptions=False)
    yield c
    _srv._API_KEY = orig_key


class _FakeArgs:
    """argparse.Namespace stand-in with the fields _do_model_load reads."""
    def __init__(self, mlx_model_dir: str = "", model_dir: str = "/tmp/x",
                 compressed_dir: str = "/tmp/x-c", verbose: bool = False) -> None:
        self.mlx_model_dir = mlx_model_dir
        self.model_dir = model_dir
        self.compressed_dir = compressed_dir
        self.verbose = verbose


# ═══════════════════════════════════════════════════════════════════════════════
# Eager mode — default behavior unchanged
# ═══════════════════════════════════════════════════════════════════════════════


def test_eager_is_default_mode():
    """`_LOAD_MODE` defaults to "eager" before main() runs."""
    assert _srv._LOAD_MODE == "eager"


def test_eager_mode_reports_loaded_after_load(client):
    """After a successful load, /health and /model/status both report loaded."""
    load_called, cleanup = _install_fast_loader()
    try:
        _srv._LOAD_MODE = "eager"
        _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")
        # Simulate main()'s eager path: directly call the underlying loader
        # and set the event (which is what main() does inside the with-span).
        _srv.load_mlx_model("/tmp/fake", verbose=False)
        _srv._LOAD_COMPLETE.set()

        r = client.get("/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_loaded"] is True
        assert body["status"] == "ok"
        assert body["load_mode"] == "eager"

        r2 = client.get("/model/status")
        assert r2.status_code == 200
        assert r2.json()["model_loaded"] is True
        assert r2.json()["load_mode"] == "eager"
    finally:
        cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
# Lazy mode — bind first, load on first request
# ═══════════════════════════════════════════════════════════════════════════════


def test_lazy_health_reports_ready_without_model(client):
    """Before any request, /health says status=ready and model_loaded=False."""
    _srv._LOAD_MODE = "lazy"
    _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")
    # _LOAD_COMPLETE deliberately left unset

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is False
    assert body["load_mode"] == "lazy"


def test_lazy_model_status_endpoint_pre_load(client):
    """/model/status reports the lazy load mode before any request."""
    _srv._LOAD_MODE = "lazy"
    _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")

    r = client.get("/model/status")
    assert r.status_code == 200
    body = r.json()
    assert body["load_mode"] == "lazy"
    assert body["model_loaded"] is False
    assert body["load_error"] is None


def test_lazy_first_request_triggers_load_then_subsequent_skip(client):
    """
    First call to _ensure_loaded_blocking() drives the load. The second
    call observes _LOAD_COMPLETE set and returns immediately without
    re-invoking the loader.
    """
    load_called, cleanup = _install_fast_loader()
    try:
        _srv._LOAD_MODE = "lazy"
        _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")

        assert not _srv._LOAD_COMPLETE.is_set()
        assert not load_called.is_set()

        _srv._ensure_loaded_blocking()
        assert load_called.is_set(), "First call should have triggered the loader"
        assert _srv._LOAD_COMPLETE.is_set()
        assert _srv._state.model is not None

        # Second call: clear the load_called flag and verify it's NOT re-triggered
        load_called.clear()
        _srv._ensure_loaded_blocking()
        assert not load_called.is_set(), \
            "Second call must not re-invoke the loader once _LOAD_COMPLETE is set"
    finally:
        cleanup()


def test_lazy_concurrent_first_requests_load_only_once():
    """
    Two threads racing through _ensure_loaded_blocking together must
    result in exactly one underlying load. The slow-loader stub sleeps
    long enough that both threads enter the function before either
    finishes, validating the lock + double-checked is_set() guard.
    """
    load_call_count = {"n": 0}
    call_lock = threading.Lock()

    def counting_load(mlx_model_dir, verbose=False):
        with call_lock:
            load_call_count["n"] += 1
        time.sleep(0.2)  # window for the second thread to enter
        _srv._state.model = _FakeModel()
        _srv._state.tokenizer = _FakeTokenizer()
        _srv._state.model_name = "test-model"

    orig = _srv.load_mlx_model
    _srv.load_mlx_model = counting_load
    try:
        _srv._LOAD_MODE = "lazy"
        _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")

        errors: list[BaseException] = []

        def worker():
            try:
                _srv._ensure_loaded_blocking()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Workers raised: {errors}"
        assert load_call_count["n"] == 1, (
            f"Expected exactly one underlying load; saw {load_call_count['n']}"
        )
        assert _srv._LOAD_COMPLETE.is_set()
    finally:
        _srv.load_mlx_model = orig


def test_lazy_load_failure_surfaces_503(client):
    """When the deferred loader raises, /v1/chat/completions returns 503
    with the error message in the detail field, and /model/status reflects
    the error."""
    def boom_load(mlx_model_dir, verbose=False):
        raise RuntimeError("disk on fire")

    orig = _srv.load_mlx_model
    _srv.load_mlx_model = boom_load
    try:
        _srv._LOAD_MODE = "lazy"
        _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")

        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert r.status_code == 503
        assert "disk on fire" in r.json().get("detail", "")

        s = client.get("/model/status").json()
        assert s["model_loaded"] is False
        assert s["load_error"] is not None
        assert "disk on fire" in s["load_error"]
    finally:
        _srv.load_mlx_model = orig


# ═══════════════════════════════════════════════════════════════════════════════
# Preload-async mode — bind first, background load
# ═══════════════════════════════════════════════════════════════════════════════


def test_preload_async_health_during_background_load(client):
    """
    While the background loader is still running, /health remains responsive
    and reports status=ready/model_loaded=False. Once the loader completes,
    /health flips to ok/True without anyone calling _ensure_loaded.
    """
    load_called, cleanup = _install_fast_loader(load_delay_s=0.3)
    try:
        _srv._LOAD_MODE = "preload_async"
        _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")

        # Spawn the background load (this is what main() does for preload_async)
        t = threading.Thread(
            target=_srv._do_model_load, args=(_srv._LOAD_ARGS,),
            name="test-preload", daemon=True,
        )
        t.start()

        # Hit /health immediately — should be responsive (port is "bound")
        # before the load completes.
        r = client.get("/health")
        body = r.json()
        assert r.status_code == 200
        # NB: depending on scheduler this might already be True if the stub
        # ran ahead; the important property is the endpoint is *responsive*
        # without blocking on the load.
        assert body["load_mode"] == "preload_async"

        t.join(timeout=5.0)
        assert not t.is_alive(), "Background load did not finish in time"
        assert load_called.is_set()
        assert _srv._LOAD_COMPLETE.is_set()

        r2 = client.get("/health")
        body2 = r2.json()
        assert body2["model_loaded"] is True
        assert body2["status"] == "ok"
    finally:
        cleanup()


def test_preload_async_first_request_waits_for_background_load():
    """A request that arrives mid-background-load blocks on the lock until
    the background loader finishes, then sees a loaded model."""
    load_called, cleanup = _install_fast_loader(load_delay_s=0.3)
    try:
        _srv._LOAD_MODE = "preload_async"
        _srv._LOAD_ARGS = _FakeArgs(mlx_model_dir="/tmp/fake")

        bg = threading.Thread(
            target=_srv._do_model_load, args=(_srv._LOAD_ARGS,),
            daemon=True,
        )
        bg.start()
        # Immediately have a "request" call _ensure_loaded_blocking from the
        # main thread. Lock contention means we wait for bg to release.
        t0 = time.perf_counter()
        _srv._ensure_loaded_blocking()
        elapsed = time.perf_counter() - t0
        assert _srv._LOAD_COMPLETE.is_set()
        # The request should have waited ≥ load_delay_s (minus a small fudge)
        # because it raced with bg through the same lock.
        assert elapsed >= 0.15, f"Request returned before load finished: {elapsed:.3f}s"
        bg.join(timeout=2.0)
    finally:
        cleanup()
