"""tests/e2e/conftest.py

Session-scoped harness that boots a **real** squish inference server against a
small MLX model and tears it down at the end of the session.  Everything under
``tests/e2e/`` that needs a live server depends on the ``live_server`` fixture.

Gating
──────
The whole suite is marked ``@pytest.mark.e2e`` (see each test module) and is
skipped by ``tests/conftest.py`` unless ``SQUISH_E2E=1`` or ``--run-e2e`` is
passed.  The ``live_server`` fixture adds a second guard: it skips when the host
is not Apple Silicon or ``import mlx.core`` fails, so ``--run-e2e`` on Linux
degrades gracefully instead of hanging on a server that can never load.

Model
─────
Default ``mlx-community/Qwen2.5-0.5B-Instruct-4bit`` (~300 MB, fast).  Override
with ``--model`` / ``SQUISH_E2E_MODEL`` — bump to the 1.5B variant when the
0.5B model is too weak for agent tool-calling.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

# ── Prerequisites ────────────────────────────────────────────────────────────
_MLX_AVAILABLE = False
if sys.platform == "darwin":
    try:
        import mlx.core as _mx

        _mx.array([0])  # force Metal initialisation
        _MLX_AVAILABLE = True
    except Exception:  # noqa: BLE001 — any MLX/Metal failure means "unavailable"
        _MLX_AVAILABLE = False

_APPLE_SILICON = sys.platform == "darwin" and _MLX_AVAILABLE

_DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
_HEALTH_TIMEOUT_S = 180.0
_HEALTH_POLL_INTERVAL_S = 1.0


@dataclass
class ServerHandle:
    """Connection details for a booted squish server."""

    url: str
    api_key: str


def _free_port() -> int:
    """Bind ``:0`` to grab a free TCP port, then release it for the server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_model(request: pytest.FixtureRequest) -> str:
    opt = request.config.getoption("--model", default=None)
    return opt or os.environ.get("SQUISH_E2E_MODEL") or _DEFAULT_MODEL


def _ensure_local_model(spec: str) -> str:
    """Resolve *spec* to a local model directory ``squish serve`` can load.

    ``squish serve`` only loads a catalogued model or an on-disk path — a bare
    HuggingFace id (e.g. ``mlx-community/...``) is reported "Unknown model".

    For an HF id we (1) pre-pull via ``squish pull hf:<id>`` so the mandatory
    pre-download safety scan runs (CLAUDE.md), then (2) resolve the concrete
    on-disk snapshot path with ``huggingface_hub.snapshot_download`` (idempotent
    — a cache hit after the pull).  Resolving via the snapshot path is robust
    across environments: a bare ``squish pull`` does **not** always materialise a
    ``models/<basename>`` copy (e.g. on a CI runner whose HF cache was restored).
    An existing path/dir is used as-is.
    """
    if Path(spec).exists():
        return spec

    basename = spec.split("/")[-1]
    target = Path("models") / basename
    if target.exists():
        return str(target)

    # Pre-scan + fetch (best-effort: snapshot resolution below is the source of
    # truth, so a non-zero pull exit must not abort the whole session).
    try:
        subprocess.run(  # noqa: S603 — fixed interpreter, no shell
            [sys.executable, "-m", "squish", "pull", f"hf:{spec}"],
            check=False,
            timeout=900,
            env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        )
    except subprocess.TimeoutExpired:
        pass  # fall through to snapshot_download (may already be cached)

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        return snapshot_download(repo_id=spec)
    except Exception as exc:  # noqa: BLE001 — fail-fast with a clear message
        pytest.fail(f"could not resolve model {spec!r} to a local path: {exc}")


def _poll_health(url: str, api_key: str, log_path: str) -> None:
    """Block until ``/health`` reports ``loaded == true`` or time out.

    Raises ``pytest.fail`` (via ``pytest.fail``) with the captured server log on
    timeout so a boot failure is diagnosable rather than a bare assertion.
    """
    deadline = time.time() + _HEALTH_TIMEOUT_S
    last_err = "no response"
    req = urllib.request.Request(  # noqa: S310 — fixed localhost URL
        f"{url}/health",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode())
            if payload.get("loaded") is True:
                return
            last_err = f"health={payload.get('status')!r} loaded={payload.get('loaded')!r}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(_HEALTH_POLL_INTERVAL_S)

    server_log = "<no log captured>"
    with contextlib.suppress(OSError):
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            server_log = fh.read()[-8000:]
    pytest.fail(
        f"squish server did not become healthy within {_HEALTH_TIMEOUT_S:.0f}s "
        f"(last: {last_err}).\n──── server log tail ────\n{server_log}"
    )


@pytest.fixture(scope="session")
def live_server(request, tmp_path_factory) -> ServerHandle:
    """Boot a real squish server once per session; yield its URL + key."""
    if not _APPLE_SILICON:
        pytest.skip("e2e requires Apple Silicon + importable mlx.core")

    model = _ensure_local_model(_resolve_model(request))
    port = _free_port()
    api_key = os.environ.get("SQUISH_E2E_KEY", "squish-e2e-test-key")
    url = f"http://127.0.0.1:{port}"

    log_dir = tmp_path_factory.mktemp("squish-e2e")
    log_path = str(log_dir / "server.log")

    cmd = [
        sys.executable,
        "-m",
        "squish",
        "serve",
        model,
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--log-level",
        "warning",
    ]
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        )

    try:
        # Fail fast if the process dies during startup.
        time.sleep(0.5)
        if proc.poll() is not None:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                pytest.fail(
                    f"squish server exited early (code {proc.returncode}).\n"
                    f"──── server log ────\n{fh.read()[-8000:]}"
                )
        _poll_health(url, api_key, log_path)

        # Hand off to the agent-e2e helpers which read these env vars.
        os.environ["SQUISH_E2E_URL"] = url
        os.environ["SQUISH_E2E_KEY"] = api_key
        os.environ.setdefault("SQUISH_E2E", "1")

        yield ServerHandle(url=url, api_key=api_key)
    finally:
        _terminate(proc)


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM → wait → SIGKILL fallback."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


@pytest.fixture(scope="session")
def server_url(live_server: ServerHandle) -> str:
    return live_server.url


@pytest.fixture(scope="session")
def api_key(live_server: ServerHandle) -> str:
    return live_server.api_key
