"""tests/test_demo_server.py — W109 demo server endpoint tests.

Coverage focus:
  - `GET /api/recommend?model_size_b=&ctx_len=[&budget_mb=]` — the W109
    addition. Uses a real HTTPServer on a free port; the closed-form
    planner is fast enough to keep the test under a second.
  - Architecture-snapping helper `_arch_for_model_size_b` invariants.
  - Reasoning-string content gates: must mention the binding constraint
    (context vs budget), must include the chosen tier, must NOT lie
    about the budget when the budget actually decided.
  - Backward compat: existing POST /api/recommend, /api/health, and the
    static index path still respond as before.

Security gates (CLAUDE.md security.md):
  - Required params produce 400 not 500.
  - Out-of-range params produce 400 not 500.
  - Non-numeric params produce 400 not 500.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from demo.server import (
    Handler,
    _arch_for_model_size_b,
    _build_recommendation,
    _REC_ARCH_TABLE,
)
from http.server import HTTPServer


# ---------------------------------------------------------------------------
# 1. _arch_for_model_size_b — pure function, no server needed
# ---------------------------------------------------------------------------

def test_arch_table_is_sorted_by_size_ascending():
    sizes = [row[0] for row in _REC_ARCH_TABLE]
    assert sizes == sorted(sizes)


def test_arch_table_rows_are_well_formed():
    for size_b, n_layers, n_kv_heads, head_dim, label in _REC_ARCH_TABLE:
        assert size_b > 0
        assert n_layers > 0
        assert n_kv_heads > 0
        assert head_dim > 0 and head_dim % 4 == 0   # all 3 codecs valid
        assert isinstance(label, str) and label


@pytest.mark.parametrize("requested,expected_label", [
    (0.4, "Qwen2.5-0.5B"),
    (0.5, "Qwen2.5-0.5B"),
    (0.6, "Qwen2.5-0.5B"),
    (1.5, "Qwen2.5-1.5B"),
    (3.0, "Qwen2.5-3B"),
    (7.0, "Qwen2.5-7B"),
    (50,  "Qwen2.5-32B"),
    (100, "Llama-3.1-70B"),
])
def test_arch_snap_picks_closest_preset(requested, expected_label):
    assert _arch_for_model_size_b(requested)["preset_label"] == expected_label


def test_arch_snap_ties_break_to_larger_preset():
    """Halfway between 7B and 8B → 8B (the conservative pick for memory)."""
    arch = _arch_for_model_size_b(7.5)
    assert arch["preset_label"] == "Llama-3.1-8B"


# ---------------------------------------------------------------------------
# 2. _build_recommendation — closed-form path
# ---------------------------------------------------------------------------

def test_recommend_short_context_picks_int8():
    r = _build_recommendation(7.0, 4000, None)
    assert r["mode"] == "int8"
    assert r["basis"] == "context"


def test_recommend_medium_context_picks_int4():
    r = _build_recommendation(7.0, 12000, None)
    assert r["mode"] == "int4"
    assert r["basis"] == "context"


def test_recommend_long_context_picks_int2():
    r = _build_recommendation(7.0, 32000, None)
    assert r["mode"] == "int2"
    assert r["basis"] == "context"


def test_recommend_tight_budget_returns_none_with_explanation():
    """100 MB cannot hold a 7B 32K cache even at INT2 (~265 MB)."""
    r = _build_recommendation(7.0, 32000, 100.0)
    assert r["mode"] == "none"
    assert r["basis"] == "budget"
    assert "exceeds" in r["reason"].lower()


def test_recommend_generous_budget_overrides_context_pick():
    """Budget allows higher-quality tier than context alone would prescribe."""
    r = _build_recommendation(7.0, 32000, 4096.0)
    assert r["mode"] in {"int8", "int4"}     # both fit, INT8 wins
    assert r["mode"] == "int8"
    assert r["basis"] == "budget"
    assert r["by_context"] == "int2"
    assert r["by_budget"] == "int8"


def test_recommend_basis_is_agreement_when_both_pick_same_tier():
    """Pure-context int4 + budget tight enough that int4 is the highest tier
    that fits → both pick int4 and basis='agreement'."""
    # 7B 12K: int8 ≈ 362 MB, int4 ≈ 190 MB. A 250 MB budget rules out int8
    # but fits int4 — both context (int4) and budget (int4) agree.
    r = _build_recommendation(7.0, 12000, 250.0)
    assert r["by_context"] == "int4"
    assert r["by_budget"] == "int4"
    assert r["mode"] == "int4"
    assert r["basis"] == "agreement"


def test_recommend_memory_dict_includes_all_four_tiers():
    r = _build_recommendation(7.0, 8000, None)
    assert set(r["memory_mb"].keys()) == {"fp16", "int8", "int4", "int2"}
    # Monotone: fp16 > int8 > int4 > int2
    m = r["memory_mb"]
    assert m["fp16"] > m["int8"] > m["int4"] > m["int2"]


def test_recommend_reason_mentions_chosen_tier():
    r = _build_recommendation(7.0, 12000, None)
    assert "INT4" in r["reason"]


def test_recommend_reason_mentions_budget_when_budget_decides():
    r = _build_recommendation(7.0, 32000, 4096.0)
    assert "budget" in r["reason"].lower()
    assert "context alone" in r["reason"].lower()


def test_recommend_thresholds_match_kv_cache_constants():
    from squish.kv.kv_cache import KV_INT2_AUTO_THRESHOLD, KV_INT4_DEFAULT_THRESHOLD
    r = _build_recommendation(1.5, 4096, None)
    assert r["thresholds"]["int4_above"] == KV_INT2_AUTO_THRESHOLD
    assert r["thresholds"]["int2_above"] == KV_INT4_DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# 3. HTTP server fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server():
    """Start a real demo HTTPServer on a free port; tear down after tests."""
    sock = socket.socket(); sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]; sock.close()
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    httpd.timeout = 1
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    # Wait for the listening socket to actually accept.
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5).read()
            break
        except OSError:
            time.sleep(0.05)
    else:
        httpd.shutdown(); pytest.fail("demo server did not come up")
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    thread.join(timeout=2)


def _get(server: str, path: str) -> tuple[int, dict]:
    try:
        resp = urllib.request.urlopen(f"{server}{path}", timeout=5)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(server: str, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{server}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# 4. GET /api/recommend — happy paths
# ---------------------------------------------------------------------------

def test_get_recommend_short_context(server):
    code, payload = _get(server, "/api/recommend?model_size_b=7&ctx_len=4000")
    assert code == 200
    assert payload["mode"] == "int8"
    assert payload["basis"] == "context"
    assert payload["model_size_b"] == 7.0
    assert payload["ctx_len"] == 4000
    assert payload["live"] is True


def test_get_recommend_with_budget(server):
    code, payload = _get(
        server, "/api/recommend?model_size_b=7&ctx_len=32000&budget_mb=4096"
    )
    assert code == 200
    assert payload["mode"] == "int8"
    assert payload["basis"] == "budget"
    assert payload["by_context"] == "int2"


def test_get_recommend_returns_none_when_budget_too_tight(server):
    code, payload = _get(
        server, "/api/recommend?model_size_b=70&ctx_len=32000&budget_mb=500"
    )
    assert code == 200
    assert payload["mode"] == "none"


def test_get_recommend_includes_arch_block(server):
    code, payload = _get(server, "/api/recommend?model_size_b=1.5&ctx_len=4000")
    assert code == 200
    assert payload["arch"]["preset_label"] == "Qwen2.5-1.5B"
    assert payload["arch"]["n_layers"] == 28
    assert payload["arch"]["head_dim"] == 128


# ---------------------------------------------------------------------------
# 5. GET /api/recommend — input validation (CLAUDE.md security.md)
# ---------------------------------------------------------------------------

def test_get_recommend_missing_model_size_b_returns_400(server):
    code, payload = _get(server, "/api/recommend?ctx_len=4000")
    assert code == 400
    assert "model_size_b" in payload["error"]


def test_get_recommend_missing_ctx_len_returns_400(server):
    code, payload = _get(server, "/api/recommend?model_size_b=7")
    assert code == 400
    assert "ctx_len" in payload["error"]


def test_get_recommend_non_numeric_model_size_returns_400(server):
    code, payload = _get(server, "/api/recommend?model_size_b=abc&ctx_len=4000")
    assert code == 400
    assert "model_size_b" in payload["error"]


def test_get_recommend_negative_ctx_len_returns_400(server):
    code, payload = _get(server, "/api/recommend?model_size_b=7&ctx_len=-1")
    assert code == 400
    assert "ctx_len" in payload["error"]


def test_get_recommend_huge_ctx_len_returns_400(server):
    code, payload = _get(
        server, "/api/recommend?model_size_b=7&ctx_len=999999999"
    )
    assert code == 400


def test_get_recommend_huge_model_size_returns_400(server):
    code, payload = _get(server, "/api/recommend?model_size_b=999&ctx_len=4000")
    assert code == 400


def test_get_recommend_negative_budget_returns_400(server):
    code, payload = _get(
        server, "/api/recommend?model_size_b=7&ctx_len=4000&budget_mb=-5"
    )
    assert code == 400


# ---------------------------------------------------------------------------
# 6. Backward compatibility — existing endpoints still work
# ---------------------------------------------------------------------------

def test_health_endpoint_unchanged(server):
    code, payload = _get(server, "/api/health")
    assert code == 200
    assert payload["status"] == "ok"
    assert payload["service"] == "squish-demo"


def test_post_recommend_still_works(server):
    code, payload = _post(server, "/api/recommend", {"ctx_len": 12000})
    assert code == 200
    assert payload["mode"] == "int4"


def test_unknown_get_path_returns_404(server):
    code, payload = _get(server, "/api/does-not-exist")
    assert code == 404
