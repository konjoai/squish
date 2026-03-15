"""tests/benchmarks/test_bench_base.py — Unit tests for squish/benchmarks/base.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import (
    ENGINE_REGISTRY,
    LLAMACPP_ENGINE,
    LMSTUDIO_ENGINE,
    MLXLM_ENGINE,
    OLLAMA_ENGINE,
    SQUISH_ENGINE,
    BenchmarkRunner,
    EngineClient,
    EngineConfig,
    ResultRecord,
    parse_engines,
)


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_chat_url_strips_trailing_slash(self):
        cfg = EngineConfig("x", "http://localhost:8080/")
        assert cfg.chat_url() == "http://localhost:8080/v1/chat/completions"

    def test_chat_url_no_slash(self):
        cfg = EngineConfig("x", "http://localhost:8080")
        assert cfg.chat_url() == "http://localhost:8080/v1/chat/completions"

    def test_health_url(self):
        cfg = EngineConfig("x", "http://localhost:8080")
        assert cfg.health_url() == "http://localhost:8080/health"

    def test_models_url(self):
        cfg = EngineConfig("x", "http://localhost:8080")
        assert cfg.models_url() == "http://localhost:8080/v1/models"

    def test_default_api_key(self):
        cfg = EngineConfig("x", "http://localhost:8080")
        assert cfg.api_key == "squish"

    def test_default_timeout(self):
        cfg = EngineConfig("x", "http://localhost:8080")
        assert cfg.timeout == 120.0

    def test_custom_timeout(self):
        cfg = EngineConfig("x", "http://localhost:8080", timeout=30.0)
        assert cfg.timeout == 30.0


# ---------------------------------------------------------------------------
# ENGINE_REGISTRY
# ---------------------------------------------------------------------------

class TestEngineRegistry:
    def test_all_known_engines_present(self):
        for name in ("squish", "ollama", "lmstudio", "mlxlm", "llamacpp"):
            assert name in ENGINE_REGISTRY

    def test_squish_engine_port(self):
        assert "11434" in SQUISH_ENGINE.base_url

    def test_ollama_engine_port(self):
        assert "11434" in OLLAMA_ENGINE.base_url

    def test_lmstudio_engine_port(self):
        assert "1234" in LMSTUDIO_ENGINE.base_url

    def test_mlxlm_engine_port(self):
        assert "8080" in MLXLM_ENGINE.base_url

    def test_llamacpp_engine_port(self):
        assert "8080" in LLAMACPP_ENGINE.base_url


# ---------------------------------------------------------------------------
# parse_engines
# ---------------------------------------------------------------------------

class TestParseEngines:
    def test_single_known_engine(self):
        result = parse_engines("squish")
        assert len(result) == 1
        assert result[0].name == "squish"

    def test_multiple_known_engines(self):
        result = parse_engines("squish,ollama")
        assert len(result) == 2
        names = [e.name for e in result]
        assert "squish" in names
        assert "ollama" in names

    def test_custom_engine_name_eq_url(self):
        result = parse_engines("myengine=http://custom:9090")
        assert len(result) == 1
        assert result[0].name == "myengine"
        assert result[0].base_url == "http://custom:9090"

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            parse_engines("nonexistent_engine")

    def test_whitespace_trimmed(self):
        result = parse_engines("squish , ollama")
        assert len(result) == 2

    def test_empty_spec_returns_empty_list(self):
        result = parse_engines("")
        assert result == []


# ---------------------------------------------------------------------------
# ResultRecord
# ---------------------------------------------------------------------------

class TestResultRecord:
    def _make_record(self):
        return ResultRecord(
            track="quality",
            engine="squish",
            model="qwen3:8b",
            metrics={"mmlu_acc": 0.75},
            metadata={"limit": 100},
        )

    def test_to_dict_keys(self):
        r = self._make_record()
        d = r.to_dict()
        assert set(d.keys()) == {"track", "engine", "model", "timestamp", "metrics", "metadata"}

    def test_to_dict_values(self):
        r = self._make_record()
        d = r.to_dict()
        assert d["track"] == "quality"
        assert d["engine"] == "squish"
        assert d["model"] == "qwen3:8b"
        assert d["metrics"] == {"mmlu_acc": 0.75}

    def test_timestamp_is_string(self):
        r = self._make_record()
        assert isinstance(r.timestamp, str)
        assert len(r.timestamp) > 0

    def test_save_and_load_roundtrip(self):
        r = self._make_record()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "result.json"
            r.save(p)
            loaded = ResultRecord.load(p)
        assert loaded.track == r.track
        assert loaded.engine == r.engine
        assert loaded.model == r.model
        assert loaded.metrics == r.metrics
        assert loaded.metadata == r.metadata

    def test_save_creates_parent_dirs(self):
        r = self._make_record()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "subdir" / "nested" / "result.json"
            r.save(p)
            assert p.exists()

    def test_load_handles_missing_optional_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "minimal.json"
            p.write_text(json.dumps({
                "track": "perf",
                "engine": "ollama",
                "model": "llama3",
            }))
            r = ResultRecord.load(p)
        assert r.metrics == {}
        assert r.metadata == {}
        assert r.timestamp == ""

    def test_default_metrics_is_empty_dict(self):
        r = ResultRecord(track="t", engine="e", model="m")
        assert r.metrics == {}

    def test_default_metadata_is_empty_dict(self):
        r = ResultRecord(track="t", engine="e", model="m")
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# EngineClient
# ---------------------------------------------------------------------------

class TestEngineClient:
    def _make_client(self, url="http://localhost:8080"):
        return EngineClient(EngineConfig("test", url))

    def test_is_alive_returns_true_on_200(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert client.is_alive() is True

    def test_is_alive_returns_false_on_exception(self):
        client = self._make_client()
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            assert client.is_alive() is False

    def test_chat_returns_parsed_json(self):
        client = self._make_client()
        fake_response = json.dumps({
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"total_tokens": 10},
        }).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = client.chat("mymodel", [{"role": "user", "content": "hi"}])
        assert "choices" in result
        assert "_ttft_s" in result

    def test_chat_raises_connection_error_on_url_error(self):
        client = self._make_client()
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(ConnectionError):
                client.chat("m", [{"role": "user", "content": "hi"}])

    def test_chat_includes_tools_in_payload_when_provided(self):
        client = self._make_client()
        sent_payloads = []

        def capture_request(req, timeout=None):
            sent_payloads.append(json.loads(req.data))
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({"choices": []}).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=capture_request):
            client.chat("m", [], tools=[{"type": "function", "function": {"name": "f"}}])
        assert "tools" in sent_payloads[0]

    def test_chat_omits_tools_when_none(self):
        client = self._make_client()
        sent_payloads = []

        def capture_request(req, timeout=None):
            sent_payloads.append(json.loads(req.data))
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({"choices": []}).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=capture_request):
            client.chat("m", [])
        assert "tools" not in sent_payloads[0]


# ---------------------------------------------------------------------------
# BenchmarkRunner (abstract)
# ---------------------------------------------------------------------------

class TestBenchmarkRunnerAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BenchmarkRunner()  # type: ignore[abstract]

    def test_concrete_subclass_requires_track_name_and_run(self):
        class Incomplete(BenchmarkRunner):
            pass  # no track_name or run
        with pytest.raises(TypeError):
            Incomplete()

    def test_output_path_format(self):
        class Concrete(BenchmarkRunner):
            @property
            def track_name(self):
                return "mytrack"
            def run(self, engine, model, *, limit=None):
                return ResultRecord(track="mytrack", engine=engine.name, model=model)

        runner = Concrete()
        path = runner.output_path("squish", "qwen3:8b", base_dir="out")
        parts = path.name.split("_")
        assert parts[0] == "mytrack"
        assert "squish" in path.name
        assert "qwen3" in path.name
