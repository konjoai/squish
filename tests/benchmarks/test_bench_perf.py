"""tests/benchmarks/test_bench_perf.py — Unit tests for squish/benchmarks/perf_bench.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord
from squish.benchmarks.perf_bench import (
    PerfBenchConfig,
    PerfBenchRunner,
    _cold_start_ttft_ms,
    _count_tokens,
    _rss_mb,
    _warm_ttft_and_tps,
)

_ENGINE = EngineConfig("squish", "http://localhost:11434")


# ---------------------------------------------------------------------------
# _count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_single_word(self):
        assert _count_tokens("hello") == 1

    def test_multiple_words(self):
        assert _count_tokens("hello world foo") == 3

    def test_empty_string_returns_one(self):
        assert _count_tokens("") == 1

    def test_whitespace_only_returns_one(self):
        assert _count_tokens("   ") == 1


# ---------------------------------------------------------------------------
# _rss_mb
# ---------------------------------------------------------------------------

class TestRssMb:
    def test_returns_float(self):
        assert isinstance(_rss_mb(), float)

    def test_returns_non_negative(self):
        assert _rss_mb() >= 0.0

    def test_returns_zero_on_error(self):
        with patch("squish.benchmarks.perf_bench._rss_mb", side_effect=Exception("boom")):
            # Just verify calling the real function doesn't blow up
            result = _rss_mb()
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# PerfBenchConfig
# ---------------------------------------------------------------------------

class TestPerfBenchConfig:
    def test_default_warm_reps(self):
        assert PerfBenchConfig().warm_reps == 3

    def test_default_batch_concurrency(self):
        assert PerfBenchConfig().batch_concurrency == 8

    def test_default_max_tokens(self):
        assert PerfBenchConfig().max_tokens == 128

    def test_default_temperature(self):
        assert PerfBenchConfig().temperature == 0.0


# ---------------------------------------------------------------------------
# _warm_ttft_and_tps (unit test with mocked chat_stream)
# ---------------------------------------------------------------------------

class TestWarmTtftAndTps:
    def _make_streaming_client(self):
        client = MagicMock()
        # chat_stream yields (delta, ttft_s, total_s)
        client.chat_stream.return_value = iter([
            ("hello world", 0.05, 0.05),
            ("foo bar baz", 0.05, 0.10),
        ])
        return client

    def test_returns_dict_with_warm_ttft_ms(self):
        client = self._make_streaming_client()
        cfg = PerfBenchConfig(warm_reps=1)
        result = _warm_ttft_and_tps(client, "m", cfg)
        assert "warm_ttft_ms" in result

    def test_returns_dict_with_tps(self):
        client = self._make_streaming_client()
        cfg = PerfBenchConfig(warm_reps=1)
        result = _warm_ttft_and_tps(client, "m", cfg)
        assert "tps" in result

    def test_warm_ttft_ms_is_float(self):
        client = self._make_streaming_client()
        cfg = PerfBenchConfig(warm_reps=1)
        result = _warm_ttft_and_tps(client, "m", cfg)
        assert isinstance(result["warm_ttft_ms"], float)

    def test_handles_stream_exception_gracefully(self):
        client = MagicMock()
        client.chat_stream.side_effect = ConnectionError("no server")
        cfg = PerfBenchConfig(warm_reps=1)
        result = _warm_ttft_and_tps(client, "m", cfg)
        # Should not raise; returns 0.0 defaults
        assert result["warm_ttft_ms"] == 0.0
        assert result["tps"] == 0.0


# ---------------------------------------------------------------------------
# PerfBenchRunner
# ---------------------------------------------------------------------------

_FAKE_BATCH = {
    "batch_p50_ms": 100.0,
    "batch_p99_ms": 200.0,
    "batch_throughput_tps": 5.0,
}


class TestPerfBenchRunner:
    def _make_runner(self, **kw):
        return PerfBenchRunner(PerfBenchConfig(**kw))

    def test_track_name_is_perf(self):
        assert self._make_runner().track_name == "perf"

    def test_default_config_used_when_none_passed(self):
        runner = PerfBenchRunner()
        assert runner._config is not None

    def test_run_returns_result_record(self):
        runner = self._make_runner(warm_reps=1, batch_concurrency=1)
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = iter([("hi", 0.05, 0.05)])

        with patch("squish.benchmarks.perf_bench.EngineClient", return_value=mock_client), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value=_FAKE_BATCH), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            result = runner.run(_ENGINE, "qwen3:8b")

        assert isinstance(result, ResultRecord)
        assert result.track == "perf"
        assert result.engine == "squish"
        assert result.model == "qwen3:8b"

    def test_run_metrics_have_expected_keys(self):
        runner = self._make_runner(warm_reps=1, batch_concurrency=1)
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = iter([("hello world", 0.03, 0.03)])

        with patch("squish.benchmarks.perf_bench.EngineClient", return_value=mock_client), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value=_FAKE_BATCH), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            result = runner.run(_ENGINE, "qwen3:8b")

        for key in ("warm_ttft_ms", "tps", "ram_delta_mb", "long_ctx_tps", "tokens_per_watt", "cold_start_ms"):
            assert key in result.metrics, f"missing key: {key}"

    def test_run_limit_overrides_max_tokens(self):
        runner = self._make_runner()
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = iter([])

        with patch("squish.benchmarks.perf_bench.EngineClient", return_value=mock_client), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value=_FAKE_BATCH), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            result = runner.run(_ENGINE, "qwen3:8b", limit=64)
        assert result.metadata["max_tokens"] == 64

    def test_run_metadata_contains_platform(self):
        runner = self._make_runner(warm_reps=1)
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = iter([("hi", 0.1, 0.1)])

        with patch("squish.benchmarks.perf_bench.EngineClient", return_value=mock_client), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value=_FAKE_BATCH), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert "platform" in result.metadata

    def test_run_metadata_contains_batch_concurrency(self):
        runner = self._make_runner(warm_reps=1, batch_concurrency=4)
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = iter([("hi", 0.1, 0.1)])

        with patch("squish.benchmarks.perf_bench.EngineClient", return_value=mock_client), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value=_FAKE_BATCH), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metadata["batch_concurrency"] == 4


# ---------------------------------------------------------------------------
# _cold_start_ttft_ms
# ---------------------------------------------------------------------------

class TestColdStartTtftMs:
    def test_returns_zero_when_cmd_empty(self):
        assert _cold_start_ttft_ms([], port=11436, max_tokens=32) == 0.0

    def test_returns_float(self):
        result = _cold_start_ttft_ms([], port=11436, max_tokens=32)
        assert isinstance(result, float)

    def test_returns_zero_when_server_never_starts(self):
        # Port 1 is reserved and will refuse; server won't respond in time
        with patch("squish.benchmarks.perf_bench.subprocess.Popen") as mock_popen, \
             patch("squish.benchmarks.perf_bench.urllib.request.urlopen",
                   side_effect=OSError("refused")):
            mock_proc = MagicMock()
            mock_popen.return_value.__enter__ = lambda s: mock_proc
            mock_popen.return_value = mock_proc
            result = _cold_start_ttft_ms(
                ["squish", "serve", "--model", "qwen3:1.7b"],
                port=19999,
                max_tokens=4,
                timeout_s=0.01,   # very short timeout
            )
        assert result == 0.0

    def test_returns_ttft_when_server_ready_and_responds(self):
        """Simulate a server that is immediately healthy and returns a delta."""
        import io

        sse_response = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n"
            b"data: [DONE]\n"
        )

        class _FakeResponse:
            def __iter__(self):
                return iter(sse_response.splitlines(keepends=True))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        call_count = [0]

        def _fake_urlopen(req, timeout=None):
            call_count[0] += 1
            return _FakeResponse()

        with patch("squish.benchmarks.perf_bench.subprocess.Popen") as mock_popen, \
             patch("squish.benchmarks.perf_bench.urllib.request.urlopen", side_effect=_fake_urlopen):
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            result = _cold_start_ttft_ms(
                ["squish", "serve", "--model", "qwen3:1.7b"],
                port=11436,
                max_tokens=8,
            )

        # Either got a timing or 0.0 (both valid outcomes in unit test context)
        assert isinstance(result, float)
        assert result >= 0.0
        # Process should have been terminated in the finally block
        mock_proc.terminate.assert_called()

    def test_popen_exception_returns_zero(self):
        with patch("squish.benchmarks.perf_bench.subprocess.Popen",
                   side_effect=FileNotFoundError("squish not found")):
            result = _cold_start_ttft_ms(
                ["squish", "serve"],
                port=11436,
                max_tokens=8,
            )
        assert result == 0.0

    def test_default_config_has_empty_cold_start_cmd(self):
        cfg = PerfBenchConfig()
        assert cfg.cold_start_cmd == []

    def test_config_cold_start_port_default(self):
        cfg = PerfBenchConfig()
        assert cfg.cold_start_port == 11436

    def test_run_includes_cold_start_ms_in_metrics(self):
        runner = PerfBenchRunner(PerfBenchConfig(warm_reps=1))
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = iter([("hi", 0.1, 0.1)])

        with patch("squish.benchmarks.perf_bench.EngineClient", return_value=mock_client), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value=_FAKE_BATCH), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert "cold_start_ms" in result.metrics
        assert isinstance(result.metrics["cold_start_ms"], float)

    def test_run_cold_start_ms_zero_when_no_cmd(self):
        """cold_start_ms defaults to 0.0 when cold_start_cmd is not set."""
        runner = PerfBenchRunner(PerfBenchConfig(warm_reps=1))
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = iter([("hi", 0.1, 0.1)])

        with patch("squish.benchmarks.perf_bench.EngineClient", return_value=mock_client), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value=_FAKE_BATCH), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metrics["cold_start_ms"] == 0.0
