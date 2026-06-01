"""squish/daemon/client.py — thin Unix-socket client for squishd.

Usage
-----
    from squish.daemon.client import DaemonClient

    client = DaemonClient()
    if client.available():
        result = client.chat(messages=[{"role": "user", "content": "Hello"}])
        print(result["text"])
    else:
        # Fall back to direct inference
        ...

The client speaks the squishd wire protocol (length-prefixed JSON frames).
It never imports any ML framework — the dependency is just the stdlib socket.
"""
from __future__ import annotations

import time
from typing import Any

from squish.daemon.squishd import (
    SOCK_PATH,
    _recv_frame,
    _send_frame,
    is_running,
    send_request,
)

import socket


class DaemonClient:
    """Thin client for squishd.

    Parameters
    ----------
    sock_path : str
        Path to the Unix domain socket.
    timeout : float
        Per-request socket timeout in seconds.
    model_dir : str
        Default model directory sent with each request.
    """

    def __init__(
        self,
        sock_path: str = SOCK_PATH,
        timeout: float = 120.0,
        model_dir: str = "",
        compressed_dir: str = "",
    ) -> None:
        self._sock_path     = sock_path
        self._timeout       = timeout
        self._model_dir     = model_dir
        self._compressed_dir = compressed_dir

    def available(self) -> bool:
        """Return True if squishd is reachable."""
        return is_running(self._sock_path)

    def ping(self) -> dict[str, Any]:
        """Send a ping and return the status dict."""
        return send_request({"_cmd": "ping"}, self._sock_path, self._timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model_dir: str = "",
        compressed_dir: str = "",
    ) -> dict[str, Any]:
        """Send a chat request; return the response dict.

        Returns
        -------
        dict with keys: text, tokens, tok_s, finish
        """
        payload: dict[str, Any] = {
            "messages":   messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p":      top_p,
        }
        mdir = model_dir or self._model_dir
        cdir = compressed_dir or self._compressed_dir
        if mdir:
            payload["model_dir"]      = mdir
        if cdir:
            payload["compressed_dir"] = cdir

        return send_request(payload, self._sock_path, self._timeout)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        """Send a plain-text completion request."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def ttft(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1,
    ) -> float:
        """Measure time-to-first-token in seconds (generates 1 token)."""
        t0 = time.perf_counter()
        self.chat(messages, max_tokens=max_tokens)
        return time.perf_counter() - t0

    def wall_latency(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 64,
        n_runs: int = 3,
    ) -> dict[str, float]:
        """Measure wall latency for *n_runs* identical requests.

        Returns dict: p50, p95, p99, mean (all in seconds).
        """
        import statistics
        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.chat(messages, max_tokens=max_tokens)
            times.append(time.perf_counter() - t0)
        times.sort()

        def _pct(p: float) -> float:
            idx = (p / 100.0) * (len(times) - 1)
            lo, hi = int(idx), min(int(idx) + 1, len(times) - 1)
            frac = idx - lo
            return times[lo] * (1 - frac) + times[hi] * frac

        return {
            "p50":  _pct(50),
            "p95":  _pct(95),
            "p99":  _pct(99),
            "mean": statistics.mean(times),
        }
