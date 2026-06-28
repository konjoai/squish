"""Host introspection — RAM size and tokenizer loading.

The RAM parser is pure (unit-tested); the tokenizer loader is MLX-bound and only
runs on the bench host.
"""

from __future__ import annotations

import subprocess


def parse_sysctl_memsize(text: str) -> int | None:
    """Parse `sysctl hw.memsize` output ('hw.memsize: 17179869184') to bytes."""
    for tok in text.replace(":", " ").split():
        try:
            v = int(tok)
            if v > 1 << 20:  # ignore the small key fragments
                return v
        except ValueError:
            continue
    return None


def detect_ram_bytes(default: int = 16 * 1024**3) -> int:
    """Total physical RAM in bytes (psutil -> sysctl -> default)."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        out = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True, timeout=10)
        v = parse_sysctl_memsize(out.stdout)
        if v:
            return v
    except (OSError, subprocess.TimeoutExpired):
        pass
    return default


class MLXTokenizer:
    """Thin adapter exposing encode/decode for the corpus from an mlx_lm tokenizer."""

    def __init__(self, model_dir: str) -> None:
        from mlx_lm import load  # pragma: no cover - MLX only on the bench host

        _, self._tok = load(model_dir)

    def encode(self, text: str) -> list[int]:  # pragma: no cover - MLX only
        return list(self._tok.encode(text))

    def decode(self, ids: list[int]) -> str:  # pragma: no cover - MLX only
        return self._tok.decode(ids)
