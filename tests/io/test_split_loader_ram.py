"""Cover _total_ram_bytes() sysctl path via a mocked ctypes.CDLL (host-agnostic)."""

import ctypes

from squish.io import split_loader


def test_total_ram_bytes_reads_sysctl(monkeypatch):
    class _Lib:
        def sysctlbyname(self, *_args):
            return 0  # success; leaves the out-buffer at its initial 0

    monkeypatch.setattr(ctypes, "CDLL", lambda _name: _Lib())
    assert split_loader._total_ram_bytes() == 0
