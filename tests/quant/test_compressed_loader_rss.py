"""Cover _rss_mb()'s Darwin branch by patching the platform probe (host-agnostic)."""

from squish.quant import compressed_loader as cl


def test_rss_mb_darwin_branch(monkeypatch):
    monkeypatch.setattr(cl._platform, "system", lambda: "Darwin")
    assert cl._rss_mb() >= 0.0
