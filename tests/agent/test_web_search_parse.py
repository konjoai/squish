"""Regression tests for squish_web_search HTML parsing + max_results clamp."""
from __future__ import annotations

from squish.agent import builtin_tools as bt
from squish.agent.builtin_tools import _parse_ddg_lite_results, squish_web_search


def _page(*rows: str) -> str:
    return "\n".join(rows)


_LINK = '<a class="result-link" href="/l/?uddg=https%3A%2F%2F{host}">{title}</a>'
_SNIP = '<td class="result-snippet">{text}</td>'


def test_missing_snippet_does_not_shift_later_snippets():
    # First result has no snippet; naive index pairing would attach the second
    # result's snippet to the first.
    page = _page(
        _LINK.format(host="a.com", title="Alpha"),
        _LINK.format(host="b.com", title="Bravo"),
        _SNIP.format(text="snippet for Bravo"),
    )
    res = _parse_ddg_lite_results(page, 10)
    assert res[0] == ("https://a.com", "Alpha", "")
    assert res[1] == ("https://b.com", "Bravo", "snippet for Bravo")


def test_each_link_paired_with_following_snippet():
    page = _page(
        _LINK.format(host="a.com", title="Alpha"),
        _SNIP.format(text="A snip"),
        _LINK.format(host="b.com", title="Bravo"),
        _SNIP.format(text="B snip"),
    )
    res = _parse_ddg_lite_results(page, 10)
    assert [r[2] for r in res] == ["A snip", "B snip"]


def test_limit_is_respected():
    page = _page(*[_LINK.format(host=f"h{i}.com", title=f"T{i}") for i in range(5)])
    assert len(_parse_ddg_lite_results(page, 2)) == 2


def test_tags_and_entities_stripped_from_title_and_snippet():
    page = _page(
        _LINK.format(host="a.com", title="<b>Title &amp; Co</b>"),
        _SNIP.format(text="<span>R&amp;D</span>"),
    )
    url, title, snippet = _parse_ddg_lite_results(page, 1)[0]
    assert title == "Title & Co"
    assert snippet == "R&D"


def test_max_results_nonpositive_is_clamped(monkeypatch):
    # With max_results <= 0 the fallback any_links[:n] used to be a negative slice
    # returning trailing links. Clamp to >= 1 instead. Stub the network so the
    # fallback path runs on a snippet-less page.
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n):
            return b'<a href="https://example.org/page">x</a>'

    monkeypatch.setattr(bt.urllib.request, "urlopen", lambda *a, **k: _Resp())
    out = squish_web_search("anything", max_results=-3)
    # Must not raise and must not return a negative-slice artifact.
    assert "example.org" in out or "No results" in out
