"""
tests/test_wave74_web_ui.py

Wave 74 — Web UI rename: "Squish Chat" → "Squish Agent"

Asserts that the static index.html:
  1. Has the new <title>Squish Agent</title>
  2. Has the logo text "Squish Agent"
  3. Does NOT contain the old string "Squish Chat" anywhere
"""
import os
import sys
import unittest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HTML_PATH = os.path.join(_repo_root, "squish", "static", "index.html")


class TestWebUIRename(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(_HTML_PATH, encoding="utf-8") as f:
            cls.html = f.read()

    def test_title_is_squish_agent(self):
        """<title> must be 'Squish Agent'."""
        self.assertIn("<title>Squish Agent</title>", self.html)

    def test_logo_says_squish_agent(self):
        """Logo div must contain 'Squish Agent'."""
        self.assertIn("<span>Squish</span> Agent", self.html)

    def test_no_squish_chat_anywhere(self):
        """The old 'Squish Chat' text must not appear anywhere in the HTML."""
        self.assertNotIn("Squish Chat", self.html)


if __name__ == "__main__":
    unittest.main()
