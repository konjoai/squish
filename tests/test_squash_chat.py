"""Unit tests for squish.squash.chat (Wave 56).

Test taxonomy: pure unit — no real LLM endpoint; mocks urllib.request.urlopen.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestArtifactChunker(unittest.TestCase):
    """Tests for ArtifactChunker document loading and chunking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_bom(self, path: Path) -> None:
        bom = {
            "bomFormat": "CycloneDX",
            "serialNumber": "urn:uuid:test-123",
            "components": [
                {
                    "name": "weights.safetensors",
                    "type": "ml-model",
                    "hashes": [{"alg": "SHA-256", "content": "aa" * 32}],
                    "vulnerabilities": [{"id": "CVE-2024-0001", "severity": "high"}],
                }
            ],
        }
        path.write_text(json.dumps(bom))

    def test_load_returns_chunks_from_bom(self):
        from squish.squash.chat import ArtifactChunker
        bom_path = Path(self.tmpdir) / "cyclonedx-mlbom.json"
        self._write_bom(bom_path)
        chunks = ArtifactChunker.load_model_dir(Path(self.tmpdir))
        self.assertGreater(len(chunks), 0)

    def test_chunk_source_file_matches_filename(self):
        from squish.squash.chat import ArtifactChunker
        bom_path = Path(self.tmpdir) / "cyclonedx-mlbom.json"
        self._write_bom(bom_path)
        chunks = ArtifactChunker.load_model_dir(Path(self.tmpdir))
        sources = {c.source_file for c in chunks}
        self.assertIn("cyclonedx-mlbom.json", sources)

    def test_empty_dir_returns_no_chunks(self):
        from squish.squash.chat import ArtifactChunker
        chunks = ArtifactChunker.load_model_dir(Path(self.tmpdir))
        self.assertEqual(0, len(chunks))

    def test_chunk_text_contains_component_fields(self):
        from squish.squash.chat import ArtifactChunker
        bom_path = Path(self.tmpdir) / "cyclonedx-mlbom.json"
        self._write_bom(bom_path)
        chunks = ArtifactChunker.load_model_dir(Path(self.tmpdir))
        all_text = " ".join(c.text for c in chunks)
        self.assertIn("weights.safetensors", all_text)


class TestBM25Retriever(unittest.TestCase):
    """Tests for BM25Retriever keyword scoring."""

    def _make_chunks(self):
        from squish.squash.chat import Chunk
        return [
            Chunk("bom.json", "components[0]", "CVE-2024-0001 critical vulnerability in tokenizer"),
            Chunk("bom.json", "components[1]", "Apache License 2.0 model weights sha256"),
            Chunk("policy.json", "findings[0]", "policy violation: missing dataset provenance record"),
            Chunk("scan.json", "scan_findings[0]", "deserialization risk in pickle file"),
        ]

    def test_retrieves_most_relevant_chunk(self):
        from squish.squash.chat import BM25Retriever
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("CVE vulnerability", k=1)
        self.assertEqual(1, len(results))
        self.assertIn("CVE", results[0].text)

    def test_returns_empty_for_no_match(self):
        from squish.squash.chat import BM25Retriever
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        # Query with terms that don't appear in any chunk
        results = retriever.retrieve("xyzzy quark frobnitz", k=3)
        # With zero BM25 score, retrieve returns nothing
        self.assertEqual(0, len(results))

    def test_empty_corpus_returns_empty(self):
        from squish.squash.chat import BM25Retriever
        retriever = BM25Retriever([])
        results = retriever.retrieve("anything", k=5)
        self.assertEqual([], results)

    def test_k_limits_results(self):
        from squish.squash.chat import BM25Retriever
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("model", k=2)
        self.assertLessEqual(len(results), 2)


class TestChatAnswer(unittest.TestCase):
    """Tests for ChatAnswer str() formatting."""

    def test_str_contains_sources_block(self):
        from squish.squash.chat import ChatAnswer
        answer = ChatAnswer(
            question="Is there a CVE?",
            text="Yes, CVE-2024-0001 is present.",
            sources=["cyclonedx-mlbom.json:components[0]"],
            model_name="llama3",
            backend="http://localhost/v1",
        )
        rendered = str(answer)
        self.assertIn("Sources:", rendered)
        self.assertIn("cyclonedx-mlbom.json", rendered)

    def test_str_no_sources(self):
        from squish.squash.chat import ChatAnswer
        answer = ChatAnswer(
            question="q",
            text="answer",
            sources=[],
            model_name="llama3",
            backend="http://localhost/v1",
        )
        rendered = str(answer)
        self.assertIn("(no sources)", rendered)


class TestChatSessionMocked(unittest.TestCase):
    """Tests for ChatSession.ask() with mocked LLM backend."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_bom(self):
        bom = {
            "bomFormat": "CycloneDX",
            "components": [
                {"name": "weights", "hashes": [{"alg": "SHA-256", "content": "aa" * 32}]}
            ],
        }
        (Path(self.tmpdir) / "cyclonedx-mlbom.json").write_text(json.dumps(bom))

    def _mock_llm_response(self, text: str):
        class FakeResponse:
            def read(self_inner):
                return json.dumps({
                    "choices": [{"message": {"content": text}}]
                }).encode()
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *args):
                pass
        return FakeResponse()

    def test_ask_returns_chat_answer(self):
        from squish.squash.chat import ChatSession, ChatAnswer
        self._write_bom()
        session = ChatSession.from_model_dir(
            Path(self.tmpdir), endpoint="http://localhost/v1", model="llama3"
        )
        llm_text = "The SHA-256 is aaaa... according to cyclonedx-mlbom.json:components[0].hashes."
        with patch("urllib.request.urlopen", return_value=self._mock_llm_response(llm_text)):
            answer = session.ask("What is the hash of the weights?")
        self.assertIsInstance(answer, ChatAnswer)
        self.assertEqual(llm_text, answer.text)
        self.assertGreater(len(answer.sources), 0)

    def test_ask_populates_history(self):
        from squish.squash.chat import ChatSession
        self._write_bom()
        session = ChatSession.from_model_dir(
            Path(self.tmpdir), endpoint="http://localhost/v1", model="llama3"
        )
        with patch("urllib.request.urlopen",
                   return_value=self._mock_llm_response("answer 1")):
            session.ask("question 1")
        self.assertEqual(2, len(session._history))  # user + assistant

    def test_clear_history_resets(self):
        from squish.squash.chat import ChatSession
        self._write_bom()
        session = ChatSession.from_model_dir(
            Path(self.tmpdir), endpoint="http://localhost/v1", model="llama3"
        )
        with patch("urllib.request.urlopen",
                   return_value=self._mock_llm_response("answer")):
            session.ask("question")
        self.assertGreater(len(session._history), 0)
        session.clear_history()
        self.assertEqual(0, len(session._history))

    def test_backend_error_returns_error_text(self):
        from squish.squash.chat import ChatSession
        self._write_bom()
        session = ChatSession.from_model_dir(
            Path(self.tmpdir), endpoint="http://localhost/v1", model="llama3"
        )
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            answer = session.ask("any question?")
        self.assertIn("unavailable", answer.text.lower())

    def test_from_model_dir_empty_dir_creates_session(self):
        from squish.squash.chat import ChatSession
        # Empty dir — no artifacts found; session should still be created
        session = ChatSession.from_model_dir(
            Path(self.tmpdir), endpoint="http://localhost/v1", model="llama3"
        )
        self.assertIsNotNone(session)
        self.assertEqual(0, len(session._retriever._chunks))


if __name__ == "__main__":
    unittest.main()
