import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ollama

from query.engine import QueryEngine


class DummyCollection:
    def query(self, query_embeddings, n_results, **kwargs):
        return {"metadatas": [[]], "documents": [[]]}

    def get(self, include=None):
        return {"documents": [], "metadatas": []}


class DummyStore:
    def __init__(self):
        self.collection = DummyCollection()

    def get_graph_context(self, entity: str):
        return []


def _family_for_model(model_name: str) -> str:
    if "embed" in model_name:
        return "nomic-bert"
    if "llama" in model_name:
        return "llama"
    if "qwen" in model_name:
        return "qwen2"
    return "unknown"


def _ollama_list_response(*model_names: str):
    models = []
    for name in model_names:
        family = _family_for_model(name)
        models.append(
            SimpleNamespace(
                model=name,
                details=SimpleNamespace(family=family, families=[family]),
            )
        )
    return SimpleNamespace(models=models)


class QueryEngineModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.store = DummyStore()
        self.context_note = [
            {
                "title": "Project Atlas",
                "path": "/notes/project-atlas.md",
                "content": "Project Atlas milestone notes.",
                "tags": "project,atlas",
            }
        ]

    @patch("query.engine.ollama.list")
    def test_preferred_model_is_selected_when_available(self, mock_list):
        mock_list.return_value = _ollama_list_response(
            "qwen2.5:7b",
            "llama3.2:latest",
            "nomic-embed-text:latest",
        )
        engine = QueryEngine(self.store)

        resolved = engine._resolve_chat_model("qwen2.5:7b")

        self.assertEqual("qwen2.5:7b", resolved)

    @patch("query.engine.ollama.list")
    def test_falls_back_to_llama_when_preferred_missing(self, mock_list):
        mock_list.return_value = _ollama_list_response(
            "llama3.2:latest",
            "nomic-embed-text:latest",
        )
        engine = QueryEngine(self.store)

        resolved = engine._resolve_chat_model("qwen2.5:7b")

        self.assertEqual("llama3.2:latest", resolved)

    @patch("query.engine.ollama.list")
    def test_falls_back_to_first_local_non_embedding_model(self, mock_list):
        mock_list.return_value = _ollama_list_response(
            "qwen3.5:9b",
            "qwen2.5-coder:7b",
            "nomic-embed-text:latest",
        )
        engine = QueryEngine(self.store)

        resolved = engine._resolve_chat_model("qwen2.5:7b")

        self.assertEqual("qwen2.5-coder:7b", resolved)

    @patch("query.engine.ollama.list")
    @patch("query.engine.ollama.chat")
    def test_override_model_takes_precedence(self, mock_chat, mock_list):
        mock_list.return_value = _ollama_list_response(
            "qwen2.5:7b",
            "llama3.2:latest",
            "nomic-embed-text:latest",
        )
        mock_chat.return_value = {"message": {"content": "answer"}}
        engine = QueryEngine(self.store, model="llama3.2:latest")

        with patch.object(engine, "search", return_value=self.context_note):
            answer, model_used = engine.ask_with_model(
                "What does Project Atlas say?",
                model="qwen2.5:7b",
            )

        self.assertEqual("answer", answer)
        self.assertEqual("qwen2.5:7b", model_used)
        self.assertEqual("qwen2.5:7b", mock_chat.call_args.kwargs["model"])

    @patch("query.engine.ollama.list")
    @patch("query.engine.ollama.chat")
    def test_not_found_error_re_resolves_once_and_retries(self, mock_chat, mock_list):
        mock_list.side_effect = [
            _ollama_list_response("qwen2.5:7b", "llama3.2:latest"),
            _ollama_list_response("qwen2.5:7b", "llama3.2:latest"),
        ]
        mock_chat.side_effect = [
            ollama.ResponseError("model 'qwen2.5:7b' not found", status_code=404),
            {"message": {"content": "retried answer"}},
        ]
        engine = QueryEngine(self.store)

        with patch.object(engine, "search", return_value=self.context_note):
            answer, model_used = engine.ask_with_model("Summarize Project Atlas")

        self.assertEqual("retried answer", answer)
        self.assertEqual("llama3.2:latest", model_used)
        self.assertEqual(2, mock_chat.call_count)
        self.assertEqual("qwen2.5:7b", mock_chat.call_args_list[0].kwargs["model"])
        self.assertEqual("llama3.2:latest", mock_chat.call_args_list[1].kwargs["model"])

    @patch("query.engine.ollama.list")
    @patch("query.engine.ollama.chat")
    def test_non_404_chat_error_is_raised_without_fallback(self, mock_chat, mock_list):
        mock_list.return_value = _ollama_list_response("qwen2.5:7b", "llama3.2:latest")
        mock_chat.side_effect = ollama.ResponseError("upstream failure", status_code=500)
        engine = QueryEngine(self.store)

        with patch.object(engine, "search", return_value=self.context_note):
            with self.assertRaises(ollama.ResponseError):
                engine.ask_with_model("Summarize Project Atlas")

        self.assertEqual(1, mock_chat.call_count)


if __name__ == "__main__":
    unittest.main()
