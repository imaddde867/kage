import unittest
from unittest.mock import patch

from api import server


class ApiAskModelContractTests(unittest.TestCase):
    @patch.object(server, "engine")
    def test_ask_without_model_override_remains_compatible(self, mock_engine):
        mock_engine.ask_with_model.return_value = ("Answer text", "qwen2.5:7b")
        mock_engine.search.return_value = [
            {"title": "Atlas", "path": "/notes/atlas.md", "tags": "project"}
        ]

        response = server.ask(server.AskRequest(question="What is Atlas?"))

        self.assertEqual("Answer text", response["answer"])
        self.assertEqual("qwen2.5:7b", response["model_used"])
        self.assertEqual(1, len(response["sources"]))
        mock_engine.ask_with_model.assert_called_once_with("What is Atlas?", model=None)

    @patch.object(server, "engine")
    def test_ask_with_model_override_passes_through_and_returns_model(self, mock_engine):
        mock_engine.ask_with_model.return_value = ("Answer text", "qwen2.5:7b")
        mock_engine.search.return_value = []

        response = server.ask(
            server.AskRequest(question="What is Atlas?", model="qwen2.5:7b")
        )

        self.assertEqual("qwen2.5:7b", response["model_used"])
        mock_engine.ask_with_model.assert_called_once_with(
            "What is Atlas?",
            model="qwen2.5:7b",
        )


if __name__ == "__main__":
    unittest.main()
