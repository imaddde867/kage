import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from interface import cli


class CliAskModelTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("interface.cli._get_store_and_engine")
    def test_ask_uses_default_resolution_when_no_override_is_provided(self, mock_get):
        mock_engine = MagicMock()
        mock_engine.ask_with_model.return_value = ("Answer text", "qwen2.5:7b")
        mock_get.return_value = (None, mock_engine)

        result = self.runner.invoke(cli.app, ["ask", "What is Atlas?"])

        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("Model used: qwen2.5:7b", result.output)
        mock_engine.ask_with_model.assert_called_once_with("What is Atlas?", model=None)

    @patch("interface.cli._get_store_and_engine")
    def test_ask_passes_model_override_to_engine(self, mock_get):
        mock_engine = MagicMock()
        mock_engine.ask_with_model.return_value = ("Answer text", "qwen2.5:7b")
        mock_get.return_value = (None, mock_engine)

        result = self.runner.invoke(
            cli.app,
            ["ask", "What is Atlas?", "--model", "qwen2.5:7b"],
        )

        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("Model used: qwen2.5:7b", result.output)
        mock_engine.ask_with_model.assert_called_once_with(
            "What is Atlas?",
            model="qwen2.5:7b",
        )


if __name__ == "__main__":
    unittest.main()
