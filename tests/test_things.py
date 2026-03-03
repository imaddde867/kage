"""Tests for the Things 3 connector."""
from __future__ import annotations

import unittest
from unittest import mock

from connectors import things


class ThingsConnectorTests(unittest.TestCase):
    def test_returns_empty_when_osascript_fails(self) -> None:
        with mock.patch("connectors.things.run_osascript", return_value=""):
            result = things.get_context()
        self.assertEqual(result, "")

    def test_formats_tasks_as_bulleted_section(self) -> None:
        raw = "Buy milk\nFinish report\n"
        with mock.patch("connectors.things.run_osascript", return_value=raw):
            result = things.get_context()
        self.assertIn("[Things 3", result)
        self.assertIn("Buy milk", result)
        self.assertIn("Finish report", result)

    def test_returns_empty_for_whitespace_only_output(self) -> None:
        with mock.patch("connectors.things.run_osascript", return_value="   \n  \n"):
            result = things.get_context()
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
