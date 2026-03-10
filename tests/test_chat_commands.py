from __future__ import annotations

import unittest

from core.chat_commands import parse_slash_command


class SlashCommandTests(unittest.TestCase):
    def test_parse_known_alias(self) -> None:
        command = parse_slash_command("/copy-last-answer")
        assert command is not None
        self.assertEqual(command.name, "copy_last_answer")

    def test_parse_with_argument(self) -> None:
        command = parse_slash_command("/memory now")
        assert command is not None
        self.assertEqual(command.name, "show_memory")
        self.assertEqual(command.argument, "now")

    def test_non_command_returns_none(self) -> None:
        self.assertIsNone(parse_slash_command("hello"))


if __name__ == "__main__":
    unittest.main()
