from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import core.cli
import main as entrypoint


_SETTINGS = SimpleNamespace(
    llm_backend="fake",
    mlx_model="fake/model",
    agent_enabled=False,
    second_brain_enabled=False,
    text_mode_tts_enabled=False,
    memory_dir="./data/memory",
)


class CliTests(unittest.TestCase):
    def test_chat_plain_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch("core.cli.launch_plain_chat") as plain:
            rc = core.cli.main(["chat", "--plain"])
        self.assertEqual(rc, 0)
        plain.assert_called_once()

    def test_voice_is_default_command(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch("core.cli.launch_voice") as voice:
            rc = core.cli.main([])
        self.assertEqual(rc, 0)
        voice.assert_called_once()

    def test_global_help_is_not_rewritten_to_voice(self) -> None:
        self.assertEqual(core.cli.normalize_legacy_argv(["--help"]), ["--help"])

    def test_bench_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch("core.cli.launch_bench") as bench:
            rc = core.cli.main(["bench"])
        self.assertEqual(rc, 0)
        bench.assert_called_once()

    def test_main_py_text_shim_uses_chat(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch("core.cli.launch_textual_chat") as chat:
            rc = entrypoint.main(["--text"])
        self.assertEqual(rc, 0)
        chat.assert_called_once()

    def test_main_without_argv_uses_sys_argv(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.launch_textual_chat"
        ) as chat, patch("core.cli.sys.argv", ["kage", "--text"]):
            rc = core.cli.main()
        self.assertEqual(rc, 0)
        chat.assert_called_once()

    def test_textual_import_error_returns_nonzero(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.launch_textual_chat", side_effect=ImportError("missing textual")
        ), patch("builtins.print"):
            rc = core.cli.main(["chat"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
