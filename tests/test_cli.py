from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import core.cli
import main as entrypoint


_SETTINGS = SimpleNamespace(
    llm_backend="fake",
    mlx_model="fake/model",
    agent_enabled=False,
    agent_policy_mode="strict",
    agent_approval_required_tiers=("moderate_change", "high_impact"),
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

    def test_backup_create_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_backup_create", return_value=0
        ) as backup:
            rc = core.cli.main(["backup", "create"])
        self.assertEqual(rc, 0)
        backup.assert_called_once()

    def test_backup_verify_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_backup_verify", return_value=0
        ) as verify:
            rc = core.cli.main(["backup", "verify", "archive.tar.gz"])
        self.assertEqual(rc, 0)
        verify.assert_called_once_with(archive="archive.tar.gz")

    def test_approvals_list_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_approvals_list", return_value=0
        ) as approvals_list:
            rc = core.cli.main(["approvals", "list"])
        self.assertEqual(rc, 0)
        approvals_list.assert_called_once()

    def test_approvals_grant_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_approvals_grant", return_value=0
        ) as grant:
            rc = core.cli.main(["approvals", "grant", "tool", "shell"])
        self.assertEqual(rc, 0)
        grant.assert_called_once()

    def test_approvals_revoke_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_approvals_revoke", return_value=0
        ) as revoke:
            rc = core.cli.main(["approvals", "revoke", "tool", "shell"])
        self.assertEqual(rc, 0)
        revoke.assert_called_once()

    def test_service_install_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_service_install", return_value=0
        ) as install:
            rc = core.cli.main(["service", "install"])
        self.assertEqual(rc, 0)
        install.assert_called_once()

    def test_service_status_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_service_status", return_value=0
        ) as status:
            rc = core.cli.main(["service", "status"])
        self.assertEqual(rc, 0)
        status.assert_called_once()

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

    def test_doctor_agent_flag_dispatch(self) -> None:
        with patch("core.cli.config.get", return_value=_SETTINGS), patch(
            "core.cli.run_doctor"
        ) as doctor:
            rc = core.cli.main(["doctor", "--agent"])
        self.assertEqual(rc, 0)
        doctor.assert_called_once()

    def test_collect_agent_doctor_checks_uses_exact_count_entries(self) -> None:
        settings = SimpleNamespace(
            agent_policy_mode="strict",
            agent_approval_required_tiers=("moderate_change", "high_impact"),
            memory_dir="./data/memory",
        )

        class _Store:
            def __init__(self, _db_path):
                pass

            def count_entries(self) -> int:
                return 321

            def list_entries(self, limit: int = 200):  # pragma: no cover - should never be called
                raise AssertionError(f"list_entries should not be used (limit={limit})")

        with patch("core.platform.storage.ApprovalStore", _Store):
            checks = core.cli.collect_agent_doctor_checks(settings=settings)
        detail = {check.name: check.detail for check in checks}
        self.assertEqual(detail["agent_approvals_store"], "321 persisted approval(s)")

    def test_format_doctor_report_flags_backend_model_mismatch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = SimpleNamespace(
                llm_backend="mlx",
                mlx_model="mlx-community/Qwen3.5-4B-MLX-4bit",
                agent_enabled=True,
                second_brain_enabled=False,
                text_mode_tts_enabled=False,
                memory_dir=f"{tmpdir}/memory",
                stt_backend="apple",
            )
            with patch("core.cli._module_available", return_value=False), patch(
                "core.cli.shutil.which", return_value=None
            ):
                report = core.cli.format_doctor_report(settings=settings)
        self.assertIn("model_compat: warning", report)
        self.assertIn("voice_stack: warning", report)
        self.assertIn("apple_automation: warning", report)


if __name__ == "__main__":
    unittest.main()
