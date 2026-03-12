from __future__ import annotations

import plistlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core import service


class ServiceTests(unittest.TestCase):
    def test_install_writes_plist_and_bootstraps_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            home = Path(tmpdir) / "home"
            root.mkdir(parents=True)
            home.mkdir(parents=True)
            (root / "main.py").write_text("print('kage')\n", encoding="utf-8")

            calls: list[list[str]] = []

            def runner(cmd: list[str], **kwargs):
                _ = kwargs
                calls.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("core.service.sys.platform", "darwin"):
                ok, _ = service.install_voice_service(
                    project_root=root,
                    home=home,
                    python_executable="/usr/bin/python3",
                    uid=501,
                    runner=runner,
                )

            self.assertTrue(ok)
            plist_path = service.voice_service_plist_path(home=home)
            self.assertTrue(plist_path.exists())
            with plist_path.open("rb") as handle:
                payload = plistlib.load(handle)
            self.assertEqual(payload["ProgramArguments"][0], "/usr/bin/python3")
            self.assertEqual(
                Path(payload["ProgramArguments"][1]).resolve(),
                (root / "main.py").resolve(),
            )
            self.assertEqual(payload["ProgramArguments"][2], "voice")
            self.assertIn(["launchctl", "bootstrap", "gui/501", str(plist_path)], calls)
            self.assertIn(["launchctl", "kickstart", "-k", "gui/501/com.kage.voice"], calls)

    def test_status_parses_pid_when_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            plist_path = service.voice_service_plist_path(home=home)
            plist_path.parent.mkdir(parents=True, exist_ok=True)
            plist_path.write_text("plist", encoding="utf-8")

            def runner(cmd: list[str], **kwargs):
                _ = kwargs
                self.assertEqual(cmd[:2], ["launchctl", "print"])
                return SimpleNamespace(returncode=0, stdout="pid = 4242", stderr="")

            with patch("core.service.sys.platform", "darwin"):
                status = service.voice_service_status(home=home, uid=501, runner=runner)

            self.assertTrue(status.installed)
            self.assertTrue(status.loaded)
            self.assertEqual(status.pid, 4242)

    def test_stop_treats_not_loaded_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"

            def runner(cmd: list[str], **kwargs):
                _ = (cmd, kwargs)
                return SimpleNamespace(returncode=1, stdout="", stderr="Could not find service")

            with patch("core.service.sys.platform", "darwin"):
                ok, message = service.stop_voice_service(home=home, uid=501, runner=runner)

            self.assertTrue(ok)
            self.assertIn("Stopped", message)


if __name__ == "__main__":
    unittest.main()
