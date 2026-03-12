from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.backup import create_backup, verify_backup


class BackupTests(unittest.TestCase):
    def test_create_and_verify_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_path = root / ".env"
            memory_dir = root / "data" / "memory"
            memory_dir.mkdir(parents=True)
            env_path.write_text("ASSISTANT_NAME=Kage\n", encoding="utf-8")
            db_path = memory_dir / "kage_memory.db"
            db_path.write_text("sqlite-data", encoding="utf-8")

            settings = SimpleNamespace(
                assistant_name="Kage",
                memory_dir="data/memory",
            )
            archive = create_backup(settings=settings, root=root)
            self.assertTrue(archive.is_file())

            ok, lines = verify_backup(archive)
            self.assertTrue(ok)
            self.assertTrue(any("config/.env" in line for line in lines))
            self.assertTrue(any("memory/kage_memory.db" in line for line in lines))

    def test_verify_detects_missing_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "broken.tar.gz"
            payload = b'{"files":[{"archive_path":"missing.txt","size":1,"sha256":"x"}]}'
            with tarfile.open(archive, "w:gz") as broken:
                info = tarfile.TarInfo("manifest.json")
                info.size = len(payload)
                broken.addfile(info, io.BytesIO(payload))

            ok, lines = verify_backup(archive)
            self.assertFalse(ok)
            self.assertTrue(any("missing.txt" in line for line in lines))
