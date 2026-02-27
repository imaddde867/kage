import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.memory import MemoryStore


class MemoryStorePathTests(unittest.TestCase):
    def test_memory_dir_tilde_expands_to_home_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_home, tempfile.TemporaryDirectory() as tmp_cwd:
            previous_home = os.environ.get("HOME")
            previous_cwd = Path.cwd()

            os.environ["HOME"] = tmp_home
            os.chdir(tmp_cwd)

            try:
                with mock.patch(
                    "core.memory.config.get_settings",
                    return_value=SimpleNamespace(memory_dir="~/kage-memory-test"),
                ):
                    store = MemoryStore()

                expected = Path(tmp_home) / "kage-memory-test" / "kage_memory.db"
                self.assertEqual(store.db_path, expected)
                self.assertTrue(expected.parent.is_dir())
            finally:
                os.chdir(previous_cwd)
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home


if __name__ == "__main__":
    unittest.main()
