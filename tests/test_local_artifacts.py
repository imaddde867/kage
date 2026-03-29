from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from connectors.local_artifacts import (
    LocalExtractDocxTool,
    LocalExtractPdfTool,
    LocalExtractSheetTool,
    LocalFindFilesTool,
    LocalReadTextTool,
)


def _settings(*, roots: tuple[str, ...], deny: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        local_artifact_safe_roots=roots,
        local_artifact_deny_roots=deny,
        local_artifact_max_chars=1200,
        local_artifact_max_file_bytes=1024 * 1024,
        local_artifact_find_max_files=5000,
    )


class LocalArtifactToolTests(unittest.TestCase):
    def test_find_files_returns_matches(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "Lasku.pdf"
            target.write_text("dummy", encoding="utf-8")
            with patch("connectors.local_artifacts._config.get", return_value=_settings(roots=(tmpdir,))):
                tool = LocalFindFilesTool()
                result = tool.execute(query="lasku")

        self.assertFalse(result.is_error)
        self.assertIn("Lasku.pdf", result.content)
        self.assertIsNotNone(result.outcome)
        assert result.outcome is not None
        self.assertEqual(result.outcome.status, "ok")

    def test_find_files_blocks_outside_safe_roots(self) -> None:
        with TemporaryDirectory() as safe_dir, TemporaryDirectory() as outside_dir:
            outside_path = str(Path(outside_dir))
            with patch("connectors.local_artifacts._config.get", return_value=_settings(roots=(safe_dir,))):
                tool = LocalFindFilesTool()
                result = tool.execute(query="anything", directory=outside_path)

        self.assertTrue(result.is_error)
        self.assertIn("outside safe roots", result.content.lower())
        self.assertIsNotNone(result.outcome)
        assert result.outcome is not None
        self.assertEqual(result.outcome.status, "blocked")

    def test_read_text_reads_file_contents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "specs.txt"
            path.write_text("cpu: m4\nram: 32gb", encoding="utf-8")
            with patch("connectors.local_artifacts._config.get", return_value=_settings(roots=(tmpdir,))):
                tool = LocalReadTextTool()
                result = tool.execute(path=str(path))

        self.assertFalse(result.is_error)
        self.assertIn("cpu: m4", result.content.lower())

    def test_read_text_blocks_sensitive_filename(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / ".env"
            path.write_text("TOKEN=secret", encoding="utf-8")
            with patch("connectors.local_artifacts._config.get", return_value=_settings(roots=(tmpdir,))):
                tool = LocalReadTextTool()
                result = tool.execute(path=str(path))

        self.assertTrue(result.is_error)
        self.assertIn("blocked sensitive filename", result.content.lower())

    def test_extract_pdf_reports_missing_dependency(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.pdf"
            path.write_text("not-a-real-pdf", encoding="utf-8")
            with patch("connectors.local_artifacts._PdfReader", None), patch(
                "connectors.local_artifacts._config.get",
                return_value=_settings(roots=(tmpdir,)),
            ):
                tool = LocalExtractPdfTool()
                result = tool.execute(path=str(path))

        self.assertTrue(result.is_error)
        self.assertIn("pypdf", result.content.lower())

    def test_extract_docx_reports_missing_dependency(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.docx"
            path.write_text("not-a-real-docx", encoding="utf-8")
            with patch("connectors.local_artifacts._DocxDocument", None), patch(
                "connectors.local_artifacts._config.get",
                return_value=_settings(roots=(tmpdir,)),
            ):
                tool = LocalExtractDocxTool()
                result = tool.execute(path=str(path))

        self.assertTrue(result.is_error)
        self.assertIn("python-docx", result.content.lower())

    def test_extract_sheet_reads_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "specs.csv"
            path.write_text("part,value\ncpu,m4\nram,32gb\n", encoding="utf-8")
            with patch("connectors.local_artifacts._config.get", return_value=_settings(roots=(tmpdir,))):
                tool = LocalExtractSheetTool()
                result = tool.execute(path=str(path), max_rows=10)

        self.assertFalse(result.is_error)
        self.assertIn("part | value", result.content.lower())
        self.assertIn("cpu | m4", result.content.lower())


if __name__ == "__main__":
    unittest.main()
