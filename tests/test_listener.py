"""Tests for ListenerService STT backend selection."""
from __future__ import annotations

import unittest

from core.listener import _normalize_stt_backend


class NormalizeSTTBackendTests(unittest.TestCase):
    def test_apple_aliases(self) -> None:
        for value in ("apple", "macos", "siri", "native", "APPLE", "Siri"):
            self.assertEqual(_normalize_stt_backend(value), "apple", msg=value)

    def test_whisper_aliases(self) -> None:
        for value in ("whisper", "faster_whisper", "faster-whisper", "WHISPER"):
            self.assertEqual(_normalize_stt_backend(value), "whisper", msg=value)

    def test_unknown_defaults_to_apple(self) -> None:
        self.assertEqual(_normalize_stt_backend("something_random"), "apple")

    def test_empty_defaults_to_apple(self) -> None:
        self.assertEqual(_normalize_stt_backend(""), "apple")


if __name__ == "__main__":
    unittest.main()
