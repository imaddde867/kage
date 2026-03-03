from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.speaker import _apply_name_pronunciation


def _settings(
    *,
    enabled: bool = True,
    assistant_name: str = "Kage",
    spoken: str = "Kah-geh",
) -> SimpleNamespace:
    return SimpleNamespace(
        tts_name_override_enabled=enabled,
        assistant_name=assistant_name,
        tts_name_pronunciation=spoken,
    )


class SpeakerPronunciationTests(unittest.TestCase):
    def test_replaces_name_case_insensitive(self) -> None:
        text = "kage is online. KAGE heard you."
        self.assertEqual(
            _apply_name_pronunciation(text, _settings()),
            "Kah-geh is online. Kah-geh heard you.",
        )

    def test_keeps_possessive_suffix(self) -> None:
        text = "Kage's memory is ready."
        self.assertEqual(
            _apply_name_pronunciation(text, _settings()),
            "Kah-geh's memory is ready.",
        )

    def test_no_change_when_disabled(self) -> None:
        text = "Kage is online."
        self.assertEqual(
            _apply_name_pronunciation(text, _settings(enabled=False)),
            text,
        )


if __name__ == "__main__":
    unittest.main()
