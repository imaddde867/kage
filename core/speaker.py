from __future__ import annotations

import subprocess
from typing import Any

import numpy as np
import sounddevice as sd

import config

# KittenTTS voice map — named aliases for readability in .env
# These correspond to the expr-voice-X-m/f presets.
VOICE_MAP = {
    "Jasper": "expr-voice-2-m",
    "Hugo": "expr-voice-3-m",
    "Bruno": "expr-voice-4-m",
    "Leo": "expr-voice-5-m",
    "Bella": "expr-voice-2-f",
    "Luna": "expr-voice-3-f",
    "Rosie": "expr-voice-4-f",
    "Kiki": "expr-voice-5-f",
}


def _resolve_voice(name: str) -> str:
    if name.startswith("expr-voice"):
        return name
    return VOICE_MAP.get(name, "expr-voice-2-m")


class SpeakerService:
    def __init__(self, *, settings: config.Settings | None = None) -> None:
        self.settings = settings or config.get_settings()
        self._model: Any | None = None
        self._use_kittentts = False
        self._load_attempted = False

    def load_engine(self) -> None:
        if self._load_attempted:
            return

        self._load_attempted = True
        print("[Speaker] Loading KittenTTS model...")

        try:
            from kittentts import KittenTTS

            self._model = KittenTTS(self.settings.kittentts_model)
            self._use_kittentts = True
            print(
                f"[Speaker] KittenTTS ready. Voice: {self.settings.tts_voice} "
                f"→ {_resolve_voice(self.settings.tts_voice)}"
            )
        except ImportError:
            print("[Speaker] KittenTTS not installed. Run: pip install kittentts")
            print("[Speaker] Also make sure espeak is installed: brew install espeak")
            print("[Speaker] Falling back to macOS say.")
        except Exception as exc:
            print(f"[Speaker] KittenTTS failed to load: {exc}")
            print("[Speaker] If you see espeak errors, run: brew install espeak")
            print("[Speaker] Falling back to macOS say.")

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return text.replace("*", "").replace("#", "").replace("`", "").replace("_", "")

    def _speak_kittentts(self, text: str) -> bool:
        if not self._use_kittentts or self._model is None:
            return False

        try:
            voice = _resolve_voice(self.settings.tts_voice)
            audio = self._model.generate(text=text, voice=voice, speed=1.0)
            sd.play(np.asarray(audio, dtype=np.float32), samplerate=self.settings.kittentts_sample_rate)
            sd.wait()
            return True
        except Exception as exc:
            print(f"[Speaker] KittenTTS generate error: {exc}")
            print("[Speaker] Falling back to macOS say.")
            return False

    def _speak_macos_say(self, text: str) -> None:
        try:
            subprocess.run(["say", "-v", self.settings.say_fallback_voice, text], check=True)
        except FileNotFoundError:
            print("[Speaker] macOS say command not found.")
        except subprocess.CalledProcessError as exc:
            print(f"[Speaker] macOS say failed ({exc.returncode}).")

    def speak(self, text: str) -> None:
        """
        Speak text aloud using KittenTTS neural TTS.
        Falls back to macOS say if KittenTTS isn't available.
        """
        print(f"\n[Kage]: {text}\n")
        clean = self._sanitize_text(text)

        if not self._load_attempted:
            self.load_engine()

        if self._speak_kittentts(clean):
            return

        self._speak_macos_say(clean)


_DEFAULT_SPEAKER: SpeakerService | None = None


def get_default_speaker() -> SpeakerService:
    global _DEFAULT_SPEAKER
    if _DEFAULT_SPEAKER is None:
        _DEFAULT_SPEAKER = SpeakerService()
    return _DEFAULT_SPEAKER


def speak(text: str) -> None:
    get_default_speaker().speak(text)


__all__ = ["SpeakerService", "get_default_speaker", "speak", "VOICE_MAP"]
