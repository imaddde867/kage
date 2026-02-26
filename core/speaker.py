from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
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


def _configure_phonemizer_espeak_library() -> None:
    """
    phonemizer sometimes fails to locate Homebrew's espeak shared library on macOS
    even when the `espeak` binary is installed. Try common dylib locations and
    pin the backend library before KittenTTS initializes.
    """
    try:
        from phonemizer.backend import EspeakBackend
    except Exception:
        return

    # If phonemizer already works, leave its resolution untouched.
    try:
        EspeakBackend(language="en-us")
        return
    except Exception:
        pass

    candidates: list[Path] = []
    espeak_bin = shutil.which("espeak") or shutil.which("espeak-ng")
    if espeak_bin:
        lib_dir = Path(espeak_bin).resolve().parent.parent / "lib"
        candidates.extend(sorted(lib_dir.glob("libespeak-ng*.dylib")))
        candidates.extend(sorted(lib_dir.glob("libespeak*.dylib")))
    for lib_dir in (Path("/opt/homebrew/lib"), Path("/usr/local/lib"), Path("/opt/local/lib")):
        candidates.extend(sorted(lib_dir.glob("libespeak-ng*.dylib")))
        candidates.extend(sorted(lib_dir.glob("libespeak*.dylib")))

    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        try:
            EspeakBackend.set_library(candidate_str)
            EspeakBackend(language="en-us")
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", candidate_str)
            return
        except Exception:
            continue

    # Restore default behavior if none of the candidates worked.
    try:
        EspeakBackend.set_library(None)
    except Exception:
        pass


def _resolve_kittentts_model_path(value: str) -> str | None:
    """
    kittentts expects a local ONNX file path. If the configured value is empty
    or not a readable file (for example a Hugging Face repo id), fall back to
    the library default model download by returning None.
    """
    raw = (value or "").strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    if path.is_file():
        return str(path)

    print(
        f"[Speaker] Ignoring KITTENTTS_MODEL='{raw}' (expected local .onnx file path). "
        "Using KittenTTS default model."
    )
    return None


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
            _configure_phonemizer_espeak_library()
            from kittentts import KittenTTS

            model_path = _resolve_kittentts_model_path(self.settings.kittentts_model)
            self._model = KittenTTS(model_path=model_path) if model_path else KittenTTS()
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
            if "espeak" in str(exc).lower():
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
