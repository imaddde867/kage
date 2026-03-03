from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

import config

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

_DEFAULT_TTS_SPEED = 1.0
_TTS_CHUNK_MAX_CHARS = 180
_TTS_CHUNK_GAP_SECONDS = 0.12
_KITTENTTS_PROFILE_SPECS = {
    "nano": None,  # library default loader (currently nano)
    "mini": ("KittenML/kitten-tts-mini-0.8", "kitten_tts_mini_v0_8.onnx"),
}
_FALLBACK_TTS_BACKEND = "kittentts"  # used when TTS_BACKEND is set to an unknown value


def _normalize_tts_backend(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"", "default", "kittentts", "kitten"}:
        return "kittentts"
    if raw in {"macos", "say", "macos_say", "apple", "siri"}:
        return "macos_say"
    return raw


def _resolve_voice(name: str) -> str:
    if name.startswith("expr-voice"):
        return name
    return VOICE_MAP.get(name, "expr-voice-2-m")


def _sanitize_text(text: str) -> str:
    text = text.replace("*", "").replace("#", "").replace("`", "").replace("_", "")
    text = text.replace("...", ".").replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_tts_chunks(text: str, max_chars: int = _TTS_CHUNK_MAX_CHARS) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        subparts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", sentence) if p.strip()]
        if not subparts:
            subparts = [sentence]

        for subpart in subparts:
            if len(subpart) <= max_chars:
                chunks.append(subpart)
                continue

            words = subpart.split()
            current: list[str] = []
            current_len = 0
            for word in words:
                word_len = len(word)
                projected_len = current_len + (1 if current else 0) + word_len
                if current and projected_len > max_chars:
                    chunks.append(" ".join(current))
                    current = [word]
                    current_len = word_len
                else:
                    current.append(word)
                    current_len = projected_len
            if current:
                chunks.append(" ".join(current))

    return [chunk for chunk in chunks if chunk]


def _configure_phonemizer_espeak_library() -> None:
    """
    phonemizer sometimes fails to locate Homebrew's espeak shared library on macOS
    even when the `espeak` binary is installed. Try common dylib locations and
    set the backend library before KittenTTS initializes.
    """
    try:
        from phonemizer.backend import EspeakBackend
    except Exception:
        return

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

    try:
        EspeakBackend.set_library(None)
    except Exception:
        pass


def _resolve_kittentts_model_path(value: str) -> str | None:
    """
    kittentts expects a local ONNX file path when model_path is provided.
    If the configured value is empty or not a readable file (for example an HF
    repo id), return None so KittenTTS uses its default model download path.
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


def _normalize_kittentts_profile(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"", "default"}:
        return "nano"
    if raw in {"nano", "nano-0.1"}:
        return "nano"
    if raw in {"mini", "mini-0.8"}:
        return "mini"
    return raw


def _create_kittentts_model(settings: config.Settings) -> Any:
    from kittentts import KittenTTS

    model_path = _resolve_kittentts_model_path(settings.kittentts_model)
    if model_path:
        return KittenTTS(model_path=model_path)

    profile = _normalize_kittentts_profile(settings.kittentts_profile)
    profile_spec = _KITTENTTS_PROFILE_SPECS.get(profile)
    if profile_spec is None:
        if profile == "nano":
            return KittenTTS()
        raise ValueError(
            f"Unsupported KITTENTTS_PROFILE='{settings.kittentts_profile}'. "
            f"Supported: {', '.join(sorted(_KITTENTTS_PROFILE_SPECS))}"
        )

    repo_id, model_filename = profile_spec
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id, model_filename)
    voices_path = hf_hub_download(repo_id, "voices.npz")
    return KittenTTS(model_path=model_path, voices_path=voices_path)


class SpeakerService:
    def __init__(self, *, settings: config.Settings | None = None) -> None:
        self.settings = settings or config.get_settings()
        self._model: Any | None = None
        self._use_kittentts = False
        self._load_attempted = False
        self._tts_backend = _normalize_tts_backend(self.settings.tts_backend)

    def load_engine(self) -> None:
        if self._load_attempted:
            return

        self._load_attempted = True

        if self._tts_backend == "macos_say":
            print(f"[Speaker] Using macOS say voice: {self.settings.say_fallback_voice}")
            return

        if self._tts_backend != "kittentts":
            print(
                f"[Speaker] Unknown TTS_BACKEND='{self.settings.tts_backend}'. "
                f"Using '{_FALLBACK_TTS_BACKEND}'."
            )
            self._tts_backend = _FALLBACK_TTS_BACKEND

        print("[Speaker] Loading KittenTTS model...")

        try:
            _configure_phonemizer_espeak_library()
            self._model = _create_kittentts_model(self.settings)
            self._use_kittentts = True
            print(
                f"[Speaker] KittenTTS ready. Profile: {_normalize_kittentts_profile(self.settings.kittentts_profile)} "
                f"Voice: {self.settings.tts_voice} "
                f"→ {_resolve_voice(self.settings.tts_voice)}"
            )
        except ImportError:
            print("[Speaker] KittenTTS not installed. Run: pip install kittentts")
            print("[Speaker] Also make sure espeak is installed: brew install espeak")
            print("[Speaker] Falling back to macOS say.")
            self._tts_backend = "macos_say"
        except Exception as exc:
            print(f"[Speaker] KittenTTS failed to load: {exc}")
            if "espeak" in str(exc).lower():
                print("[Speaker] If you see espeak errors, run: brew install espeak")
            print("[Speaker] Falling back to macOS say.")
            self._tts_backend = "macos_say"

    def _speak_kittentts(self, text: str) -> bool:
        if not self._use_kittentts or self._model is None:
            return False

        voice = _resolve_voice(self.settings.tts_voice)
        sample_rate = self.settings.kittentts_sample_rate
        chunks = _split_tts_chunks(text)
        if not chunks:
            return True

        rendered_chunks: list[np.ndarray] = []
        for chunk in chunks:
            try:
                audio = self._model.generate(text=chunk, voice=voice, speed=_DEFAULT_TTS_SPEED)
            except Exception as exc:
                print(f"[Speaker] KittenTTS generate error on chunk: {exc}")
                return False

            rendered_chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if not rendered_chunks:
            return True

        if len(rendered_chunks) == 1:
            combined = rendered_chunks[0]
        else:
            gap = np.zeros(int(sample_rate * _TTS_CHUNK_GAP_SECONDS), dtype=np.float32)
            parts: list[np.ndarray] = []
            for index, chunk_audio in enumerate(rendered_chunks):
                parts.append(chunk_audio)
                if index < len(rendered_chunks) - 1:
                    parts.append(gap)
            combined = np.concatenate(parts)

        try:
            sd.play(combined, samplerate=sample_rate)
            sd.wait()
            return True
        except Exception as exc:
            print(f"[Speaker] Audio playback error: {exc}")
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
        Speak text aloud using the selected TTS backend.
        """
        print(f"\n[Kage]: {text}\n")
        clean = _sanitize_text(text)
        if not clean:
            return

        if not self._load_attempted:
            self.load_engine()

        if self._tts_backend == "macos_say":
            self._speak_macos_say(clean)
            return

        if self._speak_kittentts(clean):
            return

        print("[Speaker] Falling back to macOS say.")
        self._tts_backend = "macos_say"
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
