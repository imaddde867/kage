from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int

    # Wake word / STT
    wake_word: str
    wake_word_model: str
    wake_word_threshold: float
    whisper_model: str

    # TTS
    tts_voice: str
    kittentts_model: str
    kittentts_sample_rate: int
    say_fallback_voice: str

    # Memory
    memory_dir: str

    # User
    user_name: str

    # Audio
    sample_rate: int
    wake_word_chunk_size: int
    record_chunk_size: int
    silence_threshold: int
    silence_duration: float
    max_record_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_env_str("OLLAMA_MODEL", "qwen3:8b"),
        ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 60),
        wake_word=_env_str("WAKE_WORD", "hey jarvis"),
        wake_word_model=_env_str("WAKE_WORD_MODEL", "hey_jarvis"),
        wake_word_threshold=_env_float("WAKE_WORD_THRESHOLD", 0.5),
        whisper_model=_env_str("WHISPER_MODEL", "base"),
        tts_voice=_env_str("TTS_VOICE", "Jasper"),
        kittentts_model=_env_str("KITTENTTS_MODEL", "KittenML/kitten-tts-mini-0.8"),
        kittentts_sample_rate=_env_int("KITTENTTS_SAMPLE_RATE", 24000),
        say_fallback_voice=_env_str("SAY_FALLBACK_VOICE", "Daniel"),
        # Preserve env var name for backwards compatibility; expose better alias.
        memory_dir=_env_str("CHROMA_PERSIST_DIR", "./data/memory"),
        user_name=_env_str("USER_NAME", "Imad"),
        sample_rate=_env_int("SAMPLE_RATE", 16000),
        wake_word_chunk_size=_env_int("WAKE_WORD_CHUNK_SIZE", 1280),
        record_chunk_size=_env_int("RECORD_CHUNK_SIZE", 1024),
        silence_threshold=_env_int("SILENCE_THRESHOLD", 500),
        silence_duration=_env_float("SILENCE_DURATION", 1.5),
        max_record_seconds=_env_int("MAX_RECORD_SECONDS", 30),
    )


_SETTINGS = get_settings()

# Module-level compatibility aliases (legacy imports depend on these names)
OLLAMA_BASE_URL = _SETTINGS.ollama_base_url
OLLAMA_MODEL = _SETTINGS.ollama_model
OLLAMA_TIMEOUT_SECONDS = _SETTINGS.ollama_timeout_seconds

WAKE_WORD = _SETTINGS.wake_word
WAKE_WORD_MODEL = _SETTINGS.wake_word_model
WAKE_WORD_THRESHOLD = _SETTINGS.wake_word_threshold
WHISPER_MODEL = _SETTINGS.whisper_model

TTS_VOICE = _SETTINGS.tts_voice
KITTENTTS_MODEL = _SETTINGS.kittentts_model
KITTENTTS_SAMPLE_RATE = _SETTINGS.kittentts_sample_rate
SAY_FALLBACK_VOICE = _SETTINGS.say_fallback_voice

MEMORY_DIR = _SETTINGS.memory_dir
CHROMA_PERSIST_DIR = _SETTINGS.memory_dir

USER_NAME = _SETTINGS.user_name

SAMPLE_RATE = _SETTINGS.sample_rate
WAKE_WORD_CHUNK_SIZE = _SETTINGS.wake_word_chunk_size
RECORD_CHUNK_SIZE = _SETTINGS.record_chunk_size
SILENCE_THRESHOLD = _SETTINGS.silence_threshold
SILENCE_DURATION = _SETTINGS.silence_duration
MAX_RECORD_SECONDS = _SETTINGS.max_record_seconds
